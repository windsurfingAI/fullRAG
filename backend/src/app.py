import os
import json
import redis # [NOUVEAU] Pour la mémoire court terme
import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from google import genai

# Dépendances pour la recherche Hybride (LangChain & BM25)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Dépendance pour le Reranking
from sentence_transformers import CrossEncoder

# import query rewriter
from query_rewriter import rewrite_query


app = FastAPI(title="Local & Free RAG Orchestrator (Hybrid Search + Memory)")

# ---------------------------------------------------------
# [NOUVEAU] INITIALISATION DE REDIS (Mémoire de session Docker)
# Dans un docker-compose, le host sera généralement 'redis'
# ---------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    WINDOW_SIZE = 4 # On garde les 4 derniers échanges
except Exception as e:
    print(f"⚠️ Erreur de connexion à Redis : {e}")
    redis_client = None

# 1. Initialisation de Gemini (Cloud)
try:
    gemini_client = genai.Client()
except Exception:
    gemini_client = None

# 2. Configuration des chemins et modèles
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "documents"
MODEL_NAME = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'}
)

# 3. Initialisation du Retriever Hybride (ChromaDB + BM25)
# [Le code du retriever reste identique à ton original...]
print("🔄 Initialisation de la recherche hybride (ChromaDB + BM25)...")
vectorstore = Chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION_NAME, embedding_function=embeddings)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
all_data = vectorstore.get()

if not all_data or not all_data.get('documents'):
    all_docs = []
else:
    all_docs = [Document(page_content=t, metadata=m) for t, m in zip(all_data['documents'], all_data['metadatas'])]

bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
if bm25_retriever:
    bm25_retriever.k = 25
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.3, 0.7])
else:
    hybrid_retriever = chroma_retriever

# 4. Initialisation du Reranker (Local)
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ---------------------------------------------------------
# [NOUVEAU] SCHÉMAS DE DONNÉES MIS À JOUR
# ---------------------------------------------------------
class UserProfile(BaseModel):
    # Représente les claims extraits du JWT (SSO) en entreprise
    name: str = "Utilisateur Anonyme"
    role: str = "Collaborateur"
    department: str = "Général"
    clearance: str = "Standard"

class ChatRequest(BaseModel):
    session_id: str  # [NOUVEAU] Obligatoire pour retrouver la conversation
    query: str
    provider: str = "gemini" 
    top_k: int = 5
    user_profile: UserProfile = UserProfile() # [NOUVEAU] Mémoire Long terme (SSO)

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    try:
        # ---------------------------------------------------------
        # ÉTAPE 1 : Récupération de l'historique Redis (Déplacé ici !)
        # ---------------------------------------------------------
        chat_history = []
        history_text = "Aucun historique pour le moment."
        redis_key = f"session:{request.session_id}"
        
        if redis_client:
            history_json = redis_client.get(redis_key)
            if history_json:
                chat_history = json.loads(history_json)
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # ---------------------------------------------------------
        # ÉTAPE 2 : Réécriture de la requête
        # ---------------------------------------------------------
        search_query = rewrite_query(
            query=request.query, 
            chat_history=chat_history, 
            provider=request.provider, 
            gemini_client=gemini_client
        )
        print(f"🔍 Requête originale : {request.query}")
        print(f"✨ Requête réécrite  : {search_query}")

        # ---------------------------------------------------------
        # ÉTAPE 3 : Recherche Hybride (sur la requête réécrite)
        # ---------------------------------------------------------
        retrieved_docs = hybrid_retriever.invoke(search_query)
        documents = [doc.page_content for doc in retrieved_docs]
        
        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document trouvé dans la base.")

        # ---------------------------------------------------------
        # ÉTAPE 4 : Reranking Local (sur la requête réécrite)
        # ---------------------------------------------------------
        pairs = [[search_query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs[:6]]

        # ---------------------------------------------------------
        # ÉTAPE 5 : Mémoire Long Terme (Injection Profil SSO)
        # ---------------------------------------------------------
        system_context = f"""Tu es un assistant RAG d'entreprise expert.
INFORMATIONS SUR L'UTILISATEUR ACTUEL :
- Nom : {request.user_profile.name}
- Rôle : {request.user_profile.role}
- Département : {request.user_profile.department}
Adapte tes réponses en fonction de ce profil métier."""

        # ---------------------------------------------------------
        # ÉTAPE 6 : Construction du prompt final
        # ---------------------------------------------------------
        context = "\n\n".join(reranked_docs)
        prompt = f"""{system_context}
        
HISTORIQUE DE LA CONVERSATION :
{history_text}

CONTEXTE DOCUMENTAIRE (RAG) :
{context}

INSTRUCTION :
Réponds à la nouvelle question en te basant UNIQUEMENT sur le contexte documentaire fourni. 
Si la réponse ne s'y trouve pas, dis-le. Prends en compte l'historique si la question contient des pronoms.

Nouvelle question : {request.query}
Réponse :"""

        # ---------------------------------------------------------
        # ÉTAPE 7 : Génération de la réponse
        # ---------------------------------------------------------
        if request.provider == "gemini":
            if not gemini_client:
                raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
            response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            answer = response.text

        elif request.provider == "mistral":
            ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            ollama_response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            if ollama_response.status_code == 200:
                answer = ollama_response.json().get("response", "")
            else:
                raise HTTPException(status_code=500, detail="Erreur lors de l'appel à Ollama.")
        else:
            raise HTTPException(status_code=400, detail="Fournisseur non supporté.")

        # ---------------------------------------------------------
        # ÉTAPE 8 : SAUVEGARDE DE LA SESSION
        # ---------------------------------------------------------
        if redis_client:
            # Ajout du nouveau tour de parole
            chat_history.append({"role": "Utilisateur", "content": request.query})
            chat_history.append({"role": "Assistant", "content": answer})
            
            # Application de la fenêtre glissante
            if len(chat_history) > (WINDOW_SIZE * 2):
                chat_history = chat_history[-(WINDOW_SIZE * 2):]
                
            # Sauvegarde avec TTL
            redis_client.set(redis_key, json.dumps(chat_history), ex=7200)

        return ChatResponse(
            answer=answer,
            sources=reranked_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")