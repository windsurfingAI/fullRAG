import os
import json
import redis
import requests
from fastapi import FastAPI, HTTPException
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

# Correction de l'import pour Docker
try:
    from src.query_rewriter import rewrite_query
except ModuleNotFoundError:
    from query_rewriter import rewrite_query

# --- LANGFUSE IMPORTS ---
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
langfuse = Langfuse()

app = FastAPI(title="Local & Free RAG Orchestrator (Hybrid Search + Memory)")

# ---------------------------------------------------------
# INITIALISATION DE REDIS
# ---------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    WINDOW_SIZE = 4
except Exception as e:
    print(f"⚠️ Erreur de connexion à Redis : {e}")
    redis_client = None

# 1. Initialisation de Gemini
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

# 3. Initialisation du Retriever Hybride
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

# 4. Initialisation du Reranker
reranker = CrossEncoder("BAAI/bge-reranker-base")


class UserProfile(BaseModel):
    name: str = "Utilisateur Anonyme"
    role: str = "Collaborateur"
    department: str = "Général"
    clearance: str = "Standard"

class ChatRequest(BaseModel):
    session_id: str
    query: str
    provider: str = "gemini" 
    top_k: int = 5
    user_profile: UserProfile = UserProfile()

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------------------------------------------------
# SOUS-FONCTIONS TRACÉES PAR LANGFUSE (V5)
# ---------------------------------------------------------

@observe(name="query_rewriter")
def run_query_rewrite(query: str, chat_history: list, provider: str) -> str:
    return rewrite_query(
        query=query, 
        chat_history=chat_history, 
        provider=provider, 
        gemini_client=gemini_client
    )

@observe(name="hybrid_retrieval")
def run_hybrid_retrieval(search_query: str) -> list[str]:
    retrieved_docs = hybrid_retriever.invoke(search_query)
    return [doc.page_content for doc in retrieved_docs]

@observe(name="reranking_cross_encoder")
def run_reranking(search_query: str, documents: list[str]) -> list[str]:
    pairs = [[search_query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:6]]

@observe(as_type="generation", name="llm_generation")
def run_llm_generation(provider: str, prompt: str) -> str:
    if provider == "gemini":
        if not gemini_client:
            raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
        
    if provider == "mistral":
        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ollama_response = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        if ollama_response.status_code == 200:
            return ollama_response.json().get("response", "")
        raise HTTPException(status_code=500, detail="Erreur lors de l'appel à Ollama.")
        
    raise HTTPException(status_code=400, detail="Fournisseur non supporté.")


# ---------------------------------------------------------
# ENDPOINT PRINCIPAL
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
@observe(name="rag_cinema_pipeline")
async def handle_chat(request: ChatRequest):
    try:
        # Étape 1 : Historique Redis
        chat_history = []
        history_text = "Aucun historique pour le moment."
        redis_key = f"session:{request.session_id}"
        
        if redis_client:
            history_json = redis_client.get(redis_key)
            if history_json:
                chat_history = json.loads(history_json)
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # Étape 2 : Réécriture de la requête
        search_query = run_query_rewrite(request.query, chat_history, request.provider)

        # Étape 3 : Recherche Hybride
        documents = run_hybrid_retrieval(search_query)
        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document trouvé dans la base.")

        # Étape 4 : Reranking Local
        reranked_docs = run_reranking(search_query, documents)

        # Étape 5 & 6 : Prompt final
        system_context = f"""Tu es un assistant RAG d'entreprise expert.
INFORMATIONS SUR L'UTILISATEUR ACTUEL :
- Nom : {request.user_profile.name}
- Rôle : {request.user_profile.role}
- Département : {request.user_profile.department}
Adapte tes réponses en fonction de ce profil métier."""

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

        # Étape 7 : Génération de la réponse
        answer = run_llm_generation(request.provider, prompt)

        # Étape 8 : Sauvegarde Redis
        if redis_client:
            chat_history.append({"role": "Utilisateur", "content": request.query})
            chat_history.append({"role": "Assistant", "content": answer})
            if len(chat_history) > (WINDOW_SIZE * 2):
                chat_history = chat_history[-(WINDOW_SIZE * 2):]
            redis_client.set(redis_key, json.dumps(chat_history), ex=7200)


        # Force l'envoi des données à Langfuse
        langfuse.flush() 

        return ChatResponse(
            answer=answer,
            sources=reranked_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")