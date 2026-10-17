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

# Imports locaux
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_rewriter import rewrite_query
from anonymizer import sanitize_input

# --- LANGFUSE IMPORTS ---
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

# --- Nemo the fish ---
from nemoguardrails import LLMRails, RailsConfig

app = FastAPI(title="Local & Free RAG Orchestrator (Hybrid Search + Memory + Security)")

# ---------------------------------------------------------
# INITIALISATION DE NEMO GUARDRAILS (AU DÉMARRAGE)
# ---------------------------------------------------------
print("🛡️ Chargement de NeMo Guardrails...")
try:
    # On pointe vers le dossier qui contient config.yml et cinema.co
    rails_config = RailsConfig.from_path("./guardrails")
    rails = LLMRails(rails_config)
    print("✅ NeMo Guardrails opérationnel !")
except Exception as e:
    print(f"⚠️ Erreur d'initialisation de NeMo Guardrails : {e}")
    rails = None

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
    trace_id: str 


# ---------------------------------------------------------
# SOUS-FONCTIONS TRACÉES PAR LANGFUSE (V2)
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
        model_name = "gemini-2.5-flash"
        langfuse_context.update_current_observation(model=model_name)
        
        if not gemini_client:
            raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
        response = gemini_client.models.generate_content(model=model_name, contents=prompt)
        
        if hasattr(response, "usage_metadata") and response.usage_metadata:
             langfuse_context.update_current_observation(
                 usage={
                     "input": response.usage_metadata.prompt_token_count,
                     "output": response.usage_metadata.candidates_token_count,
                     "total": response.usage_metadata.total_token_count
                 }
             )
        
        return response.text
        
    if provider == "mistral":
        model_name = "mistral"
        langfuse_context.update_current_observation(model=model_name)
        
        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ollama_response = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        if ollama_response.status_code == 200:
             response_json = ollama_response.json()
             langfuse_context.update_current_observation(
                 usage={
                     "input": response_json.get("prompt_eval_count", 0),
                     "output": response_json.get("eval_count", 0),
                     "total": response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0)
                 }
             )
             return response_json.get("response", "")
        raise HTTPException(status_code=500, detail="Erreur lors de l'appel à Ollama.")
        
    raise HTTPException(status_code=400, detail="Fournisseur non supporté.")


# ---------------------------------------------------------
# ENDPOINT PRINCIPAL
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
@observe(name="rag_cinema_pipeline")
async def handle_chat(request: ChatRequest):
    try:
        trace_id = langfuse_context.get_current_trace_id()
        
        # =========================================================
        # ÉTAPE 0.1 : PRIVACY (Presidio)
        # =========================================================
        clean_query, pii_detected = sanitize_input(request.query)
        
        langfuse_context.update_current_trace(
            session_id=request.session_id,
            user_id=request.user_profile.name,
            input=clean_query,  # On logge la question nettoyée
            metadata={
                "role": request.user_profile.role,
                "department": request.user_profile.department,
                "provider": request.provider,
                "pii_masked": pii_detected
            }
        )

        # =========================================================
        # ÉTAPE 0.2 : SÉCURITÉ (NeMo Guardrails)
        # =========================================================
        if rails:
            # On vérifie l'intention avant d'activer le RAG
            nemo_response = await rails.generate_async(messages=[
                {"role": "user", "content": clean_query}
            ])
            
            # Si NeMo déclenche la phrase de refus (définie dans cinema.co)
            if "strictement dédié au cinéma" in nemo_response["content"]:
                langfuse_context.update_current_trace(
                    output=nemo_response["content"], 
                    tags=["blocked_by_guardrails"]
                )
                return ChatResponse(
                    answer=nemo_response["content"],
                    sources=["⚠️ Requête bloquée (Hors-sujet ou Injection de prompt)"],
                    trace_id=trace_id
                )

        # =========================================================
        # ÉTAPE 1 à 8 : RAG CLASSIQUE
        # =========================================================
        chat_history = []
        history_text = "Aucun historique pour le moment."
        redis_key = f"session:{request.session_id}"
        
        if redis_client:
            history_json = redis_client.get(redis_key)
            if history_json:
                chat_history = json.loads(history_json)
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # On utilise "clean_query" partout désormais
        search_query = run_query_rewrite(clean_query, chat_history, request.provider)

        documents = run_hybrid_retrieval(search_query)
        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document trouvé dans la base.")

        reranked_docs = run_reranking(search_query, documents)

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

Nouvelle question : {clean_query}
Réponse :"""

        answer = run_llm_generation(request.provider, prompt)
        
        langfuse_context.update_current_trace(output=answer)

        if redis_client:
            chat_history.append({"role": "Utilisateur", "content": clean_query})
            chat_history.append({"role": "Assistant", "content": answer})
            if len(chat_history) > (WINDOW_SIZE * 2):
                chat_history = chat_history[-(WINDOW_SIZE * 2):]
            redis_client.set(redis_key, json.dumps(chat_history), ex=7200)

        # Étape 9 : Envoi d'un score (démo)
        langfuse.score(
            trace_id=trace_id,
            name="qualite_rag_demo",
            value=1, 
            comment="Note automatique générée par le backend"
        )
        langfuse.flush() 

        return ChatResponse(
            answer=answer,
            sources=reranked_docs,
            trace_id=trace_id
        )

    except HTTPException:
        raise
    except Exception as e:
        langfuse_context.update_current_trace(level="ERROR", status_message=str(e))
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")