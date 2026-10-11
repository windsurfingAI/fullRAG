import os
import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

# Dépendances pour la recherche Hybride (LangChain & BM25)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Variables d'environnement
from dotenv import load_dotenv
load_dotenv()

# Dépendance pour le Reranking
from sentence_transformers import CrossEncoder

app = FastAPI(title="Local & Free RAG Orchestrator (Hybrid Search)")

# 1. Initialisation de Gemini (Cloud)
gemini_client = None
if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
    try:
        gemini_client = genai.Client()
    except Exception as e:
        print(f"⚠️ Avertissement Gemini : {e}")

# Configuration Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# 2. Configuration des chemins et modèles
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "documents"
MODEL_NAME = "all-MiniLM-L6-v2"

# Détection sécurisée du device (évite l'erreur CUDA unknown error)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

print(f"⚙️ Périphérique d'inférence embeddings : {device.upper()}")

# Initialisation du modèle d'embedding pour LangChain Chroma
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': device}
)

# 3. Initialisation du Retriever Hybride (ChromaDB + BM25)
print("🔄 Initialisation de la recherche hybride (ChromaDB + BM25)...")

# A. Retriever Dense (ChromaDB)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# B. Retriever Sparse (BM25 - Recherche par mots-clés exacts)
all_data = vectorstore.get()
if not all_data or not all_data.get('documents'):
    print("⚠️ Attention: Aucun document trouvé dans ChromaDB. Assurez-vous d'avoir exécuté ingest_csv.py.")
    all_docs = []
else:
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_data['documents'], all_data['metadatas'])
    ]

bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
if bm25_retriever:
    bm25_retriever.k = 25

# C. Fusion Hybride (RRF)
if bm25_retriever:
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]
    )
else:
    hybrid_retriever = chroma_retriever

# 4. Initialisation du Reranker (Local)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

# Schémas de données
class ChatRequest(BaseModel):
    query: str
    provider: str = "ollama"  # 'ollama', 'gemini', ou 'mistral'
    model: str | None = None
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    try:
        # Étape 1 : Recherche Hybride
        retrieved_docs = hybrid_retriever.invoke(request.query)
        documents = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
        
        # Garde-fou si aucun document n'est retourné
        if not documents:
            return ChatResponse(
                answer="Désolé, aucune information pertinente n'a été trouvée dans la base documentaire.",
                sources=[]
            )

        # Étape 2 : Reranking Local
        pairs = [[request.query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs[:request.top_k]]

        # Étape 3 : Construction du prompt RAG
        context = "\n\n".join(reranked_docs)
        prompt = f"""Réponds à la question en te basant uniquement sur le contexte fourni ci-dessous.
        
Contexte:
{context}

Question: {request.query}
Réponse:"""

        # Étape 4 : Génération selon le fournisseur
        if request.provider == "gemini":
            if not gemini_client:
                raise HTTPException(
                    status_code=500, 
                    detail="Clé API Gemini absente ou invalide dans les variables d'environnement."
                )
            
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            answer = response.text

        elif request.provider in ("ollama", "mistral"):
            target_model = request.model or ("mistral" if request.provider == "mistral" else DEFAULT_OLLAMA_MODEL)
            
            try:
                ollama_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": target_model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=90
                )
                if ollama_response.status_code == 200:
                    answer = ollama_response.json().get("response", "")
                else:
                    raise HTTPException(
                        status_code=ollama_response.status_code, 
                        detail=f"Erreur renvoyée par Ollama : {ollama_response.text}"
                    )
            except requests.exceptions.ConnectionError:
                raise HTTPException(
                    status_code=503, 
                    detail="Impossible de joindre Ollama. Vérifiez qu'il tourne sur le port 11434."
                )
        else:
            raise HTTPException(status_code=400, detail=f"Fournisseur '{request.provider}' non supporté.")

        return ChatResponse(
            answer=answer,
            sources=reranked_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")