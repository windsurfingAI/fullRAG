import os
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

app = FastAPI(title="Local & Free RAG Orchestrator (Hybrid Search)")

# 1. Initialisation de Gemini (Cloud)
try:
    gemini_client = genai.Client()
except Exception:
    gemini_client = None

# 2. Configuration des chemins et modèles
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "documents"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialisation du modèle d'embedding pour LangChain Chroma
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'}
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
        weights=[0.3, 0.7]  # 50% BM25, 50% Vecteur
    )
else:
    hybrid_retriever = chroma_retriever

# 4. Initialisation du Reranker (Local)
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Schémas de données
class ChatRequest(BaseModel):
    query: str
    provider: str = "gemini"  # 'gemini' ou 'mistral'
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    try:
        # Étape 1 : Recherche Hybride (Recoupe les résultats BM25 exacts + la proximité vectorielle)
        retrieved_docs = hybrid_retriever.invoke(request.query)
        
        # Extraction du texte des documents récupérés
        documents = [doc.page_content for doc in retrieved_docs]
        
        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document trouvé dans la base.")

        # Étape 2 : Reranking Local
        # Le Reranker affine le classement du Retriever Hybride
        pairs = [[request.query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        
        # Tri selon le score et sélection des meilleurs résultats (par défaut top 6)
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs[:6]]

        # Étape 3 : Construction du prompt RAG
        context = "\n\n".join(reranked_docs)
        prompt = f"""Réponds à la question en te basant uniquement sur le contexte fourni ci-dessous.
        
Contexte:
{context}

Question: {request.query}
Réponse:"""

        # Étape 4 : Génération de la réponse selon le fournisseur choisi
        if request.provider == "gemini":
            if not gemini_client:
                raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
            
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            answer = response.text

        elif request.provider == "mistral":
            try:
                # Utiliser le nom du service Docker 'ollama' au lieu de 'localhost' si vous êtes dans un conteneur Docker
                ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                ollama_response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False
                    }
                )
                if ollama_response.status_code == 200:
                    answer = ollama_response.json().get("response", "")
                else:
                    raise HTTPException(status_code=500, detail="Erreur lors de l'appel à Ollama.")
            except requests.exceptions.ConnectionError:
                raise HTTPException(
                    status_code=500, 
                    detail="Impossible de contacter Ollama. Vérifiez que le service tourne correctement."
                )
        else:
            raise HTTPException(status_code=400, detail="Fournisseur d'IA non supporté.")

        return ChatResponse(
            answer=answer,
            sources=reranked_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")