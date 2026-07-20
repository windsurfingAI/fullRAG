import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import requests
from google import genai  # Nouveau SDK Google GenAI (recommandé en 2026)
from sentence_transformers import CrossEncoder

app = FastAPI(title="Local & Free RAG Orchestrator")

# 1. Initialisation de Gemini (Cloud)
# Le SDK récupère automatiquement la variable d'environnement GEMINI_API_KEY
try:
    gemini_client = genai.Client()
except Exception:
    gemini_client = None

# 2. Initialisation de ChromaDB (Local)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

# 3. Initialisation du Reranker (Local & Léger)
# Télécharge un modèle de reranking au premier lancement (~200 Mo) et s'exécute en local
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Schémas de données
class ChatRequest(BaseModel):
    query: str
    provider: str = "gemini"  # Choix de l'utilisateur : 'gemini' ou 'mistral'
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    try:
        # Étape 1 : Récupération des documents dans la base locale
        collection = chroma_client.get_or_create_collection(name="documents")
        search_results = collection.query(
            query_texts=[request.query],
            n_results=10  # On récupère une base large de 10 candidats
        )
        
        documents = search_results.get("documents", [[]])[0]
        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document trouvé dans la base.")

        # Étape 2 : Reranking Local
        # On passe les paires (requête, document) au modèle de reranking
        pairs = [[request.query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        
        # On trie les documents selon le score obtenu et on garde les 3 meilleurs
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs[:3]]

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
            
            # Appel à Gemini (2.5 Flash est l'option gratuite recommandée)
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            answer = response.text

        elif request.provider == "mistral":
            # Appel à l'instance locale d'Ollama (Mistral 7B)
            try:
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
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
                    detail="Impossible de contacter Ollama. Est-il lancé avec 'ollama serve' ?"
                )
        else:
            raise HTTPException(status_code=400, detail="Fournisseur d'IA non supporté.")

        return ChatResponse(
            answer=answer,
            sources=reranked_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")