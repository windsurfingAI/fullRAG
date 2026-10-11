# -*- coding: utf-8 -*-
import sys
import types
import time

# --- CONTOURNEMENT DU BUG D'IMPORT RAGAS 0.1 ---
if "langchain_community.chat_models.vertexai" not in sys.modules:
    fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai
# ------------------------------------------------
import os
import json
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# 1. Gestion des chemins et URL API
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Priorité au fichier 3 modèles s'il existe, sinon fallback sur synthetic_testset.json
JSON_PATH_3MODELS = os.path.join(BASE_DIR, "synthetic_testset_3models.json")
JSON_PATH_DEFAULT = os.path.join(BASE_DIR, "synthetic_testset_gemini.json")
JSON_PATH = JSON_PATH_3MODELS if os.path.exists(JSON_PATH_3MODELS) else JSON_PATH_DEFAULT

RESULTS_CSV_PATH = os.path.join(BASE_DIR, "rag_evaluation_results.csv")

# URL de l'API RAG FastAPI
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000/chat")

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ Le jeu de test est introuvable : {JSON_PATH}. Lancez d'abord la génération.")

# 2. Chargement du jeu de test
print(f"📂 Chargement du jeu de test depuis : {os.path.basename(JSON_PATH)}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    testset = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []
query_types = []
film_ids = []

print(f"🔄 Interrogation de l'API RAG ({RAG_API_URL}) sur {len(testset)} requêtes...")

for item in testset:
    query = item["question"]


    # Pause pour respecter le quota API
    time.sleep(10)
    
    try:
        res = requests.post(
            RAG_API_URL, 
            json={"query": query, "provider": "gemini"},
            timeout=30
        )
        
        if res.status_code == 200:
            data = res.json()
            raw_sources = data.get("sources", [])
            
            # Normalisation : Ragas exige strictement une liste de chaînes (list[str])
            cleaned_sources = []
            for src in raw_sources:
                if isinstance(src, dict):
                    # Adapte les clés selon le format de ton API ('content', 'text', 'chunk', etc.)
                    cleaned_sources.append(src.get("content") or src.get("text") or str(src))
                else:
                    cleaned_sources.append(str(src))

            questions.append(query)
            answers.append(data.get("answer", ""))
            contexts.append(cleaned_sources)
            # Ragas attend une simple chaîne de caractères (str) et non une liste
            ground_truths.append(str(item.get("ground_truth", "")))
            
            # Conservation des métadonnées pour analyse ultérieure
            query_types.append(item.get("type", "inconnu"))
            film_ids.append(item.get("film_id", -1))
            
            print(f"  ✓ [{len(questions)}/{len(testset)}] Traité : {query[:45]}...")
        else:
            print(f"⚠️ Erreur API ({res.status_code}) pour : '{query}'")

    except Exception as e:
        print(f"⚠️ Impossible de contacter l'API pour '{query}' : {e}")

# 3. Calcul des métriques RAGAS
if not questions:
    raise RuntimeError("❌ Aucune donnée collectée. Vérifiez que votre serveur FastAPI tourne bien sur le port 8000.")

print("\n📊 Évaluation en cours avec RAGAS...")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
    "query_type": query_types,
    "film_id": film_ids
})

metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
]

# Détection et configuration du LLM juge (Gemini si configuré, sinon défaut OpenAI)
eval_kwargs = {
    "dataset": dataset,
    "metrics": metrics
}

google_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("❌ Aucune clé API trouvée dans GEMINI_API_KEY ou GOOGLE_API_KEY.")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. Instanciation des composants LangChain avec Gemini
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key
)
emb_gemini = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key
)

# 2. Encapsulation dans les wrappers Ragas officiels
evaluator_llm = LangchainLLMWrapper(llm_gemini)
evaluator_embeddings = LangchainEmbeddingsWrapper(emb_gemini)

# 3. Injection explicite dans le dictionnaire passé à evaluate()
eval_kwargs = {
    "dataset": dataset,
    "metrics": metrics,
    "llm": evaluator_llm,
    "embeddings": evaluator_embeddings
}

print("🤖 Utilisation de Gemini comme juge RAGAS.")
results = evaluate(**eval_kwargs)

# 4. Affichage et sauvegarde des résultats
print("\n=== SCORES GLOBAUX RAG ===")
print(results)

df_results = results.to_pandas()
df_results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
print(f"\n💾 Détail des résultats sauvegardé dans '{RESULTS_CSV_PATH}'")

# Affichage d'un récapitulatif par type de requête si la colonne existe
if "query_type" in df_results.columns:
    print("\n=== SCORES MOYENS PAR TYPE DE REQUÊTE ===")
    metric_cols = [m.name for m in metrics if m.name in df_results.columns]
    if metric_cols:
        print(df_results.groupby("query_type")[metric_cols].mean())