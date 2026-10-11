# -*- coding: utf-8 -*-
import sys
import types
import time
import os
import json
import requests
import pandas as pd
from datasets import Dataset

# Chargement automatique des variables d'environnement depuis le .env
from dotenv import load_dotenv
load_dotenv()

# --- CONTOURNEMENT DU BUG D'IMPORT RAGAS 0.1 ---
if "langchain_community.chat_models.vertexai" not in sys.modules:
    fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai
# ------------------------------------------------

from ragas import evaluate, RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. Configuration des modes d'exécution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH_3MODELS = os.path.join(BASE_DIR, "synthetic_testset_3models.json")
JSON_PATH_DEFAULT = os.path.join(BASE_DIR, "synthetic_testset_gemini.json")
JSON_PATH = JSON_PATH_3MODELS if os.path.exists(JSON_PATH_3MODELS) else JSON_PATH_DEFAULT

RESULTS_CSV_PATH = os.path.join(BASE_DIR, "rag_evaluation_results.csv")
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000/chat")

# Détection de la clé API
google_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
force_local = os.environ.get("USE_LOCAL_LLM", "false").lower() in ("true", "1", "yes")

# Mode local si forcé ou si aucune clé n'est fournie
USE_OLLAMA = force_local or (not google_api_key)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

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

provider = "ollama" if USE_OLLAMA else "gemini"
print(f"🔄 Interrogation de l'API RAG ({RAG_API_URL}) avec le provider '{provider}' sur {len(testset)} requêtes...")

for item in testset:
    query = item["question"]

    # Pause uniquement nécessaire si on est en cloud gratuit (Gemini)
    if not USE_OLLAMA:
        time.sleep(4)

    try:
        res = requests.post(
            RAG_API_URL, 
            json={"query": query, "provider": provider},
            timeout=60
        )
        
        if res.status_code == 200:
            data = res.json()
            raw_sources = data.get("sources", [])
            
            cleaned_sources = []
            for src in raw_sources:
                if isinstance(src, dict):
                    cleaned_sources.append(src.get("content") or src.get("text") or str(src))
                else:
                    cleaned_sources.append(str(src))

            questions.append(query)
            answers.append(data.get("answer", ""))
            contexts.append(cleaned_sources)
            ground_truths.append(str(item.get("ground_truth", "")))
            
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

print("\n📊 Préparation de l'évaluation RAGAS...")

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

# Initialisation du LLM et des Embeddings selon le mode
if USE_OLLAMA:
    print(f"🤖 Utilisation d'Ollama ({OLLAMA_MODEL}) comme juge RAGAS local.")
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

    llm_judge = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.0,
        request_timeout=180.0
    )
    
    # Embeddings locaux via Ollama (assure-toi d'avoir fait: ollama pull nomic-embed-text)
    # Alternative : langchain_huggingface / sentence_transformers
    emb_judge = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBED_MODEL
    )
else:
    print("🤖 Utilisation de Gemini comme juge RAGAS cloud.")
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm_judge = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key
    )
    emb_judge = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

evaluator_llm = LangchainLLMWrapper(llm_judge)
evaluator_embeddings = LangchainEmbeddingsWrapper(emb_judge)

# Configuration dynamique du débit et des timeouts
if USE_OLLAMA:
    print(f"⚙️ Configuration Ragas : Mode Local (timeout=180s, max_workers=1)")
    run_config = RunConfig(
        timeout=180,       # 3 min pour laisser le GPU local calculer sans couper
        max_workers=1,     # 1 seule requête à la fois pour éviter d'engorger la file Ollama
        max_retries=3
    )
else:
    print("⚡ Configuration Ragas : Mode Cloud Pay-as-you-go (timeout=60s, max_workers=8)")
    run_config = RunConfig(
        timeout=60,
        max_workers=8,     # 8 vérifications en parallèle pour finir en 1 à 2 min
        max_retries=5
    )

eval_kwargs = {
    "dataset": dataset,
    "metrics": metrics,
    "llm": evaluator_llm,
    "embeddings": evaluator_embeddings,
    "run_config": run_config
}

results = evaluate(**eval_kwargs)

# 4. Affichage et sauvegarde des résultats
print("\n=== SCORES GLOBAUX RAG ===")
print(results)

df_results = results.to_pandas()
df_results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
print(f"\n💾 Détail des résultats sauvegardé dans '{RESULTS_CSV_PATH}'")

if "query_type" in df_results.columns:
    print("\n=== SCORES MOYENS PAR TYPE DE REQUÊTE ===")
    metric_cols = [m.name for m in metrics if m.name in df_results.columns]
    if metric_cols:
        print(df_results.groupby("query_type")[metric_cols].mean())