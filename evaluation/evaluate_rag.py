# -*- coding: utf-8 -*-
import sys
import types
import time
import os
import json
import requests
import uuid
import pandas as pd
from datasets import Dataset

# Permet d'importer la config du backend depuis le dossier evaluation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from backend.src.config import RAGConfig
except ImportError:
    print("⚠️ Impossible d'importer backend.src.config. Assurez-vous que config.py existe.")
    # Fallback au cas où le script est lancé depuis un dossier inattendu
    class RAGConfig:
        pass

import mlflow
from langfuse import Langfuse

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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- INITIALISATION MLFLOW & LANGFUSE ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("RAG_Cinema_Experiment")

langfuse = Langfuse()

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ Le jeu de test est introuvable : {JSON_PATH}. Lancez d'abord la génération.")

# 2. Chargement du jeu de test
print(f"📂 Chargement du jeu de test depuis : {os.path.basename(JSON_PATH)}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    testset = json.load(f)

# ---> AJOUT POUR TEST RAPIDE <---
testset = testset[:3] 
print(f"⚠️ MODE TEST SANS ATTENTE : On ne garde que {len(testset)} questions.")
# --------------------------------

questions = []
answers = []
contexts = []
ground_truths = []
query_types = []
film_ids = []
trace_ids = [] # Pour lier Langfuse à Ragas

provider = "mistral" if USE_OLLAMA else "gemini"
print(f"🔄 Interrogation de l'API RAG ({RAG_API_URL}) avec le provider '{provider}' sur {len(testset)} requêtes...")

# DÉMARRAGE DE LA RUN MLFLOW
with mlflow.start_run(run_name=f"Eval_Auto_{provider}"):
    
    # --- ENREGISTREMENT AUTOMATIQUE DES PARAMÈTRES (CONFIG.PY) ---
    active_config = {k: v for k, v in RAGConfig.__dict__.items() if not k.startswith("__")}
    mlflow.log_params(active_config)
    mlflow.log_param("testset", os.path.basename(JSON_PATH))

    for item in testset:
        query = item["question"]
        unique_session_id = str(uuid.uuid4())

        if not USE_OLLAMA:
            time.sleep(4)

        try:
            payload = {
                "session_id": unique_session_id,
                "query": query,
                "provider": provider,
                "user_profile": {
                    "name": "Bot Evaluateur",
                    "role": "Testeur QA",
                    "department": "Évaluation",
                    "clearance": "Standard"
                }
            }

            res = requests.post(RAG_API_URL, json=payload, timeout=60)
            
            if res.status_code == 200:
                data = res.json()
                raw_sources = data.get("sources", [])
                
                # Récupération du trace_id renvoyé par app.py
                current_trace_id = data.get("trace_id") 
                
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
                trace_ids.append(current_trace_id)
                
                print(f"  ✓ [{len(questions)}/{len(testset)}] Traité : {query[:45]}...")
            else:
                print(f"⚠️ Erreur API ({res.status_code}) pour : '{query}' -> Détail : {res.text}")

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
        emb_judge = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
        run_config = RunConfig(timeout=180, max_workers=1, max_retries=3)
    else:
        print("🤖 Utilisation de Gemini comme juge RAGAS cloud.")
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm_judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        emb_judge = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
        run_config = RunConfig(timeout=60, max_workers=8, max_retries=5)

    evaluator_llm = LangchainLLMWrapper(llm_judge)
    evaluator_embeddings = LangchainEmbeddingsWrapper(emb_judge)

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
    df_results["trace_id"] = trace_ids  # On ajoute l'ID de la trace dans le CSV

    # --- ENREGISTREMENT SÉCURISÉ DES MÉTRIQUES DANS MLFLOW ---
    metric_names = [m.name for m in metrics]
    mean_scores = df_results[metric_names].mean(numeric_only=True).dropna().to_dict()
    if mean_scores:
        mlflow.log_metrics(mean_scores)
        print(f"📈 Métriques loggées dans MLflow : {mean_scores}")
    else:
        print("⚠️ Aucune métrique numérique valide à envoyer à MLflow.")
    
    df_results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
    print(f"\n💾 Détail des résultats sauvegardé dans '{RESULTS_CSV_PATH}'")
    
    # Envoi du CSV comme artefact MLflow
    # mlflow.log_artifact(RESULTS_CSV_PATH)

    # --- 5. ENVOI DES SCORES DÉTAILLÉS VERS LANGFUSE ---
    print("\n🚀 Injection des scores Ragas dans Langfuse...")
    for idx, row in df_results.iterrows():
        t_id = row.get("trace_id")
        if pd.notna(t_id) and t_id:
            for metric in metrics:
                metric_name = metric.name
                if metric_name in row and pd.notna(row[metric_name]):
                    langfuse.create_score(
                        trace_id=str(t_id),
                        name=f"ragas_{metric_name}",
                        value=float(row[metric_name]),
                        comment="Score généré automatiquement par Ragas"
                    )
    langfuse.flush()
    print("✅ Scores Langfuse mis à jour !")

    if "query_type" in df_results.columns:
        print("\n=== SCORES MOYENS PAR TYPE DE REQUÊTE ===")
        metric_cols = [m.name for m in metrics if m.name in df_results.columns]
        if metric_cols:
            print(df_results.groupby("query_type")[metric_cols].mean())