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
    answer_relevance,
)

# 1. Gestion des chemins et URL API
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset.json")
RESULTS_CSV_PATH = os.path.join(BASE_DIR, "rag_evaluation_results.csv")

# URL de l'API (ajustez le port ou l'hôte si besoin)
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000/chat")

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ Le jeu de test est introuvable : {JSON_PATH}. Lancez 'python generate_testset.py' d'abord.")

# 2. Chargement du jeu de test
with open(JSON_PATH, "r", encoding="utf-8") as f:
    testset = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

print(f"🔄 Interrogation de l'API RAG ({RAG_API_URL}) sur {len(testset)} questions...")

for item in testset:
    query = item["question"]
    
    try:
        res = requests.post(
            RAG_API_URL, 
            json={"query": query, "provider": "gemini"},
            timeout=30
        )
        
        if res.status_code == 200:
            data = res.json()
            questions.append(query)
            answers.append(data.get("answer", ""))
            contexts.append(data.get("sources", []))
            ground_truths.append([item["ground_truth"]])
        else:
            print(f"⚠️ Erreur API ({res.status_code}) pour : '{query}'")

    except Exception as e:
        print(f"⚠️ Impossible de contacter l'API pour '{query}' : {e}")

# 3. Calcul des métriques RAGAS
if not questions:
    raise RuntimeError("❌ Aucune donnée collectée. Vérifiez que votre API RAG est bien démarrée.")

print("\n📊 Évaluation en cours avec RAGAS...")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

results = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevance
    ]
)

# 4. Affichage et sauvegarde des résultats
print("\n=== SCORES GLOBAUX RAG ===")
print(results)

df_results = results.to_pandas()
df_results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
print(f"\n💾 Détail des résultats sauvegardé dans '{RESULTS_CSV_PATH}'")