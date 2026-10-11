import os
import json
import random
import re
import requests
import pandas as pd

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset_3models_blabla.json")

# URL de l'instance Ollama locale
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"

# 🎲 Pool des 3 modèles à faire tourner en alternance
MODELS_POOL = [
    "qwen2.5:7b",
    "mistral-nemo",
    "deepseek-r1:7b"
]

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"❌ Fichier introuvable : {CSV_FILE_PATH}")

# 2. Chargement du CSV
df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8").fillna("")

SAMPLE_SIZE = 10  # Nombre de requêtes à générer
sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

generated_dataset = []

# Répartition pondérée des 4 catégories (60% canonique, 20% bruitee, 10% mots_cles, 10% conversationnelle)
QUERY_TYPES_WEIGHTED = (
    ["canonique"] * 6 + 
    ["bruitee"] * 2 + 
    ["mots_cles"] * 1 + 
    ["conversationnelle"] * 1
)

print(f"🤖 Connexion à Ollama local sur {OLLAMA_URL}...")
print(f"🎭 Pool de modèles utilisé : {', '.join(MODELS_POOL)}")
print(f"🚀 Génération de {len(sampled_df)} paires de test diversifiées (4 catégories)...\n")

for idx, row in sampled_df.iterrows():
    film_info = "\n".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
    query_type = random.choice(QUERY_TYPES_WEIGHTED)
    
    # 🎲 Sélection aléatoire du modèle pour ce film
    selected_model = random.choice(MODELS_POOL)
    
    prompt = f"""[INST] Tu es un générateur de données de test pour un système RAG de cinéma.
À partir de la fiche du film ci-dessous, génère UNE requête d'utilisateur et la RÉPONSE exacte (ground_truth) basée uniquement sur la fiche.

Type de requête à générer pour cette fiche : **{query_type}**

RÈGLES DE GÉNÉRATION SELON LE TYPE :
- Si query_type == "canonique" :
  Question complète, polie, claire et avec une syntaxe irréprochable (ex: "En quelle année est sorti le film ... et qui en est le réalisateur ?").
- Si query_type == "bruitee" :
  Question tapée rapidement sur smartphone avec des coquilles, fautes ou syntaxe orale (ex: "c ki qui a fai ce film et c sorti kan").
- Si query_type == "mots_cles" :
  Uniquement 2 à 4 mots-clés bruts sans ponctuation ni phrase (ex: "titre du film annee realisateur").
- Si query_type == "conversationnelle" :
  Question utilisant un pronom ou une référence implicite faisant suite à un échange (ex: "Qui a réalisé ce film ?", "En quelle année est-il sorti ?").

Fiche du film :
{film_info}

Réponds STRICTEMENT sous forme d'un objet JSON valide au format suivant, sans aucun autre texte autour :
{{
    "type": "{query_type}",
    "question": "La requête utilisateur ou les mots-clés ici",
    "ground_truth": "La réponse exacte basée sur la fiche"
}} [/INST]"""

    try:
        # Configuration de la requête vers Ollama
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False
        }
        
        # On n'impose pas format: json à DeepSeek pour ne pas bloquer sa balise <think>
        if "deepseek" not in selected_model:
            payload["format"] = "json"

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=90
        )
        
        if response.status_code == 200:
            raw_text = response.json().get("response", "").strip()
            
            # Nettoyage si DeepSeek inclut ses balises de réflexion interne <think>...</think>
            if "</think>" in raw_text:
                raw_text = raw_text.split("</think>")[-1].strip()
            
            # Extraction du premier bloc JSON complet trouvé dans le texte
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"Aucun JSON exploitable détecté dans la réponse brute : {raw_text[:120]}...")
            
            qa_pair = json.loads(json_match.group(0))
            
            generated_dataset.append({
                "generator_model": selected_model,
                "type": qa_pair.get("type", query_type),
                "question": qa_pair["question"],
                "ground_truth": qa_pair["ground_truth"],
                "film_id": int(idx)
            })
            
            print(f"  └─ [{len(generated_dataset)}/{len(sampled_df)}] [{selected_model}] [{query_type}] -> \"{qa_pair['question']}\"")
        
        elif response.status_code == 404:
            print(f"⚠️ Modèle non trouvé ({selected_model}). Lancez 'ollama pull {selected_model}' dans votre terminal.")
        else:
            print(f"⚠️ Erreur Ollama ({response.status_code}) avec {selected_model} pour le film {idx}")

    except requests.exceptions.ConnectionError:
        raise RuntimeError("❌ Impossible de contacter Ollama. Assurez-vous qu'Ollama est démarré avec 'ollama serve'.")
    except Exception as e:
        print(f"⚠️ Erreur de traitement avec {selected_model} sur le film {idx} : {e}")

# 3. Sauvegarde
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(generated_dataset)} paires de test sauvegardées dans '{OUTPUT_JSON_PATH}' !")