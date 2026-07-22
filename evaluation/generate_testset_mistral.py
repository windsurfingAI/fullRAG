import os
import json
import random
import requests
import pandas as pd

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset.json")

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

SAMPLE_SIZE = 100  # Nombre de requêtes à générer
sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

generated_dataset = []
QUERY_TYPES = ["question_imperfaite", "titre_seul", "realisateur_seul"]

print(f"🤖 Connexion à Ollama local sur {OLLAMA_URL}...")
print(f"🎭 Pool de modèles utilisé : {', '.join(MODELS_POOL)}")
print(f"🚀 Génération de {len(sampled_df)} paires de test réalistes...\n")

for idx, row in sampled_df.iterrows():
    film_info = "\n".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
    query_type = random.choice(QUERY_TYPES)
    
    # 🎲 Sélection aléatoire du modèle pour ce film
    selected_model = random.choice(MODELS_POOL)
    
    prompt = f"""[INST] Tu es un générateur de données de test pour un système RAG de cinéma.
À partir de la fiche du film ci-dessous, génère UNE requête d'utilisateur et la RÉPONSE exacte (ground_truth) basée uniquement sur la fiche.

Type de requête à générer pour cette fiche : **{query_type}**

RÈGLES DE GÉNÉRATION SELON LE TYPE :
- Si query_type == "question_imperfaite" :
  Génère une question naturelle mais tapée avec des coquilles/fautes réalistes sur téléphone (ex: "tu conai le film a bout de nerf ?").
- Si query_type == "titre_seul" :
  Génère uniquement le titre du film, potentiellement sans majuscule, sans accent ou avec une petite faute de frappe (ex: "a bout de nerf").
- Si query_type == "realisateur_seul" :
  Génère uniquement le nom du réalisateur, potentiellement avec une faute d'orthographe (ex: "jean luc godar").

Fiche du film :
{film_info}

Réponds STRICTEMENT sous forme d'un objet JSON valide au format suivant, sans aucun autre texte autour :
{{
    "type": "{query_type}",
    "question": "La requête utilisateur ici",
    "ground_truth": "La réponse exacte basée sur la fiche"
}} [/INST]"""

    try:
        # Appel API à Ollama local avec le modèle tiré au sort
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": selected_model,
                "prompt": prompt,
                "format": "json",  # Force Ollama à répondre en JSON valide
                "stream": False
            },
            timeout=90  # Timeout légèrement allongé pour les modèles plus volumineux
        )
        
        if response.status_code == 200:
            raw_text = response.json().get("response", "").strip()
            
            # Nettoyage si DeepSeek inclut ses balises de réflexion interne <think>...</think>
            if "<think>" in raw_text:
                raw_text = raw_text.split("</think>")[-1].strip()
                
            qa_pair = json.loads(raw_text)
            
            generated_dataset.append({
                "generator_model": selected_model,  # Métadonnée pour savoir quel modèle a généré la question
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