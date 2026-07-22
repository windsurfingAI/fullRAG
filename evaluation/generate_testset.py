import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# 1. Chargement du fichier .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../.env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ La variable GEMINI_API_KEY est introuvable dans le fichier .env.")

# 2. Gestion des chemins
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset.json")

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"❌ Fichier introuvable : {CSV_FILE_PATH}")

# 3. Chargement du CSV
df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8").fillna("")

SAMPLE_SIZE = 20
sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

# 4. Client Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

generated_dataset = []

print(f"🔑 Clé API Gemini détectée.")
print(f"🚀 Génération de {len(sampled_df)} questions (avec gestion des limites de débit)...")

for idx, row in sampled_df.iterrows():
    film_info = "\n".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
    
    prompt = f"""
À partir de la fiche du film suivante, génère UNE question naturelle qu'un utilisateur pourrait poser, et la RÉPONSE exacte basée uniquement sur la fiche.

Fiche du film :
{film_info}

Réponds STRICTEMENT sous forme de JSON valide avec ce format exact :
{{
    "question": "La question ici",
    "ground_truth": "La réponse exacte ici"
}}
"""
    # Système de réessai automatique en cas de limite de quota (Rate Limit)
    max_retries = 3
    success = False

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            qa_pair = json.loads(response.text)
            generated_dataset.append({
                "question": qa_pair["question"],
                "ground_truth": qa_pair["ground_truth"],
                "film_id": int(idx)
            })
            print(f"  └─ [{len(generated_dataset)}/{len(sampled_df)}] QA générée pour : {row.get('Titre', 'Film')}")
            success = True
            break  # Succès, on sort de la boucle de réessai

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (attempt + 1) * 5
                print(f"  ⏳ Limite de débit atteinte. Pause de {wait_time}s avant réessai...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ Erreur non liée au quota sur le film {idx} : {e}")
                break

    # Pause systématique de 4 secondes pour respecter le quota gratuit (15 req/min)
    time.sleep(4)

# 5. Sauvegarde
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(generated_dataset)}/{len(sampled_df)} questions sauvegardées dans '{OUTPUT_JSON_PATH}' !")