import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# 1. Chargement de l'environnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../.env")
load_dotenv(ENV_PATH) if os.path.exists(ENV_PATH) else load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Clé GEMINI_API_KEY introuvable.")

# 2. Chemins
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset.json")

# 3. Chargement et échantillonnage
df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8").fillna("")
SAMPLE_SIZE = 50 
BATCH_SIZE = 4   # Taille de lot sécurisée pour éviter d'exploser le token/min

sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
client = genai.Client(api_key=GEMINI_API_KEY)
generated_dataset = []

print(f"🚀 Génération de {len(sampled_df)} questions diversifiées (Lots de {BATCH_SIZE})...")

records = list(sampled_df.iterrows())

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    batch_text = ""
    
    for idx, row in batch:
        film_info_parts = []
        for col, val in row.items():
            val_str = str(val).strip()
            if val_str and val_str.lower() not in ["nan", "none", "0"]:
                film_info_parts.append(f"{col}: {val_str}")
        
        film_info = "\n".join(film_info_parts)[:800] 
        batch_text += f"--- FILM ID : {idx} ---\n{film_info}...\n\n"
    
    # Prompt enrichi avec les contraintes strictes de catégories
    prompt = f"""
Tu es un générateur de données de test synthétiques pour évaluer un système RAG sur le cinéma.

À partir des fiches de films fournies ci-dessous, génère pour CHAQUE film UNE question associée à sa réponse exacte (ground_truth), basée strictement sur les informations fournies.

Répartis les types de questions de manière équilibrée sur le lot parmi ces 4 catégories :
- "canonique" : Question complète, polie, claire et avec une syntaxe irréprochable.
- "bruitee" : Question tapée rapidement sur smartphone avec des coquilles, fautes ou syntaxe orale (ex: "c ki qui a fai...").
- "mots_cles" : Uniquement 2 à 4 mots-clés bruts sans ponctuation ni phrase.
- "conversationnelle" : Question utilisant un pronom ou une référence implicite (ex: "Qui a réalisé ce film ?", "En quelle année est-il sorti ?").

Fiches :
{batch_text}

Renvoie STRICTEMENT un tableau JSON valide de cette forme (aucune autre phrase, pas de markdown autour si possible, ou juste le bloc json) :
[
    {{
        "film_id": ID_du_film,
        "type": "canonique ou bruitee ou mots_cles ou conversationnelle",
        "question": "La question ou les mots-clés ici",
        "ground_truth": "La réponse exacte et concise basée sur la fiche"
    }}
]
"""
    
    # Gestion des essais (Exponential Backoff)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            # Nettoyage au cas où le modèle encapsule dans du markdown ```json ... ```
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            results = json.loads(raw_text)
            if isinstance(results, list):
                generated_dataset.extend(results)
                print(f"  └─ Lot {i//BATCH_SIZE + 1} OK : {len(results)} questions ajoutées.")
            else:
                print(f"  ⚠️ Le JSON retourné n'est pas une liste au lot {i//BATCH_SIZE + 1}. Nouvel essai...")
                raise ValueError("Format JSON invalide (pas une liste)")
                
            time.sleep(8) # Pause pour respecter les limites du Free Tier
            break 

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                wait_time = (attempt + 1) * 15
                print(f"  ⏳ Limite de quota atteinte. Pause de {wait_time}s...")
                time.sleep(wait_time)
            elif "json" in error_msg or "valueerror" in error_msg:
                print(f"  ⚠️ Erreur de parsing JSON au lot {i//BATCH_SIZE + 1}. Nouvel essai...")
                time.sleep(5)
            else:
                print(f"  ❌ Erreur sur le lot {i//BATCH_SIZE + 1} : {e}")
                break

# 4. Sauvegarde finale
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(generated_dataset)} questions générées et sauvegardées dans '{OUTPUT_JSON_PATH}' !")