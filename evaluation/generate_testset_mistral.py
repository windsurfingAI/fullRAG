import os
import json
import random
import pandas as pd
from pydantic import BaseModel
import ollama

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "synthetic_testset_varied.json")

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"❌ Fichier introuvable : {CSV_FILE_PATH}")

# 2. Schéma Pydantic
class QAPair(BaseModel):
    question: str
    ground_truth: str

# 3. Chargement du CSV
df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8").fillna("")

SAMPLE_SIZE = 20
sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

# 🎭 Dictionnaires des STYLES DE QUESTIONS pour forcer la variété
QUESTION_STYLES = [
    {
        "style": "Chercher le Réalisateur",
        "instruction": "Pose une question sur QUI a réalisé ce film (ex: 'Qui a réalisé le film X ?' ou 'Quel réalisateur est derrière X ?')."
    },
    {
        "style": "Chercher l'Année ou le Format",
        "instruction": "Pose une question axée sur la date de sortie, la nationalité ou le format du film (ex: 'En quelle année est sorti X ?' ou 'Quel est le format du film Y ?')."
    },
    {
        "style": "Question de type Oui/Non ou Vérification",
        "instruction": "Pose une question de vérification VRAI/FAUX ou OUI/NON (ex: 'Est-ce que le film X est un court-métrage ?' ou 'Est-ce que Y a été produit par Z ?'). La réponse doit valider ou corriger."
    },
    {
        "style": "Recherche par critères (Sans donner le titre directement)",
        "instruction": "Pose une question où l'utilisateur cherche un film en donnant son réalisateur et son année, sans nommer le film (ex: 'Quel film a été réalisé par X en Y ?')."
    },
    {
        "style": "Focus Production/Visa",
        "instruction": "Pose une question sur la maison de production ou les restrictions/visas du film (ex: 'Quelle société a produit X ?' ou 'Quelle est la décision de classification pour X ?')."
    }
]

generated_dataset = []
MODEL_NAME = "mistral"

print(f"🏠 Utilisation de {MODEL_NAME} avec variabilité des prompts.")
print(f"🚀 Génération de {len(sampled_df)} questions variées...")

for idx, row in sampled_df.iterrows():
    film_info = "\n".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
    
    # 🎲 Sélection aléatoire d'un style de question pour ce film
    selected_style = random.choice(QUESTION_STYLES)
    
    prompt = f"""
À partir de la fiche du film suivante, génère UNE question naturelle d'un utilisateur et sa RÉPONSE exacte (ground_truth).

Fiche du film :
{film_info}

CONSIGNE STRICTE DE STYLE POUR LA QUESTION :
{selected_style['instruction']}

Évite absolument de toujours commencer par 'Quel est le titre de...'. Varie la formulation !
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "Tu es un expert qui génère des jeux de données de test variés pour évaluer un système RAG. Tu réponds au format JSON."
                },
                {"role": "user", "content": prompt}
            ],
            format=QAPair.model_json_schema()
        )
        
        qa_pair = json.loads(response.message.content)
        
        generated_dataset.append({
            "question": qa_pair["question"],
            "ground_truth": qa_pair["ground_truth"],
            "style_used": selected_style["style"], # Permet de vérifier la diversité
            "film_id": int(idx)
        })
        print(f"  └─ [{len(generated_dataset)}/{len(sampled_df)}] Style: [{selected_style['style']}] -> Film: {row.get('Titre', 'Film')}")

    except Exception as e:
        print(f"⚠️ Erreur sur le film {idx} : {e}")

# Save
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ Dataset varié sauvegardé dans '{OUTPUT_JSON_PATH}' !")