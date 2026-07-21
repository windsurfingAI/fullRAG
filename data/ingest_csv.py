import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import torch
from sentence_transformers import SentenceTransformer

# 1. Configuration et Chemins
CSV_FILE_PATH = "data/raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"
MODEL_NAME = "all-MiniLM-L6-v2"  # Vous pouvez changer pour un modèle multilingue comme 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 256  # Ajustez selon la VRAM de votre GPU

# 2. Vérification et initialisation du GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Appareil de calcul utilisé : {device.upper()}")
if device == "cpu":
    print("⚠️  Aucun GPU CUDA détecté, l'exécution se fera sur le CPU.")

print(f"📦 Chargement du modèle d'embedding '{MODEL_NAME}' sur {device.upper()}...")
embedding_model = SentenceTransformer(MODEL_NAME, device=device)

# 3. Chargement et nettoyage du CSV
print(f"📖 Lecture du fichier : {CSV_FILE_PATH}")
df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8")
df = df.fillna("")

# 4. Connexion à ChromaDB et réinitialisation de la collection
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
    print(f"🧹 Ancienne collection '{COLLECTION_NAME}' supprimée.")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

documents = []
metadatas = []
ids = []

# 5. Préparation et enrichissement des données
print("🛠️ Préparation et enrichissement sémantique des documents...")
for idx, row in df.iterrows():
    lines = []
    
    # Extraction des données
    titre = str(row.get("Titre", "")).strip()
    realisation = str(row.get("Réalisation", "")).strip()

    # Champs bruts (Clé: Valeur)
    for col, val in row.items():
        val_str = str(val).strip()
        if val_str:
            lines.append(f"{col}: {val_str}")

    # Enrichissement sémantique
    if titre and realisation:
        lines.append(f"Le film '{titre}' a été réalisé par {realisation}.")
        lines.append(f"Réalisateur : {realisation}")
        lines.append(f"Films de {realisation} : {titre}")
        
        words = realisation.split()
        if len(words) == 2:
            inverted = f"{words[1]} {words[0]}"
            lines.append(f"Réalisé par {inverted}")

    text_content = "\n".join(lines)
    documents.append(text_content)
    ids.append(f"csv_row_{idx}")
    
    # Métadonnées
    row_meta = {str(k): str(v) for k, v in row.items() if str(v).strip() != ""}
    row_meta["row_index"] = idx
    metadatas.append(row_meta)

# 6. Génération des embeddings sur GPU et insertion par lots dans ChromaDB
total_docs = len(documents)
print(f"🚀 Début de l'indexation sur GPU de {total_docs} documents...")

for i in range(0, total_docs, BATCH_SIZE):
    batch_docs = documents[i : i + BATCH_SIZE]
    batch_ids = ids[i : i + BATCH_SIZE]
    batch_meta = metadatas[i : i + BATCH_SIZE]
    
    # --- CALCUL SUR GPU ---
    # Convert_to_numpy=True transforme le tenseur CUDA en liste compatible avec ChromaDB
    batch_embeddings = embedding_model.encode(
        batch_docs, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=False, 
        convert_to_numpy=True,
        device=device
    )

    # Insertion explicite des embeddings calculés par le GPU
    collection.add(
        documents=batch_docs,
        embeddings=batch_embeddings.tolist(),
        ids=batch_ids,
        metadatas=batch_meta
    )
    
    current_batch = i // BATCH_SIZE + 1
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  └─ Lot {current_batch}/{total_batches} calculé et inséré.")

print(f"✅ {total_docs} lignes ajoutées dans la collection '{COLLECTION_NAME}' ({CHROMA_PATH}) avec succès !")