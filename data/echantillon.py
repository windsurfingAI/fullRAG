import os
import pandas as pd

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "raw/liste-des-films-sortis-dans-les-salles-de-cinema-en-france-de-1945-a-2025.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "echantillon_100_lignes.csv")

def extraire_echantillon(input_path, output_path, nb_lignes=100):
    try:
        print(f"📂 Lecture du fichier : {input_path}")
        
        # Lecture du CSV avec le bon séparateur
        df = pd.read_csv(input_path, sep=";", encoding="utf-8")
        
        # Sécurité : on prend 100 lignes, ou le maximum disponible si le fichier est plus petit
        taille_reelle = min(nb_lignes, len(df))
        
        # Sélection aléatoire (random_state=42 garantit qu'on tire toujours les mêmes lignes si on relance)
        df_echantillon = df.sample(n=taille_reelle, random_state=42)
        
        # Sauvegarde du nouveau fichier
        df_echantillon.to_csv(output_path, sep=";", index=False, encoding="utf-8")
        
        print(f"✅ Échantillon de {taille_reelle} lignes sauvegardé dans : {output_path}")
        
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier source '{input_path}' est introuvable.")
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")

# Exécution
if __name__ == "__main__":
    extraire_echantillon(INPUT_CSV, OUTPUT_CSV, nb_lignes=100)