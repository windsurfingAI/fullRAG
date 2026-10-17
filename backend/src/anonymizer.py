"""
backend/src/anonymizer.py
Gestion de la détection et du masquage des données sensibles (PII) via Microsoft Presidio.
"""

from typing import Tuple, List
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

# 1. Configuration du moteur NLP pour utiliser le modèle français
configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_sm"}  # Fallback
    ],
}

try:
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])
except Exception as e:
    print(f"⚠️ Initialisation Presidio avec config par défaut: {e}")
    analyzer = AnalyzerEngine()

# --- 🆕 AJOUT : RECONNAISSANCE DES TÉLÉPHONES FRANÇAIS ---
fr_phone_regex = r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}"
fr_phone_pattern = Pattern(
    name="fr_phone_pattern",
    regex=fr_phone_regex,
    score=0.85  # Score de confiance élevé
)
fr_phone_recognizer = PatternRecognizer(
    supported_entity="FR_PHONE_NUMBER",
    patterns=[fr_phone_pattern],
    supported_language="fr"
)
# On injecte notre règle dans le moteur
analyzer.registry.add_recognizer(fr_phone_recognizer)
# ---------------------------------------------------------

anonymizer = AnonymizerEngine()

# OPTION A : Uniquement les données ultra-sensibles.
# On exclut volontairement "PERSON" et "LOCATION" pour ne pas casser la recherche de films.
DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "FR_PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS"
]

def sanitize_input(text: str, language: str = "fr") -> Tuple[str, List[RecognizerResult]]:
    """
    Analyse le texte, détecte les PII critiques et renvoie le texte anonymisé.
    """
    if not text:
        return text, []

    # Détection
    results = analyzer.analyze(
        text=text,
        entities=DEFAULT_ENTITIES,
        language=language,
        score_threshold=0.6 # Confiance minimale de 60%
    )

    # Anonymisation
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    return anonymized_result.text, results