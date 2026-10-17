"""
Tests unitaires pour le module d'anonymisation (Presidio).
"""

from anonymizer import sanitize_input

def test_mask_email():
    """Vérifie que les emails sont correctement masqués."""
    raw_text = "Contactez john.doe@cinema.fr pour les billets."
    clean_text, detected = sanitize_input(raw_text)
    
    assert "<EMAIL_ADDRESS>" in clean_text
    assert "john.doe@cinema.fr" not in clean_text

def test_mask_french_phone_numbers():
    """Vérifie que les numéros français (avec ou sans espaces) sont masqués."""
    phones = [
        "06 12 34 56 78",
        "0612345678",
        "+336 12 34 56 78"
    ]
    for phone in phones:
        raw_text = f"Mon numéro est le {phone}."
        clean_text, _ = sanitize_input(raw_text)
        
        assert "<FR_PHONE_NUMBER>" in clean_text, f"Échec sur le format: {phone}"
        assert phone not in clean_text

def test_keep_names_and_locations():
    """Vérifie que les noms de personnes et lieux NE SONT PAS masqués (Option A)."""
    raw_text = "Jean Dujardin tourne un film à Paris."
    clean_text, _ = sanitize_input(raw_text)
    
    # Ils doivent rester intacts pour la recherche vectorielle
    assert "Jean Dujardin" in clean_text
    assert "Paris" in clean_text

def test_keep_non_sensitive_numbers():
    """Vérifie que les années ou numéros de carte de fidélité ne sont pas confondus avec des téléphones."""
    raw_text = "Ma carte UGC est la 1048291048 et on est en 2023."
    clean_text, _ = sanitize_input(raw_text)
    
    assert "1048291048" in clean_text
    assert "2023" in clean_text

def test_complex_prompt():
    """Vérifie un prompt réel complet (cas d'intégration)."""
    raw_text = "Julien Delmas (julien@mail.com, 06 45 89 12 30) cherche un film."
    clean_text, detected = sanitize_input(raw_text)
    
    assert "Julien Delmas" in clean_text
    assert "<EMAIL_ADDRESS>" in clean_text
    assert "<FR_PHONE_NUMBER>" in clean_text

def test_custom_miyazaki_prompt():
    """Vérifie un cas spécifique avec nom composé et réalisateur mal orthographié."""
    raw_text = "Marc Antoine marc.antoine@hotmail.fr cherche un film de miyazakkki. est-ce que tu pourrais l'aider stp?"
    clean_text, detected = sanitize_input(raw_text)
    
    # L'email doit disparaître au profit de la balise
    assert "<EMAIL_ADDRESS>" in clean_text
    assert "marc.antoine@hotmail.fr" not in clean_text
    
    # Le nom de l'utilisateur doit rester
    assert "Marc Antoine" in clean_text
    
    # Le nom du réalisateur (même avec la faute) doit rester pour la recherche
    assert "miyazakkki" in clean_text