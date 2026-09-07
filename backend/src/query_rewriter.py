import os
import requests

def rewrite_query(query: str, chat_history: list, provider: str, gemini_client=None) -> str:
    """
    Réécrit la requête de l'utilisateur en utilisant l'historique pour résoudre les anaphores (il, elle, etc.).
    Code paré pour la production avec fallback sur la question originale en cas d'échec.
    """
    # 1. S'il n'y a pas d'historique, inutile de gaspiller des tokens/du temps
    if not chat_history:
        return query
    
    # 2. Formatage de l'historique
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # 3. Le prompt système strict (On lui interdit formellement de répondre à la question)
    prompt = f"""Tu es un expert en linguistique. 
Ton SEUL ET UNIQUE rôle est de reformuler la "Nouvelle question" pour qu'elle soit compréhensible de manière isolée, sans avoir besoin de lire l'historique. 
Tu dois remplacer les pronoms (il, elle, ce, le, son, etc.) par les sujets explicites mentionnés dans l'historique.
NE RÉPONDS JAMAIS À LA QUESTION. Renvoie UNIQUEMENT la question reformulée.

HISTORIQUE :
{history_text}

Nouvelle question : {query}
Question reformulée :"""

    # 4. Appel au LLM choisi par l'utilisateur
    try:
        if provider == "gemini" and gemini_client:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            return response.text.strip()
            
        elif provider == "mistral":
            ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            # Un timeout court (ex: 5s) est crucial en prod pour la réécriture, 
            # afin de ne pas bloquer l'expérience utilisateur
            res = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=5
            )
            if res.status_code == 200:
                return res.json().get("response", query).strip()
                
    except Exception as e:
        print(f"⚠️ Échec du rewriting, fallback sur la requête d'origine : {e}")
        
    # Fallback de sécurité : si le LLM échoue ou time out, on renvoie la requête originale
    return query