import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# Priorité à l'environnement Docker, sinon .env, sinon localhost
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="RAG Local & Gratuit", layout="centered")
st.title("🤖 Chatbot Multi-Modèles")

# 1. Gestion de l'identifiant de session unique pour Redis
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Barre latérale pour choisir le cerveau du Chatbot
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox(
    "Modèle de langage (LLM)",
    options=["gemini", "mistral"],
    format_func=lambda x: "Gemini 2.5 (Cloud Gratuit)" if x == "gemini" else "Mistral 7B (Local via Ollama)"
)

# Affichage de la session active en bas de la barre latérale (utile pour le debug)
st.sidebar.caption(f"ID Session : `{st.session_state.session_id[:8]}...`")

# Initialisation de l'historique UI Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Saisie utilisateur
if user_query := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Appel de l'API FastAPI
    with st.chat_message("assistant"):
        with st.spinner(f"Réflexion avec {provider.capitalize()}..."):
            try:
                # Payload conforme au ChatRequest du app.py
                payload = {
                    "session_id": st.session_state.session_id,
                    "query": user_query,
                    "provider": provider,
                    "top_k": 10,
                    "user_profile": {
                        "name": "Thomas",
                        "role": "Analyste SOC",
                        "department": "Cybersécurité",
                        "clearance": "Standard"
                    }
                }

                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    
                    # Affichage des sources issues de la recherche hybride + reranking
                    with st.expander("🔍 Sources locales identifiées par le Reranker"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**Source {i} :** {source}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Erreur du backend ({response.status_code}) : {detail}")
            
            except requests.exceptions.ConnectionError:
                st.error(f"Impossible de joindre le backend à l'adresse : {BACKEND_URL}")