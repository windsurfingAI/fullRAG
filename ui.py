import streamlit as st
import requests

st.set_page_config(page_title="RAG Local & Gratuit", layout="centered")
st.title("🤖 Chatbot Multi-Modèles")

# Barre latérale pour choisir le cerveau du Chatbot
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox(
    "Modèle de langage (LLM)",
    options=["gemini", "mistral"],
    format_func=lambda x: "Gemini 2.5 (Cloud Gratuit)" if x == "gemini" else "Mistral 7B (Local via Ollama)"
)

# Initialisation de l'historique de discussion
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
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "query": user_query,
                        "provider": provider,
                        "top_k": 5
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    
                    # Affichage des sources issues de la base vectorielle
                    with st.expander("🔍 Sources locales identifiées par le Reranker"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**Source {i} :** {source}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Erreur du backend : {response.json().get('detail')}")
            
            except requests.exceptions.ConnectionError:
                st.error("Impossible de joindre FastAPI (port 8000). Lancez le serveur d'abord !")