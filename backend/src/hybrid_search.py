# Fichier: search_hybrid.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def get_hybrid_retriever(chroma_path="data/chroma_db", collection_name="documents", model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'} # ou 'cpu'
    )

    # 1. Vecteur (Chroma)
    vectorstore = Chroma(
        persist_directory=chroma_path,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Mots-clés (BM25)
    all_data = vectorstore.get()
    all_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_data['documents'], all_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5

    # 3. Fusion Hybride
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]
    )
    
    return hybrid_retriever