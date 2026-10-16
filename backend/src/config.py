class RAGConfig:
    CHROMA_K = 20
    BM25_K = 25
    BM25_WEIGHT = 0.3
    CHROMA_WEIGHT = 0.7
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    REDIS_WINDOW_SIZE = 4
    DEFAULT_PROVIDER = "gemini"