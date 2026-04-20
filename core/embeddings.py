import chromadb
from langchain_openai import OpenAIEmbeddings

_client = chromadb.HttpClient(host="localhost", port=8001)
_emb = OpenAIEmbeddings(model="text-embedding-3-small")   # 1536-dim

def index_chunks(doc_id: str, chunks: list[str]):
    col = _client.get_or_create_collection(doc_id, metadata={"hnsw:space": "cosine"})
    vecs = _emb.embed_documents(chunks)
    col.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=vecs,
        documents=chunks,
    )

def retrieve_semantic(doc_id: str, query: str, k: int = 5) -> list[str]:
    col = _client.get_collection(doc_id)
    results = col.query(query_embeddings=[_emb.embed_query(query)], n_results=k)
    return results["documents"][0]