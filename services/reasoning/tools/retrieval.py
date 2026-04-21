"""Hybrid retrieval: BM25 + semantic search via Reciprocal Rank Fusion.
BM25 catches exact keyword matches; semantic catches paraphrases.
RRF fuses both without requiring score calibration.

BM25(q, d) = sum over t in q of IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))

RRF(d) = sum over rankers r of 1/(k0 + rank_r(d)), k0=60

ref: BM25: https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-356.pdf
ref: RRF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

from rank_bm25 import BM25Okapi
from core.embeddings import retrieve_semantic

def retrieve_bm25(chunks: list[str], query: str, k: int = 5) -> list[str]:
    """BM25 retrieval over in-memory chunks."""
    if not chunks:
        return []
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    scores = bm25.get_scores(query.lower().split())
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_k]

def rank_by_bm25(chunks: list[str], query: str) -> list[int]:
    """Return chunk indices ranked by BM25 score (best first)."""
    if not chunks:
        return []
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

def rank_by_semantic(doc_id: str, query: str, n: int = 20) -> list[int]:
    """Return chunk indices ranked by semantic similarity."""
    # Unpack the documents and IDs from our updated function
    docs, sem_ids = retrieve_semantic(doc_id, query, k=n)
    
    indices = []
    for chunk_id in sem_ids:
        # chunk_id looks like "doc123_4" -> extract the "4"
        try:
            idx_str = chunk_id.split("_")[-1]
            indices.append(int(idx_str))
        except (ValueError, IndexError):
            continue
            
    return indices

def retrieve_hybrid(
    chunks: list[str],
    doc_id: str,
    query: str,
    k: int = 5,
) -> list[str]:
    """Reciprocal Rank Fusion over BM25 and semantic retrieval."""
    if not chunks:
        return []
        
    bm25_ranks = rank_by_bm25(chunks, query)
    
    # We fetch exact indices instead of mapping strings
    sem_ranks = rank_by_semantic(doc_id, query, n=min(k * 3, len(chunks)))

    # RRF scoring with k0=60
    scores: dict[int, float] = {}
    
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
        
    for rank, idx in enumerate(sem_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)

    if not scores:
        return chunks[:k]

    top_indices = sorted(scores, key=scores.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices]