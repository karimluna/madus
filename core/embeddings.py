"""Embedding backends: OpenAI API, local BAAI/bge-m3, or ColPali.
The text embedder uses mean polling + L2 normalization so cosine
similarity reduces the dot product in ChromaDB's index

ref: HNSW, https://arxiv.org/abs/1603.09320
ref: ColPali, https://arxiv.org/abs/2407.01449
"""

import os 
import logging
from typing import Optional

import chromadb

from core.config import get_settings

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """BAAI/bge-m3 is 1024-dim, runs on CPU, no API cost. Remember:
    The formula for mean pooling is v = sum(M_i * H_i) / sum(M_i)
    then L2 normalization."""
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.torch = torch

    def _encode(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tok(
            texts, 
            padding=True, 
            truncation=True,
            max_length=512, 
            return_tensors="pt",
        )
        with self.torch.no_grad():
            h = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vecs = (h * mask).sum(1) / mask.sum(1)
        return self.torch.nn.functional.normalize(vecs, p=2, dim=1).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
    

class ColPaliEmbeddings:
    """ColPali page-level image retrieval using MaxSim.
    Needs more GPU VRAM than SigLIP. So better use VISION_BACKEND="local"
    instead of "copali" if low GPU VRAM available.
    """
    def __init__(self):
        import torch
        from colpali_engine.models import ColFlor, ColFlorProcessor
        self.torch = torch
        self.model = ColFlor.from_pretrained(
            "ahmed-masry/ColFlor",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = ColFlorProcessor.from_pretrained("ahmed-masry/ColFlor")

    def embed_page_images(self, pil_images: list) -> list:
        inputs = self.processor.process_images(pil_images).to(self.model.device)
        with self.torch.no_grad():
            return self.model(**inputs)

    def embed_query(self, text: str):
        inputs = self.processor.process_queries([text]).to(self.model.device)
        with self.torch.no_grad():
            return self.model(**inputs)

    def maxsim_score(self, query_emb, page_emb) -> float:
        scores = self.torch.einsum("id,jd->ij", query_emb[0], page_emb[0])
        return scores.max(dim=1).values.sum().item()


class SigLIPEmbeddings:
    """Cheaper alternative for image retrieval.
    ColFlow would be excellent with enough computation power,
    SigLIP is designed for efficient retrieval and uses simple
    cosine similarity which is enough for local use"""
    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    def embed_page_images(self, pil_images: list):
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)

        return image_emb  # shape: [N, D]

    def embed_query(self, text: str):
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with self.torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        return text_emb  # shape: [1, D]

    def similarity(self, query_emb, page_emb):
        # normalize (important!)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        page_emb = page_emb / page_emb.norm(dim=-1, keepdim=True)

        return (page_emb @ query_emb.T).squeeze(-1)


def get_text_embedder():
    s = get_settings()
    if s.embedding_backend == "local":
        return LocalEmbeddings()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_image_embedder():
    s = get_settings()
    if s.vision_backend == "siglip":
        return SigLIPEmbeddings()
    if s.vision_backend == "colflow":
        return ColPaliEmbeddings()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_chroma_client() -> Optional[chromadb.HttpClient]:
    s = get_settings()
    try:
        client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
        client.heartbeat()
        return client
    except Exception:
        logger.warning("ChromaDB unavailable at %s:%s", s.chroma_host, s.chroma_port)
        return None
    

def index_chunks(doc_id: str, chunks: list[str]) -> None:
    """Index that chunks in ChromaDB for semantic retrieval.
    Silently skips if ChromaDB is unavailable"""
    client = _get_chroma_client()
    if client is None or not chunks:
        return
    emb = get_text_embedder()
    col = client.get_or_create_collection(doc_id, metadata={"hnsw:space": "cosine"})
    vecs = emb.embed_documents(chunks)
    col.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=vecs,
        documents=chunks,
    )


def retrieve_semantic(doc_id: str, query: str, k: int = 5) -> list[str]:
    """Semantic retrieval via ChromaDB HNSW index. Returns (documents, ids)"""
    client = _get_chroma_client()
    if client is None:
        return []
    try:
        col = client.get_collection(doc_id)
    except Exception:
        return []
    emb = get_text_embedder()
    results = col.query(query_embeddings=[emb.embed_query(query)], n_results=k)
    return results["documents"][0] if results["documents"] else [], results["ids"][0]