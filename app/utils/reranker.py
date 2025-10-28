# app/utils/reranker.py
import os
import logging
import torch
from typing import List, Tuple

# ==========================================================
# ⚙️ Konfigurasi Logging
# ==========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================================
# 🧠 Fallback CrossEncoder (local) + OpenAI reranker (cloud)
# ==========================================================
USE_LOCAL = False
try:
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    USE_LOCAL = True
    logging.info(f"✅ Menggunakan CrossEncoder lokal di device: {device}")
except Exception as e:
    logging.warning(f"⚠️ Tidak dapat memuat CrossEncoder lokal ({e}), fallback ke OpenAI reranker cloud.")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================================
# 🔎 Fungsi utama
# ==========================================================
def rerank_results(query: str, results: List[Tuple], top_n: int = 5):
    """
    Rerank hasil pencarian dari Chroma menggunakan:
      - CrossEncoder lokal (GPU/CPU), atau
      - OpenAI reranker berbasis embedding similarity (cloud)
    """
    if not results:
        return []

    if USE_LOCAL:
        # Lokal: SentenceTransformers CrossEncoder
        pairs = [(query, doc.page_content) for doc, _ in results]
        scores = reranker.predict(pairs)
    else:
        # Cloud: OpenAI reranker (cosine similarity)
        try:
            q_emb = client.embeddings.create(model="text-embedding-3-large", input=query).data[0].embedding
            scores = []
            for doc, _ in results:
                d_emb = client.embeddings.create(model="text-embedding-3-large", input=doc.page_content).data[0].embedding
                # hitung cosine similarity manual
                sim = cosine_similarity(q_emb, d_emb)
                scores.append(sim)
        except Exception as e:
            logging.error(f"❌ Gagal melakukan rerank cloud: {e}")
            return results[:top_n]

    reranked = sorted(
        zip([doc for doc, _ in results], scores),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_n]


# ==========================================================
# 🔢 Fungsi bantu: cosine similarity manual
# ==========================================================
def cosine_similarity(vec_a, vec_b):
    import numpy as np
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==========================================================
# 🧪 Test manual
# ==========================================================
if __name__ == "__main__":
    class DummyDoc:
        def __init__(self, text): self.page_content = text

    dummy_results = [
        (DummyDoc("Universitas Islam Indonesia memiliki program magister komputer."), 0.9),
        (DummyDoc("Kampus UII berada di Yogyakarta dan berfokus pada pendidikan Islam."), 0.85),
        (DummyDoc("Program sarjana informatika UII populer di Indonesia."), 0.8)
    ]

    query = "di mana kampus UII dan apa saja programnya?"
    reranked = rerank_results(query, dummy_results)
    for doc, score in reranked:
        logging.info(f"{score:.4f} | {doc.page_content[:80]}")
