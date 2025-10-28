from sentence_transformers import CrossEncoder
import torch

# Deteksi otomatis device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gunakan CPU jika CUDA tidak tersedia (supaya aman di semua mesin)
print(f"[INFO] Memuat CrossEncoder di device: {device}")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def rerank_results(query, results, top_n=5):
    """
    Rerank hasil pencarian dari Chroma menggunakan CrossEncoder
    """
    if not results:
        return []

    pairs = [(query, doc.page_content) for doc, _ in results]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip([doc for doc, _ in results], scores),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_n]

