import os
import re
import logging
from pathlib import Path
from typing import List
import numpy as np
from openai import OpenAI
import spacy

# ==========================================================
# ⚙️ 1️⃣ LOAD EMBEDDING MODEL (OpenAI)
# ==========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDING_MODEL")

if not OPENAI_API_KEY:
    raise ValueError(
        "Environment variable OPENAI_API belum di-set. "
        "Setel terlebih dahulu sebelum menjalankan aplikasi."
    )

if not EMBED_MODEL:
    raise ValueError(
        "Environment variable EMBEDING_MODEL belum di-set. "
        "Setel terlebih dahulu sebelum menjalankan aplikasi."
    )

logging.info("Memuat OpenAI embedding model untuk data cleaner: %s (dari EMBEDING_MODEL)", EMBED_MODEL)
logging.info("Menggunakan API key dari environment variable OPENAI_API")

# Inisialisasi OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Mendapatkan embeddings dari OpenAI untuk list teks."""
    try:
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        logging.error("Gagal mendapatkan embeddings dari OpenAI: %s", e)
        raise

# ==========================================================
# ⚙️ 2️⃣ LOAD NER MODEL (dengan fallback)
# ==========================================================
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except Exception:
    print("⚠️ Model NER spaCy tidak ditemukan, menggunakan MultiLanguage...")
    from spacy.lang.xx import MultiLanguage
    nlp = MultiLanguage()

# ==========================================================
# 🧭 3️⃣ REFERENSI KALIMAT UNTUK PEMBANDING
# ==========================================================
REFERENCE_SENTENCES = [
    "Mahasiswa wajib melakukan presensi di sistem kampus.",
    "Panduan ini menjelaskan langkah-langkah penting.",
    "Universitas Islam Indonesia menyediakan layanan akademik.",
    "Pendaftaran mahasiswa dilakukan secara online.",
]

# Lazy loading untuk reference embeddings
_ref_embeds_cache = None

def get_ref_embeddings() -> np.ndarray:
    """Mendapatkan reference embeddings dengan caching."""
    global _ref_embeds_cache
    if _ref_embeds_cache is None:
        _ref_embeds_cache = get_embeddings(REFERENCE_SENTENCES)
    return _ref_embeds_cache

# ==========================================================
# 🧹 4️⃣ BASIC CLEANER
# ==========================================================
def regex_clean(text: str) -> str:
    # Hapus karakter tidak perlu tapi jangan terlalu agresif
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = re.sub(r"Halaman\s*\d+", " ", text)
    text = re.sub(r"[^\w\s.,!?()’'\"%-]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ==========================================================
# 🧠 5️⃣ ENTITY HELPER
# ==========================================================
def extract_entities(text: str) -> List[str]:
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
    custom_entities = [
        "Universitas Islam Indonesia", "FTI UII", "SIM Presensi",
        "Mahasiswa", "Dosen", "SKP", "Presensi", "Kampus"
    ]
    return list(set(entities + custom_entities))

# ==========================================================
# ✂️ 6️⃣ SPLIT SENTENCES (lebih natural)
# ==========================================================
def split_sentences(text: str) -> List[str]:
    text = regex_clean(text)
    # Gunakan newline sebagai pemisah alami jika ada
    chunks = re.split(r"\n+", text)
    sentences = []
    for chunk in chunks:
        # Pisahkan lagi berdasarkan tanda baca
        for s in re.split(r"(?<=[.!?])\s+", chunk.strip()):
            if len(s.strip()) > 3:
                sentences.append(s.strip())
    return sentences

# ==========================================================
# 🧩 7️⃣ SEMANTIC FILTER (lebih lembut)
# ==========================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Menghitung cosine similarity antara dua array embeddings."""
    # Normalize
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    # Compute similarity
    return np.dot(a_norm, b_norm.T)

def filter_semantic(sentences: List[str], entities: List[str], threshold: float = 0.25) -> List[str]:
    """Pertahankan lebih banyak kalimat, terutama yang mengandung entitas."""
    if not sentences:
        return []
    # Batch processing untuk efisiensi
    batch_size = 100
    all_scores = []
    
    ref_embeds = get_ref_embeddings()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            embeds = get_embeddings(batch)
            scores = cosine_similarity(embeds, ref_embeds)
            max_scores = np.max(scores, axis=1)
            all_scores.extend(max_scores)
        except Exception as e:
            logging.warning("Gagal memproses batch embeddings: %s", e)
            # Fallback: terima semua kalimat jika embedding gagal
            all_scores.extend([threshold] * len(batch))
    
    cleaned = []
    for s, sc in zip(sentences, all_scores):
        if sc >= threshold or any(ent.lower() in s.lower() for ent in entities):
            cleaned.append(s)
    return cleaned

# ==========================================================
# 🧩 8️⃣ GROUPING NATURAL
# ==========================================================
def group_sentences(sentences: List[str], max_group_len: int = 500) -> List[str]:
    groups = []
    current = ""
    for s in sentences:
        # Jangan terlalu panjang dalam satu baris
        if len(current) + len(s) + 1 < max_group_len:
            current += (" " + s)
        else:
            groups.append(current.strip())
            current = s
    if current:
        groups.append(current.strip())
    return groups

# ==========================================================
# 🪶 9️⃣ FORMAT OUTPUT
# ==========================================================
def format_text(paragraphs: List[str]) -> str:
    formatted = []
    for p in paragraphs:
        # Bungkus teks panjang supaya tidak terlalu ke kanan
        wrapped = "\n".join(re.findall(r".{1,100}(?:\s+|$)", p))
        formatted.append(wrapped.strip())
    return "\n\n".join(formatted)

# ==========================================================
# 🏁 🔟 PIPELINE UTAMA
# ==========================================================
def clean_text_v3_3(raw_text: str) -> str:
    sentences = split_sentences(raw_text)
    entities = extract_entities(raw_text)
    filtered = filter_semantic(sentences, entities)
    grouped = group_sentences(filtered)
    formatted = format_text(grouped)
    return formatted.strip()

# ==========================================================
# 💾 1️⃣1️⃣ BATCH MODE
# ==========================================================
def clean_folder_v3_3(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in input_dir.glob("*.txt"):
        try:
            raw = file.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text_v3_3(raw)
            save_path = output_dir / f"{file.stem}_cleaned.txt"
            save_path.write_text(cleaned, encoding="utf-8")
            logging.info(f"✅ {file.name} → {save_path.name}")
        except Exception as e:
            logging.warning(f"❌ Gagal membersihkan {file.name}: {e}")

# ==========================================================
# 🚀 MAIN RUN
# ==========================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    BASE_DIR = Path(__file__).resolve().parents[2]
    input_dir = BASE_DIR / "hasil_scraping"
    output_dir = BASE_DIR / "clean_scraping"
    clean_folder_v3_3(input_dir, output_dir)
