import os
import re
import logging
from pathlib import Path
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy

# ==========================================================
# ⚙️ 1️⃣ LOAD EMBEDDING MODEL
# ==========================================================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

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
ref_embeds = model.encode(REFERENCE_SENTENCES, convert_to_tensor=True)

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
def filter_semantic(sentences: List[str], entities: List[str], threshold: float = 0.25) -> List[str]:
    """Pertahankan lebih banyak kalimat, terutama yang mengandung entitas."""
    if not sentences:
        return []
    embeds = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embeds, ref_embeds)
    max_scores = np.max(scores.cpu().numpy(), axis=1)
    cleaned = []
    for s, sc in zip(sentences, max_scores):
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
