# preprocess/embedder.py
import os
import re
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================================
# ⚙️ Konfigurasi dasar (centralized)
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIRS = [
    BASE_DIR / "ocr_cache",
    BASE_DIR / "clean_scraping",
]
DB_DIR = BASE_DIR / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# model embedding (boleh diubah dari satu tempat)
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# helper buat embeddings (jika diperlukan)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

embeddings = get_embeddings()

# ==========================================================
# 🧩 Global Chroma Settings (PASTIKAN KONSISTEN DI SELURUH PROYEK)
# ==========================================================
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=str(DB_DIR),
)

# ==========================================================
# Utilities
# ==========================================================
def _safe_remove_dir(path: Path, max_attempts: int = 3, wait_seconds: float = 0.5) -> bool:
    """Hapus direktori (retry jika gagal karena file lock)."""
    for attempt in range(1, max_attempts + 1):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logging.warning("Gagal hapus %s (attempt %d/%d): %s", path, attempt, max_attempts, e)
            time.sleep(wait_seconds * attempt)
    return False

def load_cache(cache_dir: Path) -> List[Document]:
    """
    Baca file .txt hasil OCR di cache_dir dan kembalikan list[Document].
    (Fallback bila ocr_processor tidak menyediakan helper Document)
    """
    docs = []
    if not cache_dir.exists():
        logging.warning("Folder cache %s tidak ditemukan.", cache_dir)
        return docs

    for file in sorted(cache_dir.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": str(file)}))
        except Exception as e:
            logging.warning("Gagal membaca %s: %s", file, e)
    return docs

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?()'/\"-]", "", text)
    return text.strip()

def adaptive_splitter(text: str):
    length = len(text)
    if length > 8000:
        size, overlap = 800, 100
    elif length > 4000:
        size, overlap = 600, 80
    else:
        size, overlap = 400, 60
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

# ==========================================================
# Centralized factory untuk membuat instance Chroma dengan settings yang konsisten
# ==========================================================
def create_vectordb(embedding_function=None):
    """
    Membuat / mengembalikan instance Chroma yang menggunakan CHROMA_SETTINGS.
    embedding_function: jika None, gunakan get_embeddings()
    """
    if embedding_function is None:
        embedding_function = get_embeddings()
    # pastikan DB_DIR ada
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embedding_function, client_settings=CHROMA_SETTINGS)

# ==========================================================
# Build functions (full rebuild & incremental)
# ==========================================================
def build_chroma_from_cache():
    logging.info("Mulai build_chroma_from_cache()")
    all_docs: List[Document] = []
    for folder in CACHE_DIRS:
        all_docs.extend(load_cache(folder))

    if not all_docs:
        logging.warning("Tidak ada dokumen ditemukan di cache folder.")
        return

    logging.info("Membuat embedding dan menyimpan ke Chroma DB...")

    # Hapus DB_DIR dulu untuk menghindari settings mismatch (dev-mode)
    if DB_DIR.exists():
        logging.info("DB_DIR exists, mencoba hapus sebelum create: %s", DB_DIR)
        _safe_remove_dir(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    def _create():
        vect = Chroma.from_documents(
            documents=all_docs,
            embedding=get_embeddings(),
            persist_directory=str(DB_DIR),
            client_settings=CHROMA_SETTINGS,
        )
        vect.persist()
        return vect

    try:
        _create()
        logging.info("Chroma DB berhasil dibuat.")
    except ValueError as ve:
        logging.warning("ValueError saat create Chroma: %s", ve)
        logging.info("Mencoba hapus DB_DIR dan retry sekali lagi.")
        _safe_remove_dir(DB_DIR)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _create()
            logging.info("Chroma DB berhasil dibuat setelah retry.")
        except Exception as e:
            logging.error("Gagal membuat Chroma DB setelah retry: %s", e)
            raise
    except Exception as e:
        logging.error("Gagal membuat Chroma DB: %s", e)
        raise

def build_chroma_incremental(new_docs: Optional[List[Document]] = None, batch_size: int = 200):
    logging.info("Memulai incremental build Chroma DB...")
    # pastikan DB_DIR ada
    DB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore = create_vectordb(embedding_function=get_embeddings())
    except ValueError as ve:
        logging.warning("ValueError saat inisialisasi Chroma incremental (settings conflict): %s", ve)
        logging.info("Menghapus DB_DIR dan memaksa full rebuild sebagai fallback.")
        _safe_remove_dir(DB_DIR)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        build_chroma_from_cache()
        return

    if new_docs is None:
        all_docs = []
        for folder in CACHE_DIRS:
            if not folder.exists():
                continue
            for file in sorted(folder.glob("*.txt")):
                try:
                    content = file.read_text(encoding="utf-8", errors="ignore")
                    cleaned = clean_text(content)
                    splitter = adaptive_splitter(cleaned)
                    chunks = splitter.split_text(cleaned)
                    for chunk in chunks:
                        all_docs.append(Document(page_content=chunk, metadata={"source": str(file), "folder": folder.name}))
                except Exception as e:
                    logging.warning("Gagal memproses %s: %s", file, e)
        new_docs = all_docs

    if not new_docs:
        logging.warning("Tidak ada dokumen untuk incremental build.")
        return

    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            vectorstore.persist()
            logging.info("Batch %d ditambahkan (%d dokumen).", i//batch_size + 1, len(batch))
        except Exception as e:
            logging.error("Gagal menambahkan batch %d: %s", i//batch_size + 1, e)
            break

    logging.info("Incremental build selesai.")

def rebuild_chroma_db():
    build_chroma_from_cache()

# quick test
if __name__ == "__main__":
    build_chroma_from_cache()
