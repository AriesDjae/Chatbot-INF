# preprocess/embedder.py
import os
import re
import logging
import shutil
import time
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from dotenv import load_dotenv

# ==========================================================
# 🔧 Setup & Logging
# ==========================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================================
# ⚙️ Path Configuration (tanpa config.py)
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPING_DIR = BASE_DIR / "clean_scraping"
OCR_CACHE_DIR = BASE_DIR / "ocr_cache"
DB_DIR = BASE_DIR / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ⚙️ OpenAI Embedding Model
# ==========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDING_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise ValueError("❌ Environment variable OPENAI_API_KEY belum di-set.")

logging.info("🔹 Menggunakan model embedding: %s", EMBED_MODEL)
logging.info("🔹 Menggunakan API key dari environment variable OPENAI_API_KEY")

# ==========================================================
# 🧠 Setup Embeddings & Chroma Settings
# ==========================================================
def get_embeddings():
    """Inisialisasi embedding OpenAI."""
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

CHROMA_SETTINGS = Settings(
    persist_directory=str(DB_DIR),
    anonymized_telemetry=False,
)

# ==========================================================
# 🧹 Utility Functions
# ==========================================================
def _safe_remove_dir(path: Path, max_attempts: int = 3, wait_seconds: float = 0.5) -> bool:
    """Hapus direktori dengan retry jika gagal karena file lock."""
    for attempt in range(1, max_attempts + 1):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logging.warning("Gagal hapus %s (attempt %d/%d): %s", path, attempt, max_attempts, e)
            time.sleep(wait_seconds * attempt)
    return False


def clean_text(text: str) -> str:
    """Membersihkan teks agar siap di-embed."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?()'/\"-]", "", text)
    return text.strip()


def load_txt_folder(folder: Path) -> List[Document]:
    """Membaca semua file .txt dan mengembalikannya sebagai Document list."""
    docs = []
    if not folder.exists():
        logging.warning("⚠️ Folder %s tidak ditemukan.", folder)
        return docs

    txt_files = list(folder.rglob("*.txt"))
    logging.info("📁 Folder %s: ditemukan %d file .txt", folder.name, len(txt_files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for file in txt_files:
        try:
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            cleaned = clean_text(text)
            for chunk in splitter.split_text(cleaned):
                docs.append(Document(page_content=chunk, metadata={"source": str(file)}))
        except Exception as e:
            logging.error("❌ Gagal membaca %s: %s", file, e)

    return docs


# ==========================================================
# 🧩 Factory untuk Vector DB
# ==========================================================
def create_vectordb(embedding_function=None):
    """
    Membuat atau mengembalikan instance Chroma yang memakai CHROMA_SETTINGS.
    embedding_function: jika None, gunakan get_embeddings()
    """
    if embedding_function is None:
        embedding_function = get_embeddings()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function,  
        client_settings=CHROMA_SETTINGS
    )



# ==========================================================
# 🚀 Main: Build Combined Chroma DB
# ==========================================================
def build_chroma_combined(batch_size=200):
    logging.info("🚀 Memulai build_chroma_combined()")
    logging.info("📂 OCR_CACHE_DIR: %s", OCR_CACHE_DIR)
    logging.info("📂 SCRAPING_DIR: %s", SCRAPING_DIR)
    logging.info("📂 DB_DIR: %s", DB_DIR)

    ocr_docs = load_txt_folder(OCR_CACHE_DIR)
    scraping_docs = load_txt_folder(SCRAPING_DIR)

    logging.info("📄 Total dokumen: OCR=%d | SCRAPING=%d", len(ocr_docs), len(scraping_docs))
    logging.info("🚀 Memulai proses build_chroma_combined() | batch_size=%s", batch_size)
    all_docs = ocr_docs + scraping_docs

    if not all_docs:
        logging.warning("⚠️ Tidak ada dokumen ditemukan. Proses dihentikan.")
        return False

    if DB_DIR.exists():
        logging.info("🗑 Menghapus DB lama di %s", DB_DIR)
        _safe_remove_dir(DB_DIR)
        DB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        vect = Chroma.from_documents(
            documents=all_docs,
            embedding=get_embeddings(),
            persist_directory=str(DB_DIR),
            client_settings=CHROMA_SETTINGS,
        )
        vect.persist()
        logging.info("✅ Chroma DB berhasil dibuat (%d dokumen)", len(all_docs))
        return True
    except Exception as e:
        logging.error("❌ Gagal membuat Chroma DB: %s", e)
        return False


# ==========================================================
# 🧪 Manual Test
# ==========================================================
if __name__ == "__main__":
    build_chroma_combined()
