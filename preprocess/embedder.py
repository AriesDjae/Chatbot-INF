import os
import re
import logging
import shutil
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings

# ==========================================================
# 🔧 Setup & Logging
# ==========================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================================
# ⚙️ Path Configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPING_DIR = BASE_DIR / "clean_scraping"
OCR_CACHE_DIR = BASE_DIR / "ocr_cache"
DB_DIR = BASE_DIR / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ⚙️ Embedding Configuration
# ==========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDING_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY belum di-set di .env!")

def get_embeddings():
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
def _safe_remove_dir(path: Path, max_attempts=3, wait_seconds=0.5):
    for attempt in range(1, max_attempts + 1):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logging.warning("Gagal hapus %s (percobaan %d/%d): %s", path, attempt, max_attempts, e)
            time.sleep(wait_seconds * attempt)
    return False


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?()'/\"-]", "", text)
    return text.strip()


def load_txt_folder(folder: Path) -> List[Document]:
    docs = []
    if not folder.exists():
        logging.warning("⚠️ Folder %s tidak ditemukan.", folder)
        return docs

    txt_files = list(folder.rglob("*.txt"))
    logging.info("📁 %s: %d file ditemukan", folder.name, len(txt_files))

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
    if embedding_function is None:
        embedding_function = get_embeddings()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function,
        client_settings=CHROMA_SETTINGS
    )

# ==========================================================
# 🚀 Build Combined Vector DB
# ==========================================================
def build_chroma_combined(batch_size=200):
    logging.info("🚀 Memulai build_chroma_combined() [DEBUG MODE]")
    logging.info("📂 OCR_CACHE_DIR: %s", OCR_CACHE_DIR.resolve())
    logging.info("📂 SCRAPING_DIR: %s", SCRAPING_DIR.resolve())
    logging.info("📂 DB_DIR: %s", DB_DIR.resolve())

    # 1. Load dokumen
    ocr_docs = load_txt_folder(OCR_CACHE_DIR)
    scraping_docs = load_txt_folder(SCRAPING_DIR)
    all_docs = ocr_docs + scraping_docs

    logging.info("📄 Jumlah dokumen: OCR=%d, SCRAPING=%d, TOTAL=%d",
                 len(ocr_docs), len(scraping_docs), len(all_docs))

    # 2. Jika kosong
    if not all_docs:
        logging.warning("⚠️ Tidak ada dokumen ditemukan, hentikan proses.")
        return False

    # 3. Hapus DB lama
    if DB_DIR.exists():
        _safe_remove_dir(DB_DIR)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Inisialisasi embeddings
    try:
        embeddings = get_embeddings()
        test_vec = embeddings.embed_query("uji coba koneksi embedding")
        logging.info("✅ Tes embedding berhasil, panjang vektor: %d", len(test_vec))
    except Exception as e:
        logging.error("❌ Gagal inisialisasi embedding: %s", e)
        return False

    # 5. Bangun Chroma
    try:
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
            client_settings=CHROMA_SETTINGS
        )
        vectordb.persist()
        logging.info("✅ Chroma DB berhasil dibuat dengan %d dokumen", len(all_docs))

        # 6. Cek apakah file muncul
        sqlite_file = DB_DIR / "chroma.sqlite3"
        index_dir = DB_DIR / "index"
        if sqlite_file.exists() and index_dir.exists():
            logging.info("🎉 File DB berhasil disimpan: %s", sqlite_file)
        else:
            logging.warning("⚠️ File DB belum muncul di: %s", DB_DIR)
        return True

    except Exception as e:
        logging.error("❌ Gagal membuat Chroma DB: %s", e)
        return False


# ==========================================================
# 🧪 Manual Test
# ==========================================================
if __name__ == "__main__":
    build_chroma_combined()
