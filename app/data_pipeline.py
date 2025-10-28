# preprocess/data_pipeline.py
import sys, os, logging
from pathlib import Path
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from preprocess.embedder import (
    build_chroma_combined,
    DB_DIR,
)
from preprocess.ocr_processor import process_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def full_pipeline(force_ocr: bool = False, rebuild_db: bool = True):
    """
    Pipeline utama yang:
    1️⃣ Jalankan OCR (jika force_ocr=True)
    2️⃣ Bangun / update Chroma DB dari semua sumber (OCR + scraping + docs)
    """

    logging.info("🚀 Memulai pipeline penuh | force_ocr=%s | rebuild_db=%s", force_ocr, rebuild_db)

    # === Step 1: OCR ===
    try:
        process_all(force=force_ocr)
        logging.info("✅ OCR selesai dijalankan (force_ocr=%s).", force_ocr)
    except Exception as e:
        logging.error("❌ Gagal menjalankan OCR: %s", e)

    # === Step 2: Build / Update Database ===
    if rebuild_db:
        try:
            logging.info("🧠 Membangun Chroma DB (combined)...")
            success = build_chroma_combined(batch_size=200)
            if success:
                logging.info("✅ Chroma DB berhasil dibangun / diperbarui.")
            else:
                logging.warning("⚠️ Tidak ada dokumen baru ditemukan untuk DB.")
        except Exception as e:
            logging.error("❌ Gagal build Chroma DB: %s", e)
    else:
        if not Path(DB_DIR).exists():
            logging.warning("⚠️ Database belum ada, membangun baru...")
            build_chroma_combined()
        else:
            logging.info("ℹ️ Database sudah ada, lewati proses rebuild.")

    logging.info("🏁 Pipeline selesai.")
    return True


# ==========================================================
# Untuk menjalankan langsung dari CLI
# ==========================================================
if __name__ == "__main__":
    full_pipeline(force_ocr=False, rebuild_db=True)