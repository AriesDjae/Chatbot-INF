# preprocess/data_pipeline.py
import sys, os, logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from preprocess.embedder import (
    build_chroma_from_cache,
    build_chroma_incremental,
    DB_DIR,
    CHROMA_SETTINGS,
    create_vectordb,
)
from preprocess.ocr_processor import process_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def full_pipeline(force_ocr: bool = False, rebuild_db: bool = True):
    logging.info("Memulai pipeline penuh | force_ocr=%s | rebuild_db=%s", force_ocr, rebuild_db)
    # jalankan OCR (akan menyimpan .txt ke BASE_DIR/ocr_cache)
    process_all(force=force_ocr)

    if rebuild_db:
        try:
            # incremental akan membuat instance Chroma dengan settings yang sama
            build_chroma_incremental(batch_size=200)
        except Exception as e:
            logging.error("Gagal incremental build: %s", e)
            logging.info("Coba fallback ke full rebuild...")
            build_chroma_from_cache()
    else:
        if not Path(DB_DIR).exists():
            logging.warning("Database belum ada, membuat baru...")
            build_chroma_from_cache()
        else:
            logging.info("Database sudah ada, skip rebuild.")

    logging.info("Pipeline selesai.")
    return True

if __name__ == "__main__":
    full_pipeline(force_ocr=False, rebuild_db=True)
