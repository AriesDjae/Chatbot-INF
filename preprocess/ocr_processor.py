#!/usr/bin/env python3
# preprocess/ocr_processor.py
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import traceback
import logging
import os

# gunakan BASE_DIR project agar path sinkron dengan embedder
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "dokumen_kampus"
OCR_CACHE_DIR = BASE_DIR / "ocr_cache"
DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

POPPLER_PATH = os.getenv("POPPLER_PATH", r"D:\Poppler\poppler-25.07.0\Library\bin")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"D:\Tesseract\tesseract.exe")
OCR_LANGS = ["ind", "eng"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Utility: baca cache (.txt)
# ---------------------------
def load_cache(cache_dir: Path = OCR_CACHE_DIR) -> List[Tuple[Path, str]]:
    docs: List[Tuple[Path, str]] = []
    if not cache_dir.exists():
        logger.debug("load_cache: cache dir tidak ada: %s", cache_dir)
        return docs
    for f in sorted(cache_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".txt":
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                docs.append((f, txt))
            except Exception as e:
                logger.warning("Gagal baca file cache %s: %s", f, e)
    logger.info("load_cache: menemukan %d file .txt di %s", len(docs), cache_dir)
    return docs

def load_cache_texts(cache_dir: Path = OCR_CACHE_DIR) -> List[str]:
    return [t for (_, t) in load_cache(cache_dir)]

# ---------------------------
# Lazy imports untuk OCR
# ---------------------------
def _ensure_pytesseract():
    try:
        import pytesseract
    except Exception as e:
        logger.error("Module pytesseract tidak ditemukan. Install: pip install pytesseract")
        raise
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return pytesseract

def _ensure_pdf2image():
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        logger.error("Module pdf2image tidak ditemukan. Install: pip install pdf2image")
        raise
    return convert_from_path

from PIL import Image

# ---------------------------
# OCR helpers
# ---------------------------
def image_to_text(img: Image.Image, langs: List[str] = OCR_LANGS) -> str:
    pytesseract = _ensure_pytesseract()
    last_err = None
    for lang in langs:
        try:
            text = pytesseract.image_to_string(img, lang=lang)
            if text and text.strip():
                return text
        except Exception as e:
            last_err = e
    try:
        text = pytesseract.image_to_string(img)
        return text if text and text.strip() else ""
    except Exception as e:
        logger.warning("OCR fallback gagal: %s (last_err=%s)", e, last_err)
        return ""

def pdf_to_images(pdf_path: Path, poppler_path: Optional[str] = None) -> List[Image.Image]:
    convert_from_path = _ensure_pdf2image()
    kwargs = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    images = convert_from_path(str(pdf_path), **kwargs)
    return images

def ocr_single_pdf(pdf_path: Path, out_txt_path: Path, poppler_path: Optional[str] = POPPLER_PATH, langs: List[str] = OCR_LANGS) -> Tuple[bool, str]:
    try:
        suffix = pdf_path.suffix.lower()
        pages_text = []
        if suffix == ".pdf":
            imgs = pdf_to_images(pdf_path, poppler_path=poppler_path)
            if not imgs:
                return False, f"Gagal konversi PDF -> gambar: {pdf_path}"
            for i, img in enumerate(imgs, start=1):
                logger.info("OCR halaman %d/%d dari %s", i, len(imgs), pdf_path.name)
                text = image_to_text(img, langs=langs)
                pages_text.append(f"\n=== Halaman {i} ===\n{text}")
        else:
            logger.info("OCR gambar: %s", pdf_path.name)
            img = Image.open(pdf_path)
            text = image_to_text(img, langs=langs)
            pages_text.append(text)

        full_text = "\n".join(pages_text).strip()
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        out_txt_path.write_text(full_text, encoding="utf-8")
        return True, f"OK ({len(pages_text)} halaman)"
    except Exception as e:
        traceback.print_exc()
        return False, f"Exception saat OCR: {e}"

def list_input_files(input_dir: Path = DEFAULT_INPUT_DIR) -> List[Path]:
    files = []
    if not input_dir.exists():
        return files
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            files.append(p)
    return files

def process_all(input_dir: Path = DEFAULT_INPUT_DIR, force: bool = False, poppler_path: Optional[str] = POPPLER_PATH) -> None:
    files = list_input_files(input_dir)
    if not files:
        logger.warning("Tidak ada file PDF/gambar di folder %s", input_dir.resolve())
        return

    logger.info("Menemukan %d file di %s", len(files), input_dir.resolve())
    processed = skipped = failed = 0

    for f in files:
        out_txt = OCR_CACHE_DIR / (f.stem + ".txt")
        if out_txt.exists() and not force:
            logger.info("Skip (cache ada): %s", f.name)
            skipped += 1
            continue

        logger.info("Memproses: %s", f.name)
        ok, msg = ocr_single_pdf(f, out_txt, poppler_path=poppler_path, langs=OCR_LANGS)
        if ok:
            logger.info("Hasil disimpan -> %s (%s)", out_txt, msg)
            processed += 1
        else:
            logger.error("Gagal memproses %s: %s", f.name, msg)
            failed += 1

    logger.info("Ringkasan OCR - Processed: %d, Skipped: %d, Failed: %d", processed, skipped, failed)

# CLI
def _parse_args():
    p = argparse.ArgumentParser(description="OCR processor (PDF + images) dengan cache ke ./ocr_cache")
    p.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT_DIR), help="Folder input (default: ./dokumen_kampus)")
    p.add_argument("--force", "-f", action="store_true", help="Force re-OCR walau cache ada")
    p.add_argument("--poppler", type=str, default=None, help="Path ke poppler bin (opsional)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    input_dir = Path(args.input)
    poppler = args.poppler if args.poppler else POPPLER_PATH
    logger.info("Starting OCR processor")
    logger.info(" - input dir : %s", input_dir.resolve())
    logger.info(" - cache dir : %s", OCR_CACHE_DIR.resolve())
    logger.info(" - poppler   : %s", poppler)
    process_all(input_dir=input_dir, force=args.force, poppler_path=poppler)
