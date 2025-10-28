import os
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def log_chat(user_input: str, response: str, retrieved_docs: list):
    """
    Simpan log chat dalam format JSONL (1 baris per event)
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"chat_{datetime.now().strftime('%Y%m%d')}.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_input": user_input,
        "response": response,
        "retrieved_docs": retrieved_docs
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOGGER ERROR] {e}")

def log_error(message: str, stage: str):
    error_file = LOG_DIR / "errors.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "error": message,
    }
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

