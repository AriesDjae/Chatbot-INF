# app/main.py
import sys, os, re, logging
from pathlib import Path
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = load_dotenv = None

if find_dotenv and load_dotenv:
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)

from langchain_huggingface import HuggingFaceEmbeddings

from app.data_pipeline import full_pipeline
from preprocess.embedder import DB_DIR, EMBED_MODEL, create_vectordb, get_embeddings
from app.utils.logger import log_chat, log_error
from app.utils.reranker import rerank_results
from app.utils.gemini_client import generate_response, GeminiConfigurationError, ensure_gemini_ready

LOG_LEVEL = os.getenv("CHATBOT_LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------
LLM_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
st.set_page_config(page_title="🎓 Chatbot Asisten Kampus", layout="wide")
st.title("🎓 Asisten Kampus Informatika UII")

def check_gemini():
    try:
        ensure_gemini_ready(LLM_MODEL)
        return True
    except GeminiConfigurationError as cfg_err:
        st.error(str(cfg_err))
        log_error(str(cfg_err), stage="gemini_init")
    except Exception as e:
        msg = f"Gagal terhubung ke Gemini API: {e}"
        st.error(msg)
        log_error(msg, stage="gemini_init")
        logger.exception("Gemini initialization failed.")
    return False

if not check_gemini():
    st.stop()

# ------------------------------
# Inisialisasi Embedding & DB
# ------------------------------
# gunakan embeddings yang konsisten (get_embeddings dari embedder jika mau)
embeddings = get_embeddings()
vectordb = None

def init_db():
    global vectordb
    if not Path(DB_DIR).exists():
        st.warning("Database belum dibuat. Jalankan pipeline terlebih dahulu.")
        return False
    try:
        # gunakan factory dari embedder sehingga client_settings konsisten
        vectordb = create_vectordb(embedding_function=embeddings)
        logging.info("Database siap digunakan.")
        return True
    except Exception as e:
        logging.error("Gagal inisialisasi vectordb: %s", e)
        st.error("Gagal inisialisasi database embedding. Cek log.")
        return False

def expand_query(q: str):
    q = q.lower().strip()
    if len(q.split()) < 3:
        return f"panduan kampus tentang {q}"
    return q

def retrieve_docs(question: str, top_k: int = 10):
    if vectordb is None:
        st.warning("Database belum siap.")
        return []

    results = vectordb.similarity_search_with_score(question, k=top_k)
    logger.debug("Similarity search returned %d results for query '%s'", len(results), question)
    if not results:
        st.info("Tidak ada hasil dari similarity_search.")
        return []

    rescored = rerank_results(question, results, top_n=5)
    logger.debug("After rerank, using %d documents", len(rescored))

    st.markdown("#### 🔍 Dokumen paling relevan (setelah reranker):")
    for i, (doc, score) in enumerate(rescored, start=1):
        st.markdown(f"**{i}.** {doc.metadata.get('source', 'tidak diketahui')} (skor: `{score:.3f}`)")
        st.caption(doc.page_content[:250] + "...")
    return [doc for doc, _ in rescored]

def format_llm_output(text: str) -> str:
    text = re.sub(r"^(halo|hai|salaam|assalamu.*?)[.!?\n]+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", "\n\n", text)
    wrapped = "\n".join(re.findall(r".{1,100}(?:\s+|$)", text))
    return wrapped.strip()

def ask_llm(question: str, context_docs):
    combined = "\n\n".join([re.sub(r"\s+", " ", d.page_content.strip()) for d in context_docs])
    combined = combined[:22000]
    prompt_template = """
Anda adalah **Asisten Akademik Fakultas Teknologi Industri Universitas Islam Indonesia (FTI UII)**.

Gunakan informasi berikut untuk menjawab pertanyaan mahasiswa dengan sopan, profesional, dan formal akademik.

=== INFORMASI RELEVAN ===
{context}
=== SELESAI INFORMASI ===

**Pertanyaan Mahasiswa:**
{question}

**Instruksi:**
1. Jawab secara ringkas dan langsung ke inti, tanpa sapaan informal.
2. Gunakan gaya bahasa akademik dan jelas, hindari ungkapan santai seperti “halo”, “salaam”, atau “ya kak”.
3. Jika informasi dokumen tidak cukup, berikan jawaban umum yang relevan dengan konteks kampus.
4. Format jawaban dalam paragraf yang rapi (maksimal 3–5 baris per paragraf).

**Jawaban Asisten:**
"""
    try:
        rendered_prompt = prompt_template.format(context=combined, question=question)
        logger.debug("Sending prompt to Gemini (length=%d)", len(rendered_prompt))
        answer = generate_response(rendered_prompt, model_name=LLM_MODEL)
        logger.debug("Received answer length=%d", len(answer))
        return format_llm_output(answer)
    except Exception as e:
        logging.exception("LLM error: %s", e)
        log_error(str(e), stage="gemini_generate")
        return "⚠️ Gagal memproses jawaban dari model Gemini."

# ------------------------------
if st.button("🔄 Jalankan full pipeline (OCR + DB)"):
    st.info("Memproses dokumen dan membangun database...")
    full_pipeline(force_ocr=True, rebuild_db=True)
    st.session_state.initialized = init_db()

if "initialized" not in st.session_state:
    st.session_state.initialized = init_db()

if st.session_state.initialized:
    st.success("✅ Database siap digunakan.")
else:
    st.warning("⚠️ Database belum siap. Klik tombol di atas.")

# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Tulis pertanyaan Anda di sini...", key="chat_input_main")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    expanded_q = expand_query(question)
    docs = retrieve_docs(expanded_q)
    if not docs:
        answer = ask_llm(question, [])
        retrieved_sources = []
    else:
        answer = ask_llm(question, docs)
        retrieved_sources = [d.metadata.get("source", "unknown") for d in docs]
    log_chat(question, answer, retrieved_sources)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
