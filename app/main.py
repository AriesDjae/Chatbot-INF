# app/main.py
import sys, os, re, logging, requests
from pathlib import Path
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma  # tidak perlu impor langsung
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from app.data_pipeline import full_pipeline
from preprocess.embedder import DB_DIR, EMBED_MODEL, create_vectordb, get_embeddings
from app.utils.logger import log_chat
from app.utils.reranker import rerank_results

# ------------------------------
LLM_MODEL = "mistral"
st.set_page_config(page_title="🎓 Chatbot Asisten Kampus", layout="wide")
st.title("🎓 Asisten Kampus Informatika UII")

def check_ollama():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Ollama tidak aktif di localhost:11434 — {e}")
        return False

if not check_ollama():
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
    if not results:
        st.info("Tidak ada hasil dari similarity_search.")
        return []

    rescored = rerank_results(question, results, top_n=5)

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
    llm = OllamaLLM(model=LLM_MODEL, base_url="http://127.0.0.1:11434", temperature=0.5)
    combined = "\n\n".join([re.sub(r"\s+", " ", d.page_content.strip()) for d in context_docs])
    combined = combined[:22000]
    prompt = ChatPromptTemplate.from_template("""
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
""")
    chain = prompt | llm
    try:
        answer = chain.invoke({"context": combined, "question": question})
        return format_llm_output(answer)
    except Exception as e:
        logging.error("LLM error: %s", e)
        return "⚠️ Gagal memproses jawaban dari model Ollama."

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
