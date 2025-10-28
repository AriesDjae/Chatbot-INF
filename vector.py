import os
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# === Path Konfigurasi ===
poppler_path = r"D:\Poppler\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract\tesseract.exe"

FOLDER_DOKUMEN = "./dokumen_kampus"
OCR_CACHE_DIR = "./ocr_cache"
DB_DIR = "./chroma_db"

os.makedirs(OCR_CACHE_DIR, exist_ok=True)
os.makedirs(FOLDER_DOKUMEN, exist_ok=True)

# --- Inisialisasi embedding dan splitter ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", "!", "?", " "]
)

# --- Cek apakah perlu rebuild (jika ada dokumen baru) ---
def needs_rebuild():
    """Cek apakah ada file baru yang belum di-OCR atau belum di-embed."""
    if not os.path.exists(DB_DIR):
        return True
    for filename in os.listdir(FOLDER_DOKUMEN):
        if filename.lower().endswith(".pdf"):
            cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.splitext(filename)[0]}.txt")
            if not os.path.exists(cache_file):
                return True
    return False


all_docs = []
rebuild = needs_rebuild()

if rebuild:
    print("\n🧠 Membuat ulang database dari dokumen...")
    for filename in os.listdir(FOLDER_DOKUMEN):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(FOLDER_DOKUMEN, filename)
        print(f"\n📘 Memproses file: {filename}")

        cache_file = os.path.join(OCR_CACHE_DIR, f"{os.path.splitext(filename)[0]}.txt")

        # --- Jika sudah ada hasil OCR ---
        if os.path.exists(cache_file):
            print(f"🗂️  Mengambil hasil OCR dari cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                pdf_text = f.read()
        else:
            try:
                print("🔄 Konversi PDF ke gambar...")
                images = convert_from_path(pdf_path, poppler_path=poppler_path)
                print(f"📄 Jumlah halaman: {len(images)}")

                ocr_text = ""
                for i, img in enumerate(images):
                    print(f"🧠 OCR halaman {i+1}...")
                    text = pytesseract.image_to_string(img, lang="ind")
                    if not text.strip():
                        print(f"⚠️ Halaman {i+1} kosong, fallback ke 'eng'")
                        text = pytesseract.image_to_string(img, lang="eng")
                    ocr_text += f"\n=== Halaman {i+1} ===\n{text}"

                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                pdf_text = ocr_text
                print(f"✅ OCR selesai dan disimpan di cache: {cache_file}")

            except Exception as e:
                print(f"❌ Gagal OCR {filename}: {e}")
                pdf_text = ""

        if len(pdf_text.strip()) == 0:
            print(f"🚫 Tidak ada teks dari {filename}")
            continue

        chunks = splitter.split_text(pdf_text)
        print(f"🧩 {filename}: {len(chunks)} chunk teks dihasilkan.")
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": filename}))

    if len(all_docs) > 0:
        print("💾 Menyimpan ke database Chroma...")
        vectordb = Chroma.from_documents(all_docs, embedding=embeddings, persist_directory=DB_DIR)
        vectordb.persist()
        print("✅ Embedding selesai disimpan.")
    else:
        vectordb = None
        print("⚠️ Tidak ada dokumen yang di-embedding.")
else:
    print("📚 Database sudah ada, memuat ulang Chroma...")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    print("✅ Database Chroma berhasil dimuat.")

# === Buat retriever agar bisa digunakan oleh main.py ===
if vectordb is not None:
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    print("🔍 Retriever siap digunakan.")
else:
    retriever = None
    print("⚠️ Retriever tidak dibuat karena database gagal dimuat.")
