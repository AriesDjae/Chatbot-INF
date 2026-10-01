# 🎓 KampusAI - Chatbot Asisten Kampus

Chatbot asisten akademik untuk Informatika UII. Jawaban disusun dari dokumen kampus (PDF/gambar hasil OCR) dan halaman web hasil scraping, dengan pendekatan RAG (Retrieval-Augmented Generation).

## ⚙️ Cara Kerja

```
dokumen_kampus/ (PDF, gambar) ──OCR──▶ ocr_cache/*.txt ──────┐
                                                             ├─▶ embedding (OpenAI) ─▶ chroma_db/
URL ──scraping──▶ hasil_scraping/ ──cleaner──▶ clean_scraping/┘

pertanyaan ─▶ similarity search (Chroma, top 10) ─▶ rerank (top 5) ─▶ Gemini ─▶ jawaban
```

- **OCR**: Tesseract (bahasa `ind` + `eng`) lewat `pdf2image` dan Poppler
- **Embedding**: OpenAI (`text-embedding-3-large` secara default)
- **Vector DB**: Chroma, disimpan di `chroma_db/`
- **Reranker**: CrossEncoder lokal `cross-encoder/ms-marco-MiniLM-L-6-v2`; jika gagal dimuat, fallback ke cosine similarity embedding OpenAI
- **LLM**: Google Gemini (`gemini-2.5-pro` secara default)
- **UI**: Streamlit

## 📋 Prasyarat

- Python 3.12
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) beserta data bahasa `ind` dan `eng`
- [Poppler](https://poppler.freedesktop.org/) (dibutuhkan `pdf2image`)
- Google Chrome (hanya jika memakai scraping mode Selenium)
- API key OpenAI dan Google Gemini

## 🚀 Instalasi

```bash
git clone https://github.com/AriesDjae/Chatbot-INF.git
cd Chatbot-INF

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Opsional, model NER spaCy untuk pembersih hasil scraping (tanpa ini cleaner tetap jalan dengan fallback):

```bash
python -m spacy download xx_ent_wiki_sm
```

## 🔐 Konfigurasi

Buat file `.env` di root proyek:

```env
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
EMBEDING_MODEL=text-embedding-3-large
GEMINI_MODEL_NAME=gemini-2.5-pro
```

| Variabel | Wajib | Keterangan |
| --- | --- | --- |
| `OPENAI_API_KEY` | Ya | Dipakai untuk embedding |
| `GOOGLE_API_KEY` | Ya | Dipakai untuk Gemini |
| `EMBEDING_MODEL` | Ya | Nama model embedding OpenAI (ejaan variabelnya memang `EMBEDING`) |
| `GEMINI_MODEL_NAME` | Tidak | Default `gemini-2.5-pro` |
| `TESSERACT_CMD` | Tidak | Path ke `tesseract.exe`. Default `D:\Tesseract\tesseract.exe` |
| `POPPLER_PATH` | Tidak | Path ke folder `bin` Poppler. Default `D:\Poppler\poppler-25.07.0\Library\bin` |
| `CHATBOT_LOG_LEVEL` | Tidak | Default `DEBUG` |

Jika Tesseract atau Poppler terpasang di lokasi lain, set `TESSERACT_CMD` dan `POPPLER_PATH` sesuai lokasi di komputer Anda.

## 📂 Menyiapkan Data

Folder data tidak ikut di repository (lihat `.gitignore`), jadi perlu disiapkan sendiri.

**1. Dokumen kampus**

Masukkan file PDF atau gambar (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`) ke folder `dokumen_kampus/`.

**2. Halaman web (opsional)**

```bash
streamlit run app/scraping.py
```

Masukkan URL, lalu hasilnya tersimpan di `hasil_scraping/`. Centang opsi Selenium untuk halaman yang dirender dengan JavaScript.

Setelah itu bersihkan hasil scraping:

```bash
python app/utils/data_cleaner.py
```

Hasil bersih tersimpan di `clean_scraping/`. Script ini tidak membaca `.env`, jadi `OPENAI_API_KEY` dan `EMBEDING_MODEL` harus sudah di-set sebagai environment variable di terminal.

## ▶️ Menjalankan Aplikasi

Semua perintah dijalankan dari root proyek.

**1. Bangun database vektor**

```bash
python app/data_pipeline.py
```

Perintah ini menjalankan OCR untuk file yang belum ada di `ocr_cache/`, lalu membangun ulang Chroma DB dari `ocr_cache/` dan `clean_scraping/`.

**2. Jalankan chatbot**

```bash
streamlit run app/main.py
```

Tombol **"Jalankan full pipeline (OCR + DB)"** di aplikasi melakukan hal yang sama, tetapi OCR diulang untuk semua file.

Tiap tahap juga bisa dijalankan terpisah:

```bash
python preprocess/ocr_processor.py           # OCR saja (tambahkan --force untuk mengulang semua)
python preprocess/embedder.py                # bangun ulang Chroma DB saja
```

## 📁 Struktur Proyek

```
├── app/
│   ├── main.py               # Aplikasi chatbot (Streamlit)
│   ├── scraping.py           # Aplikasi scraper halaman web (Streamlit)
│   ├── data_pipeline.py      # Pipeline OCR + build Chroma DB
│   └── utils/
│       ├── data_cleaner.py   # Pembersih hasil scraping
│       ├── gemini_client.py  # Klien Gemini
│       ├── reranker.py       # Reranker hasil pencarian
│       ├── logger.py         # Log chat dan error (JSONL)
│       └── llm_answer.py     # Jawaban sederhana tanpa LLM (tidak dipakai main.py)
├── preprocess/
│   ├── ocr_processor.py      # OCR PDF/gambar ke teks
│   └── embedder.py           # Chunking, embedding, dan Chroma DB
├── requirements.txt
│
│   # Dibuat saat dijalankan, tidak ikut di repository:
├── dokumen_kampus/           # PDF/gambar sumber
├── ocr_cache/                # Teks hasil OCR
├── hasil_scraping/           # Teks mentah hasil scraping
├── clean_scraping/           # Teks hasil scraping yang sudah dibersihkan
├── chroma_db/                # Database vektor
└── app/logs/                 # Log chat dan error
```

## 📝 Catatan

- Membangun database menghapus `chroma_db/` lama lalu membuat ulang seluruhnya, dan setiap build memanggil API embedding OpenAI (berbayar).
- Dokumen dipotong per 1000 karakter dengan overlap 200 sebelum di-embed.
- Saat pertama kali dijalankan, model CrossEncoder untuk reranker diunduh dari Hugging Face.
- Riwayat chat dicatat per hari di `app/logs/chat_YYYYMMDD.jsonl`, error di `app/logs/errors.jsonl`.
