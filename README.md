# 🎓 KampusAI - Chatbot Asisten Kampus

Aplikasi chatbot yang dapat menjawab pertanyaan tentang kampus berdasarkan dokumen PDF.

## 🚀 Cara Menjalankan

### 1. Persiapan

- Buat folder `dokumen_kampus` dan masukkan file PDF
- Install dependencies: `pip install -r requirements.txt`
- Pastikan Ollama berjalan dengan model `llama3.2` dan `mxbai-embed-large`

### 2. Jalankan Aplikasi

```bash
python run_app.py
```

## 📁 Struktur Proyek

```
├── app/main.py              # Aplikasi Streamlit (semua fungsi di sini)
├── dokumen_kampus/          # File PDF
├── ocr_cache/               # Cache OCR
├── chroma_db/               # Database vector
└── run_app.py               # Script utama
```

## 🔧 Fitur

- OCR otomatis dari PDF
- Embedding dengan Ollama
- Chat interface dengan Streamlit
- Retrieval berdasarkan konteks dokumen

## 📝 Catatan

- Aplikasi akan otomatis melakukan preprocessing saat pertama kali dijalankan
- Pastikan Tesseract dan Poppler sudah terinstall
