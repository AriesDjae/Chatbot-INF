import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import re
import textwrap
import hashlib
from urllib.parse import urlparse
import time

# Tambahan untuk halaman dinamis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException


# ========== Fungsi untuk membersihkan teks ==========
def clean_text(text):
    # Hilangkan karakter non-printable dan duplikat spasi
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    return text


# ========== Fungsi untuk menyimpan hasil scrapping ==========
# ========== Fungsi untuk membuat nama file dari URL ==========
def safe_filename_from_url(url):
    parsed = urlparse(url)
    # Gabungkan domain dan path jadi nama file
    filename = parsed.netloc + parsed.path
    # Ganti karakter ilegal dengan underscore
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Hilangkan slash di akhir
    if filename.endswith("_"):
        filename = filename[:-1]
    # Tambahkan ekstensi
    filename = filename + ".txt"
    return filename


# ========== Fungsi untuk menyimpan hasil scrapping ==========
def save_result(url, text):
    if not os.path.exists("hasil_scraping"):
        os.makedirs("hasil_scraping")

    filename = safe_filename_from_url(url)
    wrapped_text = textwrap.fill(text, width=120)

    file_path = os.path.join("hasil_scraping", filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    return file_path


# ========== Fungsi Scraping Statis (Requests) ==========
def scrape_static(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8"
    }

    response = requests.get(url, headers=headers, timeout=15)

    # Cek apakah HTML
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        st.error("⚠️ Halaman ini bukan HTML atau dilindungi (binary data).")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(s.strip() for s in soup.stripped_strings)
    return clean_text(text)


# ========== Fungsi Scraping Dinamis (Selenium) ==========
def scrape_dynamic(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(3)  # Tunggu halaman selesai render

        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(s.strip() for s in soup.stripped_strings)
        return clean_text(text)
    except WebDriverException as e:
        st.error(f"❌ Gagal memuat halaman dinamis: {e}")
        return None


# ========== Antarmuka Streamlit ==========
st.set_page_config(page_title="Web Scraper Cleaner", page_icon="🧹", layout="centered")

st.title("🌐 Web Scraper & Cleaner")
st.write("Masukkan URL untuk di-scrape dan disimpan hasil teks bersihnya.")

url = st.text_input("🔗 Masukkan URL situs:", placeholder="contoh: https://informatics.uii.ac.id/")
use_selenium = st.checkbox("Gunakan Selenium untuk halaman dinamis (JavaScript heavy)")

if st.button("🚀 Jalankan Scraping"):
    if not url:
        st.warning("Masukkan URL terlebih dahulu.")
    else:
        with st.spinner("Sedang mengambil dan membersihkan data..."):
            if use_selenium:
                text = scrape_dynamic(url)
            else:
                try:
                    text = scrape_static(url)
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Gagal mengambil data dari URL: {e}")
                    text = None

            if text:
                file_path = save_result(url, text)
                st.success(f"✅ Scraping selesai! Hasil disimpan di: `{file_path}`")

                st.divider()
                st.subheader("📄 Cuplikan Hasil Scraping:")
                st.text_area("Hasil Bersih:", text[:5000], height=400)
