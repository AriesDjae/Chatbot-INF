from typing import List
from langchain_core.documents import Document

def ask_llm(query: str, documents: List[Document]) -> str:
    """
    Generate jawaban menggunakan dokumen yang diberikan.
    
    Args:
        query: Pertanyaan user
        documents: List dokumen yang relevan
        
    Returns:
        Jawaban yang dihasilkan
    """
    if not documents:
        return "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."
    
    # Gabungkan konten dokumen
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Generate jawaban sederhana berdasarkan konteks
    answer = f"""Berdasarkan informasi yang tersedia:

{context[:1000]}{'...' if len(context) > 1000 else ''}

Jawaban untuk pertanyaan "{query}" dapat ditemukan dalam dokumen di atas. Silakan merujuk pada sumber dokumen yang ditampilkan untuk informasi lebih detail."""
    
    return answer
