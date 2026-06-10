import os
from sentence_transformers import CrossEncoder
from loguru import logger

# Inisialisasi model secara global agar di-load sekali saja saat server berjalan
# Menggunakan model MS Marco yang direkomendasikan karena ringan dan akurat
MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'

try:
    logger.info(f"Memuat model Cross-Encoder untuk Re-ranking: {MODEL_NAME}...")
    # max_length=512 menyesuaikan panjang chunk dokumen peraturan desa
    cross_encoder = CrossEncoder(MODEL_NAME, max_length=512)
    logger.info("Model Cross-Encoder berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat model Cross-Encoder: {e}")
    cross_encoder = None

def rerank_documents(query: str, documents: list, top_k: int = 5) -> list:
    """
    Melakukan re-ranking pada dokumen menggunakan MS Marco Cross-Encoder.
    
    Args:
        query: Pertanyaan dari user
        documents: List objek Document (kandidat awal)
        top_k: Jumlah final dokumen yang akan diambil setelah diurutkan ulang
    """
    if not cross_encoder or not documents:
        logger.warning("Cross-Encoder tidak tersedia atau dokumen kosong. Mengembalikan dokumen asli.")
        return documents[:top_k]
    
    # 1. Siapkan data berpasangan: (Pertanyaan, Teks_Dokumen)
    pairs = [[query, doc.page_content] for doc in documents]
    
    logger.info(f"Melakukan re-ranking untuk {len(documents)} dokumen kandidat...")
    
    # 2. Prediksi skor kecocokan secara presisi
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.error(f"Error saat prediksi skor re-ranking: {e}")
        return documents[:top_k]
    
    # 3. Gabungkan dokumen dengan skornya masing-masing
    scored_docs = list(zip(documents, scores))
    
    # 4. Urutkan dokumen berdasarkan skor tertinggi (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # (Opsional) Log hasil re-ranking untuk memastikan model bekerja dengan baik
    logger.debug(f"--- Top {top_k} Hasil Re-ranking ---")
    for rank, (doc, score) in enumerate(scored_docs[:top_k]):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '?')
        logger.debug(f"Rank {rank+1} | Score: {score:.4f} | Source: {source} (Hal. {page})")
    
    # 5. Ambil hanya dokumennya (buang skornya) sebanyak top_k
    reranked_docs = [doc for doc, score in scored_docs[:top_k]]
    
    return reranked_docs
