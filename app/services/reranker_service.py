import os
from sentence_transformers import CrossEncoder
from loguru import logger

MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'

try:
    logger.info(f"Memuat model Cross-Encoder untuk Re-ranking: {MODEL_NAME}...")
    cross_encoder = CrossEncoder(MODEL_NAME, max_length=512)
    logger.info("Model Cross-Encoder berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat model Cross-Encoder: {e}")
    cross_encoder = None

def rerank_documents(query: str, documents: list, top_k: int = 5) -> tuple:

    if not cross_encoder or not documents:
        logger.warning("Cross-Encoder tidak tersedia atau dokumen kosong. Mengembalikan dokumen asli.")
        return documents[:top_k], 0.0
    
    pairs = [[query, doc.page_content] for doc in documents]
    
    logger.info(f"Melakukan re-ranking untuk {len(documents)} dokumen kandidat...")
    
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.error(f"Error saat prediksi skor re-ranking: {e}")
        return documents[:top_k], 0.0
    
    scored_docs = list(zip(documents, scores))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"--- Top {top_k} Hasil Re-ranking ---")
    for rank, (doc, score) in enumerate(scored_docs[:top_k]):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '?')
        logger.debug(f"Rank {rank+1} | Score: {score:.4f} | Source: {source} (Hal. {page})")
    
    reranked_docs = [doc for doc, score in scored_docs[:top_k]]
    top_score = float(scored_docs[0][1]) if scored_docs else 0.0
    
    return reranked_docs, top_score
