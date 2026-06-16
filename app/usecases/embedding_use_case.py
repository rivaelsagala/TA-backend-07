from loguru import logger
from app.services.preprocessing_service import extract_and_chunk_pdf
from app.services.embedding_service import (
    store_chunks_to_supabase,
    check_document_exists,
    delete_document_chunks
)

def ingest_pdf_to_vector_db(file_path: str, original_filename: str):
    """
    Pipeline lengkap untuk memproses PDF peraturan desa:
    1. Ekstrak teks + chunking dengan metadata enrichment
    2. Pastikan setiap chunk memiliki metadata lengkap untuk disambiguasi
    3. CEK DUPLIKASI: hapus chunks lama jika dokumen sama sudah ada
    4. Simpan chunks baru ke Supabase Vector DB
    
    Duplicate Handling:
    - Jika dokumen dengan document_id yang sama sudah ada di database,
      chunks lama akan DIHAPUS dulu sebelum chunks baru disimpan.
    - Ini mencegah duplikasi embedding dan memastikan selalu menggunakan versi terbaru.
    """
    try:
        # Langkah 1: Ekstrak dan Chunking PDF (sudah termasuk metadata enrichment)
        chunks = extract_and_chunk_pdf(file_path)
        
        # Langkah 2: Pastikan setiap chunk memiliki metadata lengkap
        # Ini adalah safety net jika ada metadata yang belum terisi
        for chunk in chunks:
            chunk.metadata['source_file'] = original_filename
            
            # Pastikan field metadata penting ada
            # (extract_text_from_pdf seharusnya sudah mengisi ini)
            if 'document_id' not in chunk.metadata:
                chunk.metadata['document_id'] = f"file_{original_filename.replace('.pdf', '')}"
            if 'village_name' not in chunk.metadata:
                chunk.metadata['village_name'] = 'unknown'
            if 'perdes_number' not in chunk.metadata:
                chunk.metadata['perdes_number'] = 'unknown'
            if 'perdes_year' not in chunk.metadata:
                chunk.metadata['perdes_year'] = 'unknown'
        
        # Langkah 3: CEK DUPLIKASI — hapus chunks lama jika dokumen sudah ada
        document_id = chunks[0].metadata.get('document_id', 'unknown') if chunks else 'unknown'
        existing_count = check_document_exists(document_id)
        
        if existing_count > 0:
            logger.info(f"Dokumen '{document_id}' sudah ada ({existing_count} chunks). Menghapus versi lama...")
            deleted = delete_document_chunks(document_id)
            logger.info(f"Berhasil menghapus {deleted} chunks lama untuk '{document_id}'")
        else:
            logger.info(f"Dokumen '{document_id}' belum ada di database. Menyimpan sebagai dokumen baru.")
        
        # Langkah 4: Lakukan Embedding dan Simpan ke database
        store_chunks_to_supabase(chunks)
        
        return {
            "status": "success", 
            "message": f"Berhasil memproses dokumen {original_filename} menjadi {len(chunks)} chunks ke database.",
            "metadata": {
                "total_chunks": len(chunks),
                "document_id": document_id,
                "village_name": chunks[0].metadata.get('village_name', 'unknown') if chunks else 'unknown',
                "perdes_number": chunks[0].metadata.get('perdes_number', 'unknown') if chunks else 'unknown',
                "perdes_year": chunks[0].metadata.get('perdes_year', 'unknown') if chunks else 'unknown',
                "replaced_existing": existing_count > 0,
                "previous_chunks_deleted": existing_count
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Gagal memproses dokumen: {str(e)}"
        }