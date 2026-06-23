from loguru import logger
from app.services.preprocessing_service import extract_and_chunk_pdf
from app.services.embedding_service import (
    store_chunks_to_supabase,
    check_document_exists,
    delete_document_chunks
)

def ingest_pdf_to_vector_db(file_path: str, original_filename: str, save_to_db: bool = True):
    try:
        # Lempar parameter ke fungsi ekstraksi
        chunks = extract_and_chunk_pdf(file_path, save_to_db)
        
        for chunk in chunks:
            chunk.metadata['source_file'] = original_filename
            if 'document_id' not in chunk.metadata:
                chunk.metadata['document_id'] = f"file_{original_filename.replace('.pdf', '')}"
            if 'village_name' not in chunk.metadata:
                chunk.metadata['village_name'] = 'unknown'
            if 'perdes_number' not in chunk.metadata:
                chunk.metadata['perdes_number'] = 'unknown'
            if 'perdes_year' not in chunk.metadata:
                chunk.metadata['perdes_year'] = 'unknown'
        
        document_id = chunks[0].metadata.get('document_id', 'unknown') if chunks else 'unknown'
        existing_count = check_document_exists(document_id)
        
        # Eksekusi penyimpanan hanya jika save_to_db == True
        if save_to_db:
            if existing_count > 0:
                logger.info(f"Dokumen '{document_id}' sudah ada ({existing_count} chunks). Menghapus versi lama...")
                delete_document_chunks(document_id)
            
            store_chunks_to_supabase(chunks)
            message = f"Berhasil memproses dokumen {original_filename} menjadi {len(chunks)} chunks ke database."
        else:
            message = f"PREVIEW MODE: Dokumen {original_filename} diekstrak menjadi {len(chunks)} chunks (TIDAK disimpan ke DB)."

        return {
            "status": "success", 
            "message": message,
            "metadata": {
                "total_chunks": len(chunks),
                "document_id": document_id,
                "saved_to_db": save_to_db,
                "replaced_existing": existing_count > 0 if save_to_db else False
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Gagal memproses dokumen: {str(e)}"
        }