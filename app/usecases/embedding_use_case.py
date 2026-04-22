from app.services.preprocessing_service import extract_and_chunk_pdf
from app.services.embedding_service import store_chunks_to_supabase

def ingest_pdf_to_vector_db(file_path: str, original_filename: str):
    try:
        # Langkah 1: Ekstrak dan Chunking PDF
        chunks = extract_and_chunk_pdf(file_path)
        
        # Opsional: Tambahkan metadata tambahan ke setiap chunk (misal: nama file PDF)
        for chunk in chunks:
            chunk.metadata['source_file'] = original_filename
            
        # Langkah 2: Lakukan Embedding dan Simpan ke database
        store_chunks_to_supabase(chunks)
        
        return {
            "status": "success", 
            "message": f"Berhasil memproses dokumen {original_filename} menjadi {len(chunks)} chunks ke database."
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Gagal memproses dokumen: {str(e)}"
        }