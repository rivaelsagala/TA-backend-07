import os
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi client Supabase
supabase: Client = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))

# Inisialisasi model embedding
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1")
)


def check_document_exists(document_id: str) -> int:
    """
    Cek apakah chunks dengan document_id tertentu sudah ada di Supabase.
    Mengembalikan jumlah chunks yang ditemukan (0 = belum ada).
    """
    try:
        table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
        result = supabase.table(table_name).select(
            "id", count="exact"
        ).eq("metadata->>document_id", document_id).execute()
        
        return result.count or 0
    except Exception as e:
        logger.warning(f"Gagal cek duplikasi dokumen: {e}")
        return 0


def delete_document_chunks(document_id: str) -> int:
    """
    Hapus semua chunks dengan document_id tertentu dari Supabase.
    Digunakan saat re-ingest dokumen yang sama (mengganti versi lama).
    
    Returns:
        Jumlah chunk yang berhasil dihapus.
    """
    try:
        table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
        
        result = supabase.table(table_name).delete().eq(
            "metadata->>document_id", document_id
        ).execute()
        
        deleted_count = len(result.data) if result.data else 0
        logger.info(f"Deleted {deleted_count} existing chunks for document_id: {document_id}")
        return deleted_count
    except Exception as e:
        logger.error(f"Gagal menghapus chunks dokumen {document_id}: {e}")
        return 0


def store_chunks_to_supabase(chunks):
    # LangChain akan otomatis memanggil API Embedding dan menyimpan vektornya ke Supabase
    vector_store = SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name=os.getenv("SUPABASE_TABLE_NAME", "documents"),
        query_name="match_documents" 
    )
    return vector_store