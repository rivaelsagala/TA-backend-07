from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
from src.config import settings

# Inisialisasi client Supabase
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

# Inisialisasi model embedding
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_base_url
)

def store_chunks_to_supabase(chunks):
    # LangChain akan otomatis memanggil API Embedding dan menyimpan vektornya ke Supabase
    vector_store = SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name=settings.supabase_table_name,
        query_name="match_documents" 
    )
    return vector_store