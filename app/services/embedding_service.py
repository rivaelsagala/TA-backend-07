import os
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