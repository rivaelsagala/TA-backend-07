import requests
from typing import Optional, Dict, Any, List
from loguru import logger
from supabase import create_client, Client

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from src.config import settings

# ==========================================
# 1. INISIALISASI DATABASE & EMBEDDINGS
# ==========================================
# Catatan: Kita tetap menggunakan OpenAI untuk Embeddings sesuai setup aslimu
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_base_url
)

# ==========================================
# 2. HUGGINGFACE SERVICE (LLM Llama 3.1 8B)
# ==========================================
class HuggingFaceService:
    """Service untuk berinteraksi dengan HuggingFace Router API"""
    
    def __init__(self):
        self.api_url = settings.hf_api_url
        self.token = settings.hf_token
        self.model = settings.hf_llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def query(self, messages: List[Dict[str, str]], **kwargs) -> Optional[Dict[str, Any]]:
        """Kirim query ke HuggingFace Router API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            logger.debug(f"Sending request to HuggingFace API: {self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info("✅ HuggingFace API response successful")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HuggingFace API Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in query(): {str(e)}")
            return None
    
    def get_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Dapatkan completion text dari messages"""
        response = self.query(messages, **kwargs)
        
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        logger.warning("No valid response content from HuggingFace API")
        return None
    
    def chat_with_context(
        self,
        user_question: str,
        context: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Chat dengan context (untuk RAG)"""
        if not system_prompt:
            system_prompt = (
                "Kamu adalah asisten AI yang ahli dalam menganalisis dokumen hukum dan Peraturan Desa. "
                "Gunakan HANYA informasi dari dokumen konteks di bawah ini untuk menjawab pertanyaan pengguna. "
                "Konteks ini berasal dari beberapa dokumen PDF yang berbeda. Perhatikan baik-baik bagian 'Sumber Dokumen'.\n\n"
                "Aturan menjawab:\n"
                "1. Jika jawabannya ada di dokumen yang berbeda, sebutkan perbedaannya dengan jelas.\n"
                "2. Wajib menyebutkan nama dokumen sumber di akhir jawabanmu (misal: 'Berdasarkan Peraturan Desa No...').\n"
                "3. Jika jawaban tidak ditemukan di dalam konteks, katakan dengan jujur bahwa kamu tidak menemukan jawabannya di dokumen yang ada. Jangan mengarang bebas.\n\n"
                f"Konteks Dokumen:\n{context}"
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        logger.info(f"Calling HuggingFace API with question: {user_question[:50]}...")
        return self.get_completion(messages, **kwargs)

# Singleton instance
hf_service = HuggingFaceService()

# ==========================================
# 3. RAG CORE FUNCTIONS
# ==========================================
def retrieve_documents_only(query: str) -> list:
    """
    Fungsi khusus untuk testing RAG retrieval.
    Hanya mengambil chunk yang relevan dari vector database.
    """
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=settings.supabase_table_name,
        query_name="match_documents"
    )
    
    docs = vector_store.similarity_search(query, k=settings.top_k_results)
    
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
        
    return results

def get_answer_from_rag(query: str) -> dict:
    """
    Mengeksekusi full pipeline RAG: Retrieve context dari Supabase
    lalu generate jawaban menggunakan HuggingFace Llama 3.1.
    """
    # 1. Setup Vector Store sebagai Retriever
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=settings.supabase_table_name,
        query_name="match_documents"
    )
    
    # 2. Ambil dokumen relevan
    docs = vector_store.similarity_search(query, k=settings.top_k_results)
    
    # 3. Ekstrak konteks dan format sumber dokumen
    context_texts = []
    sources = []
    for doc in docs:
        context_texts.append(doc.page_content)
        sources.append({"content": doc.page_content, "metadata": doc.metadata})
        
    # Gabungkan semua teks konteks dengan pemisah yang jelas
    context_joined = "\n\n---\n\n".join(context_texts)

    # 4. Buat System Prompt khusus dengan instruksi Peraturan Desa
    system_prompt = (
        "Kamu adalah asisten AI yang ahli dalam menganalisis dokumen hukum dan Peraturan Desa. "
        "Gunakan HANYA informasi dari dokumen konteks di bawah ini untuk menjawab pertanyaan pengguna. "
        "Konteks ini berasal dari beberapa dokumen PDF yang berbeda. Perhatikan baik-baik bagian 'Sumber Dokumen'.\n\n"
        "Aturan menjawab:\n"
        "1. Jika jawabannya ada di dokumen yang berbeda, sebutkan perbedaannya dengan jelas.\n"
        "2. Wajib menyebutkan nama dokumen sumber di akhir jawabanmu (misal: 'Berdasarkan Peraturan Desa No...').\n"
        "3. Jika jawaban tidak ditemukan di dalam konteks, katakan dengan jujur bahwa kamu tidak menemukan jawabannya di dokumen yang ada. Jangan mengarang bebas.\n\n"
        f"Konteks Dokumen:\n{context_joined}"
    )

    # 5. Dapatkan jawaban dari LLM HuggingFace
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        system_prompt=system_prompt
    )
    
    # Fallback jika API HuggingFace gagal atau mengembalikan None
    final_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."

    return {
        "answer": final_answer,
        "sources": sources
    }