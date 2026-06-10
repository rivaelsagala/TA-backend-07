import os
import requests
from typing import Optional, Dict, Any, List
from loguru import logger
from supabase import create_client, Client
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from app.services.reranker_service import rerank_documents

# Load environment variables dari file .env
load_dotenv()

# ==========================================
# 1. INISIALISASI DATABASE & EMBEDDINGS
# ==========================================
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "")
)

# ==========================================
# 2. DAFTAR MODEL YANG TERSEDIA
# ==========================================
AVAILABLE_MODELS = {
    1: {"name": "meta-llama/Llama-3.1-8B-Instruct", "type": "original"},
    2: {"name": "Qwen/Qwen2.5-7B-Instruct", "type": "original"},
    3: {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "type": "original"},
    4: {"name": "model_merged_legal", "type": "fine-tuned"},
    5: {"name": "openai/gpt-4o-mini", "type": "openai"},
    6: {"name": "openai/gpt-3.5-turbo", "type": "openai"},
    7: {"name": "maia/gemini-2.0-flash", "type": "google"},
    8: {"name": "rivaelsagala/TA-llama-3-1-8-b-finetune", "type": "rivael"}

}

# ==========================================
# 3. HUGGINGFACE SERVICE (LLM Multi-Model Support)
# ==========================================
class HuggingFaceService:
    """Service untuk berinteraksi dengan HuggingFace Router API dan Fine-tuned Model"""
    
    def __init__(self):
        # HuggingFace Router API (Model belum fine-tuned)
        # Menggunakan HF_BASE_URL sesuai dengan .env Anda
        self.api_url = os.getenv("HF_BASE_URL", "")
        self.token = os.getenv("HF_TOKEN", "")
        
        # Fine-tuned Model API (B200 Server)
        self.finetuned_api_url = os.getenv("FINETUNED_API_URL", "")
        
        # Common settings
        self.temperature = 0.0
        self.max_tokens = 2000
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def load_finetuned_model(self) -> bool:
        """Load model fine-tuned ke memori server B200"""
        try:
            load_url = f"{self.finetuned_api_url}/load-model"
            logger.info(f"Loading fine-tuned model via: {load_url}")
            
            response = requests.post(load_url, timeout=900)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                logger.info(f"✅ Fine-tuned model loaded: {result.get('message')}")
                return True
            else:
                logger.warning(f"⚠️ Model load response: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading fine-tuned model: {str(e)}")
            return False
    
    def query(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Kirim query ke HuggingFace Router API, Fine-tuned Model API, atau Maia Router
        
        Args:
            messages: List of message dictionaries with role and content
            model_id: ID model yang akan digunakan (1-5)
                1: meta-llama/Llama-3.1-8B-Instruct (HuggingFace)
                2: Qwen/Qwen2.5-7B-Instruct (HuggingFace)
                3: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (HuggingFace)
                4: model_merged_legal (Fine-tuned)
                5: openai/gpt-4.1-mini (Maia Router)
            **kwargs: Additional parameters like temperature, max_tokens
        """
        try:
            # Dapatkan info model, default ke model id 1 (Llama 3.1) jika id tidak ditemukan
            model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
            model_type = model_info.get("type", "original")
            
            if model_type == "fine-tuned":
                # ============================================
                # FINE-TUNED MODEL API (B200 Server)
                # ============================================
                api_url = f"{self.finetuned_api_url}/chat"
                logger.debug(f"FINE-TUNED model: {model_info['name']}")
                
                # Extract system_prompt DAN user message dari messages array
                # PENTING: system_prompt berisi konteks dokumen RAG yang wajib dikirim
                # agar fine-tuned model menjawab berdasarkan konteks, bukan memori internal
                system_content = ""
                user_message = ""
                for msg in messages:
                    if msg.get("role") == "system":
                        system_content = msg.get("content", "")
                    elif msg.get("role") == "user":
                        user_message = msg.get("content", "")
                
                if not system_content:
                    logger.warning("system_prompt kosong untuk fine-tuned model! Konteks RAG tidak akan digunakan.")
                
                # Format payload untuk B200 API:
                # - system_prompt: berisi konteks dokumen RAG (WAJIB agar evaluasi RAGAS valid)
                # - message: pertanyaan user
                payload = {
                    "system_prompt": system_content,
                    "message": user_message
                }
                
                logger.debug(f"Sending request to B200: {api_url}")
                logger.debug(f"Fine-tuned payload — system_prompt length: {len(system_content)} chars, message: {user_message[:80]}...")
                
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Standardisasi format response agar konsisten
                standardized_result = {
                    "choices": [
                        {
                            "message": {
                                "content": result.get("answer", "")
                            }
                        }
                    ]
                }
                
                logger.info(f"Fine-tuned Model API response successful")
                return standardized_result

            elif model_type == "openai":
                # ============================================
                # MAIA ROUTER (OpenAI Compatible API)
                # ============================================
                # Ambil base_url dari env, hapus '/' di akhir jika ada, dan tambahkan /chat/completions
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1").rstrip('/')
                api_url = f"{base_url}/chat/completions"
                api_key = os.getenv("OPENAI_API_KEY", "")
                
                logger.debug(f"Using MAIA ROUTER model: {model_info['name']}")
                
                payload = {
                    "model": model_info["name"],
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                logger.debug(f"Sending request to Maia Router: {api_url}")
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info("Maia Router API response successful")
                return result
                
            else:
                # ============================================
                # HUGGINGFACE ROUTER API (Original Models)
                # ============================================
                api_url = self.api_url
                model_name = model_info["name"]
                logger.debug(f"Using ORIGINAL model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                logger.debug(f"Sending request to HuggingFace: {api_url}")
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Model API response successful")
                return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error detail: {error_detail}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in query(): {str(e)}")
            return None
    
    def get_completion(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Optional[str]:
        """Dapatkan completion text dari messages"""
        response = self.query(messages, model_id=model_id, **kwargs)
        
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        logger.warning("No valid response content from LLM API")
        return None
    
    def chat_with_context(
        self,
        user_question: str,
        context: str,
        system_prompt: Optional[str] = None,
        model_id: int = 1,
        **kwargs
    ) -> Optional[str]:
        """
        Chat dengan context (untuk RAG)
        
        Args:
            user_question: Pertanyaan user
            context: Context dari dokumen
            system_prompt: Custom system prompt (optional)
            model_id: ID model yang akan digunakan (1-4)
        """
        if not system_prompt:
            system_prompt = (f"""
                    Anda adalah asisten hukum pemerintahan desa.

                    Jawab pertanyaan pengguna hanya berdasarkan konteks dokumen yang diberikan.

                    Aturan:
                    1. Gunakan hanya informasi dalam konteks dokumen.
                    2. Jangan menggunakan informasi di luar konteks dokumen.
                    3. Jangan menambahkan asumsi, opini pribadi, atau informasi yang tidak ada dalam dokumen.
                    4. Sertakan pasal, ayat, atau bagian atau metadata yang ada dalam dokumen jika tersedia dalam konteks.
                    5. Gunakan bahasa yang mudah di pahami oleh manusia
                    6. Jika informasi tidak ditemukan dalam dokumen, jawab: “Informasi tidak ditemukan dalam dokumen.”

                    KONTEKS DOKUMEN:
                    {context}
                    """
            )
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
        logger.info(f"Calling Model API ({model_info['name']}) with question: {user_question[:50]}...")
        return self.get_completion(messages, model_id=model_id, **kwargs)

# Singleton instance
hf_service = HuggingFaceService()

def get_answer_from_rag(query: str, model_id: int = 1) -> dict:
    """
    Mengeksekusi full pipeline RAG: Retrieve context dari Supabase
    lalu generate jawaban menggunakan model yang dipilih.
    
    Args:
        query: Pertanyaan user
        model_id: ID model yang akan digunakan (1-4)
            1: meta-llama/Llama-3.1-8B-Instruct
            2: Qwen/Qwen2.5-7B-Instruct
            3: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
            4: model_merged_legal (fine-tuned)
    """
    # Mengambil pengaturan dari environment variable
    supabase_table = os.getenv("SUPABASE_TABLE_NAME", "")
    # 1. Setup Vector Store sebagai Retriever
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=supabase_table,
        query_name="match_documents"
    )
    
    # ==========================================
    # TAHAP 1: INITIAL RETRIEVAL (K=20)
    # ==========================================
    # Ambil lebih banyak dokumen (misal 20) untuk menjaring semua kemungkinan
    initial_k = 20
    logger.info(f"Tahap 1: Mengambil top-{initial_k} dokumen awal dari Supabase...")
    initial_docs = vector_store.similarity_search(query, k=initial_k)
    
    # ==========================================
    # TAHAP 2: RE-RANKING (K=5)
    # ==========================================
    final_k = 5
    logger.info("Tahap 2: Menerapkan metode Re-ranking menggunakan MS Marco Cross-Encoder...")
    # Masukkan 20 dokumen tadi ke fungsi rerank_documents
    reranked_docs = rerank_documents(query=query, documents=initial_docs, top_k=final_k)
    
    # 3. Ekstrak konteks dan format sumber dokumen dari hasil re-ranking
    context_texts = []
    sources = []
    for doc in reranked_docs:
        context_texts.append(doc.page_content)
        sources.append({"content": doc.page_content, "metadata": doc.metadata})
        
    # Gabungkan semua teks konteks dengan pemisah yang jelas
    context_joined = "\n\n---\n\n".join(context_texts)

    # 4. Dapatkan jawaban dari LLM berdasarkan model_id
    model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
    logger.info(f"Using Model {model_info['name']} for Generation")
    
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        model_id=model_id
    )
    
    # Fallback jika API gagal atau mengembalikan None
    final_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."

    return {
        "answer": final_answer,
        "sources": sources,
        "model_used": model_info["name"]
    }