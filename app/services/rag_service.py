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
    """Service untuk berinteraksi dengan HuggingFace Router API dan Fine-tuned Model"""
    
    def __init__(self):
        # HuggingFace Router API (Model belum fine-tuned)
        self.api_url = settings.hf_api_url
        self.token = settings.hf_token
        self.model = settings.hf_llm_model
        
        # Fine-tuned Model API (B200 Server)
        self.finetuned_api_url = settings.finetuned_api_url
        self.finetuned_model_name = settings.finetuned_model_name
        
        # Common settings
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
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
    
    def query(self, messages: List[Dict[str, str]], use_finetuned: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Kirim query ke HuggingFace Router API atau Fine-tuned Model API
        
        Args:
            messages: List of message dictionaries
            use_finetuned: True untuk menggunakan model fine-tuned, False untuk model original
            **kwargs: Additional parameters
        """
        try:
            if use_finetuned:
                # Gunakan Fine-tuned Model API (B200 Server)
                api_url = f"{self.finetuned_api_url}/chat"
                logger.debug(f"FINE-TUNED model: {self.finetuned_model_name}")
                
                # Extract user message dari messages array
                user_message = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break
                
                # Format payload untuk B200 API: hanya butuh "message"
                payload = {
                    "message": user_message
                }
                
                logger.debug(f"Sending request to B200: {api_url}")
                logger.debug(f"Payload: {payload}")
                
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Format response dari B200 ke format standar
                # B200 response: {"answer": "...", "message": "..."}
                # Convert ke format: {"choices": [{"message": {"content": "..."}}]}
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
                
            else:
                # Gunakan HuggingFace Router API (model original)
                api_url = self.api_url
                logger.debug(f"Using ORIGINAL model: {self.model}")
                
                payload = {
                    "model": self.model,
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
    
    def get_completion(self, messages: List[Dict[str, str]], use_finetuned: bool = False, **kwargs) -> Optional[str]:
        """Dapatkan completion text dari messages"""
        response = self.query(messages, use_finetuned=use_finetuned, **kwargs)
        
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
        use_finetuned: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Chat dengan context (untuk RAG)
        
        Args:
            user_question: Pertanyaan user
            context: Konteks dokumen
            system_prompt: Custom system prompt (optional)
            use_finetuned: True untuk model fine-tuned, False untuk model original
        """
        if not system_prompt:
            system_prompt = (f"""
                    Anda adalah AI Assistant untuk sistem Retrieval-Augmented Generation (RAG) yang bertugas menjawab pertanyaan berdasarkan dokumen yang diberikan.

                    ATURAN UTAMA:
                    1. Gunakan HANYA informasi yang tersedia pada konteks/dokumen.
                    2. Jangan menambahkan informasi dari pengetahuan pribadi atau asumsi di luar dokumen.
                    3. Jika jawaban tidak ditemukan pada konteks, jawab:
                    "Informasi tidak ditemukan dalam dokumen."
                    4. Jawaban harus jelas, ringkas, relevan, dan mudah dipahami.
                    5. Prioritaskan informasi yang paling sesuai dengan pertanyaan pengguna.
                    6. Jika tersedia, sertakan pasal, poin, atau bagian dokumen yang mendukung jawaban.
                    7. Jangan membuat interpretasi hukum di luar isi dokumen.
                    8. Jangan menghasilkan jawaban yang ambigu, spekulatif, atau berhalusinasi.
                    9. Gunakan Bahasa Indonesia formal dan profesional.
                    10. Jika pertanyaan meminta daftar atau poin-poin, gunakan format bullet point.

                    FORMAT KERJA:
                    - Analisis pertanyaan pengguna.
                    - Identifikasi informasi paling relevan dari konteks.
                    - Susun jawaban berdasarkan isi dokumen.
                    - Pastikan jawaban konsisten dengan konteks.

                    CONTOH:
                    Konteks:
                    Pasal 3 menjelaskan bahwa tujuan penyelenggaraan pelayanan KIBBLA adalah meningkatkan kualitas pelayanan kesehatan ibu dan anak.

                    Pertanyaan:
                    Apa tujuan penyelenggaraan pelayanan KIBBLA?

                    Jawaban:
                    Tujuan penyelenggaraan pelayanan KIBBLA adalah meningkatkan kualitas pelayanan kesehatan ibu dan anak sebagaimana dijelaskan dalam Pasal 3.

                    CONTOH JIKA JAWABAN TIDAK ADA:
                    Pertanyaan:
                    Siapa pendiri program KIBBLA?

                    Jawaban:
                    Informasi tidak ditemukan dalam dokumen.

                    INSTRUKSI TAMBAHAN:
                    - Fokus pada akurasi jawaban.
                    - Hindari pengulangan kalimat yang tidak perlu.
                    - Jangan memberikan opini pribadi.
                    - Jangan menjawab di luar konteks dokumen yang diberikan.

                    KONTEKS DOKUMEN:
                    {context}
                    """
            )
            
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        model_type = "Fine-tuned" if use_finetuned else "Original"
        logger.info(f"Calling {model_type} Model API with question: {user_question[:50]}...")
        return self.get_completion(messages, use_finetuned=use_finetuned, **kwargs)

# Singleton instance
hf_service = HuggingFaceService()


def get_answer_from_rag(query: str, use_finetuned_model: bool = False) -> dict:
    """
    Mengeksekusi full pipeline RAG: Retrieve context dari Supabase
    lalu generate jawaban menggunakan HuggingFace Llama 3.1 atau Fine-tuned Model.
    
    Args:
        query: Pertanyaan user
        use_finetuned_model: True untuk menggunakan model fine-tuned, False untuk model original
    """
    # 1. Setup Vector Store sebagai Retriever
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=settings.supabase_table_name,
        query_name="match_documents"
        # query_name="match_new_documents"
    )
    
    # 2. Ambil dokumen relevan
    docs = vector_store.similarity_search(query, k=settings.top_k_results)
    # docs = []
    
    # 3. Ekstrak konteks dan format sumber dokumen
    context_texts = []
    sources = []
    for doc in docs:
        context_texts.append(doc.page_content)
        sources.append({"content": doc.page_content, "metadata": doc.metadata})
        
    # Gabungkan semua teks konteks dengan pemisah yang jelas
    context_joined = "\n\n---\n\n".join(context_texts)

    # 4. Dapatkan jawaban dari LLM (Original atau Fine-tuned)
    # System prompt sudah di-handle oleh chat_with_context() dengan default yang sesuai
    model_type = "Fine-tuned" if use_finetuned_model else "Original"
    logger.info(f"Using {model_type} Model for RAG")
    
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        use_finetuned=use_finetuned_model
    )
    
    # Fallback jika API gagal atau mengembalikan None
    final_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."

    return {
        "answer": final_answer,
        "sources": sources,
        "model_used": model_type
    }