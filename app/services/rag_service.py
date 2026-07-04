import os
import time
import json
import requests
import re
from typing import Optional, Dict, Any, List
from loguru import logger
from supabase import create_client, Client
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from app.services.reranker_service import rerank_documents

load_dotenv()

# 1. INISIALISASI DATABASE & EMBEDDINGS
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    default_headers={"User-Agent": "curl/7.68.0"}
)

# 2. DAFTAR MODEL YANG TERSEDIA
AVAILABLE_MODELS = {
    1: {"name": "meta-llama/Llama-3.1-8B-Instruct", "type": "original"},
    2: {"name": "Qwen/Qwen2.5-7B-Instruct", "type": "original"},
    3: {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "type": "original"},
    5: {"name": "openai/gpt-4o-mini", "type": "openai"},
    6: {"name": "openai/gpt-3.5-turbo", "type": "openai"},
    7: {"name": "maia/gemini-2.0-flash", "type": "google"},
    8: {"name": "model_merged_raft_perdes", "type": "raft"}
}

# 3. HUGGINGFACE SERVICE (LLM Multi-Model Support)
class HuggingFaceService:
    """Service untuk berinteraksi dengan HuggingFace Router API dan Fine-tuned Model"""
    
    def __init__(self):
        self.api_url = os.getenv("HF_BASE_URL", "")
        self.token = os.getenv("HF_TOKEN", "")
        
        self.finetuned_api_url = os.getenv("FINETUNED_API_URL", "")
        
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
                logger.info(f"Fine-tuned model loaded: {result.get('message')}")
                return True
            else:
                logger.warning(f"Model load response: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            return False
    
    def query(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """Kirim query ke LLM API berdasarkan model_id (lihat AVAILABLE_MODELS)."""
        try:
            model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
            model_type = model_info.get("type", "original")
            
            if model_type == "raft":
                api_url = f"{self.finetuned_api_url}/chat-rag"
                logger.debug(f"RAFT model: {model_info['name']}")
                
                user_message = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                
                raw_doc_chunks = kwargs.get("raw_doc_chunks", [])
                
                if not raw_doc_chunks:
                    logger.warning("raw_doc_chunks kosong untuk RAFT model! Model tidak akan punya konteks dokumen.")
                
                payload = {
                    "pertanyaan": user_message,
                    "dokumen": raw_doc_chunks
                }
                
                logger.debug(f"Sending RAFT request to B200: {api_url}")
                logger.debug(f"RAFT payload — pertanyaan: {user_message[:80]}..., num_dokumen: {len(raw_doc_chunks)}")
                
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Standardisasi format response agar konsisten dengan model lain
                standardized_result = {
                    "choices": [
                        {
                            "message": {
                                "content": result.get("jawaban", "")
                            }
                        }
                    ],
                    # Metadata tambahan khusus RAFT — analisis dokumen & raw response
                    "raft_metadata": {
                        "analisis": result.get("analisis", ""),
                        "raw_response": result.get("raw_response", ""),
                        "num_documents": result.get("num_documents", 0),
                        "model_type": result.get("model_type", "raft"),
                        "pertanyaan": result.get("pertanyaan", ""),
                    }
                }
                
                logger.info(f"RAFT Model API response successful (status: {result.get('status', 'unknown')})")
                return standardized_result



            elif model_type == "openai":
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
    
    def get_completion_with_metadata(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Dapatkan completion text BESERTA metadata tambahan (khusus RAFT).
        
        Returns:
            Dict dengan key:
            - "content": str — jawaban dari model
            - "raft_metadata": dict — berisi analisis, raw_response, dsb. (None jika bukan RAFT)
        """
        response = self.query(messages, model_id=model_id, **kwargs)
        
        result = {"content": None, "raft_metadata": None}
        
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                result["content"] = choice["message"]["content"]
            
            if "raft_metadata" in response:
                result["raft_metadata"] = response["raft_metadata"]
        
        if result["content"] is None:
            logger.warning("No valid response content from LLM API")
        
        return result
    
    def chat_with_context(
        self,
        user_question: str,
        context: str,
        system_prompt: Optional[str] = None,
        model_id: int = 1,
        chat_history: Optional[List[Dict[str, str]]] = None,
        raw_doc_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[str]:
        model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
        is_raft = model_info.get("type") == "raft"
        
        if is_raft:
            messages = [{"role": "user", "content": user_question}]
            
            logger.info(f"Calling RAFT Model ({model_info['name']}) with question: {user_question[:50]}... (docs: {len(raw_doc_chunks or [])})")
            raft_result = self.get_completion_with_metadata(messages, model_id=model_id, raw_doc_chunks=raw_doc_chunks or [], **kwargs)
            raft_meta = raft_result.get("raft_metadata")
            if raft_meta and "_raft_metadata_out" in kwargs and isinstance(kwargs["_raft_metadata_out"], dict):
                kwargs["_raft_metadata_out"].update(raft_meta)
            return raft_result.get("content")
        
        # system_prompt TOT dengan konteks dokumen Zero-Shot
        if not system_prompt:
            system_prompt = (f"""
                    Anda adalah asisten hukum pemerintahan desa yang teliti.

                    Sebelum menjawab, lakukan penalaran internal "Tree of Thoughts" (jangan tulis proses ini ke output):
                    - Pikirkan 2-3 kemungkinan pendekatan untuk menjawab.
                    - Evaluasi setiap pendekatan: mana yang paling didukung dokumen?
                    - Pilih atau sintesiskan pendekatan terbaik.

                    Setelah penalaran selesai, tulis HANYA jawaban akhirnya langsung. JANGAN sertakan label [EKSPLORASI], [EVALUASI], atau [JAWABAN AKHIR] dalam output.

                    Aturan menjawab:
                    1. Gunakan hanya informasi dalam konteks dokumen.
                    2. Jangan menambahkan asumsi, opini pribadi, atau informasi di luar dokumen.
                    3. Sertakan pasal dan ayat yang relevan jika tersedia.
                    4. Sebutkan sumber secara spesifik: nama desa, nomor peraturan, dan tahun. Contoh: "Berdasarkan Peraturan Desa Biru No. 07 Tahun 2015, Pasal 14..."
                    5. Jika konteks berisi beberapa bagian yang relevan dengan pertanyaan, GABUNGKAN seluruh 
                        informasi relevan tersebut menjadi satu jawaban lengkap — jangan hanya menjawab dari 
                        satu bagian saja jika bagian lain juga relevan.
                    6. Jika ditemukan dua sumber yang membahas pasal/ayat yang sama namun ISINYA BERBEDA 
                        (berpotensi konflik data), JANGAN memilih salah satu secara diam-diam. Sebutkan 
                        eksplisit bahwa ditemukan perbedaan data antar sumber, dan tampilkan kedua versinya.

                    KONTEKS DOKUMEN:
                    {context}
                    """
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({"role": "user", "content": user_question})
        
        logger.info(f"Calling Model API ({model_info['name']}) with question: {user_question[:50]}... (history: {len(chat_history or [])} messages)")
        return self.get_completion(messages, model_id=model_id, **kwargs)

hf_service = HuggingFaceService()


# ==========================================
# QUERY REWRITING (Follow-up Resolution)
# ==========================================
def rewrite_query_with_history(
    original_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_history_turns: int = 5
) -> str:
    if not chat_history:
        return original_query
    
    recent_history = chat_history[-(max_history_turns * 2):]
    
    history_text = ""
    for msg in recent_history:
        role = "User" if msg.get("role") == "user" else "Asisten"
        history_text += f"{role}: {msg.get('content', '')[:300]}\n"
    
    rewrite_messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah asisten yang mengubah pertanyaan follow-up menjadi "
                "pertanyaan mandiri (standalone) untuk pencarian dokumen hukum.\n\n"
                "ATURAN:\n"
                "1. Jika pertanyaan sudah jelas dan mandiri, kembalikan APA ADANYA.\n"
                "2. Jika pertanyaan merujuk ke konteks sebelumnya (misal: 'pasal itu', "
                "'peraturan tersebut', 'dokumen di atas'), tambahkan nama dokumen/pasal "
                "yang spesifik dari riwayat percakapan.\n"
                "3. JANGAN menjawab pertanyaan. Hanya tulis ulang pertanyaan.\n"
                "4. Output HANYA pertanyaan yang sudah ditulis ulang, tanpa penjelasan.\n"
                "5. Pertanyaan hasil rewrite harus dalam Bahasa Indonesia.\n"
                "6. JANGAN menambahkan kata-kata seperti 'berdasarkan percakapan sebelumnya' "
                "atau 'dalam konteks di atas'. Langsung tulis pertanyaan mandiri."
            )
        },
        {
            "role": "user",
            "content": (
                f"Riwayat percakapan:\n{history_text}\n"
                f"Pertanyaan terbaru user: \"{original_query}\"\n\n"
                f"Tulis ulang pertanyaan di atas agar mandiri (tanpa perlu riwayat):"
            )
        }
    ]
    
    try:
        rewritten = hf_service.get_completion(
            rewrite_messages,
            model_id=6, 
            temperature=0.0,
            max_tokens=200
        )
        
        if rewritten and rewritten.strip():
            cleaned = rewritten.strip().strip('"').strip("'")
            logger.info(
                f"Query Rewriting — Original: \"{original_query}\" → "
                f"Rewritten: \"{cleaned}\""
            )
            return cleaned
        else:
            logger.warning("Query Rewriting: LLM response empty, using original query.")
            return original_query
    
    except Exception as e:
        logger.warning(f"Query Rewriting gagal: {e}. Menggunakan original query.")
        return original_query


# ==========================================
# ADJACENT CHUNK EXPANSION
# ==========================================
def fetch_adjacent_chunks(
    reranked_docs: list,
    window: int = 1
) -> dict:
    table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
    adjacent_map = {}  
    
    # STEP 1: Kumpulkan semua (document_id, chunk_index) yang perlu di-fetch
    needed_keys = set() 
    doc_chunk_pairs = []  
    
    for doc in reranked_docs:
        doc_id = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")
        
        if not doc_id or chunk_idx is None:
            doc_chunk_pairs.append((None, None))
            continue
        
        chunk_idx = int(chunk_idx)
        doc_chunk_pairs.append((doc_id, chunk_idx))
        
        for offset in range(1, window + 1):
            adj_idx = chunk_idx - offset
            if adj_idx >= 1:  
                needed_keys.add((doc_id, adj_idx))
        
        for offset in range(1, window + 1):
            adj_idx = chunk_idx + offset
            needed_keys.add((doc_id, adj_idx))
    
    if not needed_keys:
        logger.info("Adjacent Chunk Expansion: tidak ada chunk_index di metadata, skip.")
        return {}
    
    # STEP 2: Batch query ke Supabase
    adjacent_lookup = {}  
    doc_ids = set(doc_id for doc_id, _ in needed_keys)
    
    try:
        for doc_id in doc_ids:
            indices_for_doc = [idx for did, idx in needed_keys if did == doc_id]
            if not indices_for_doc:
                continue
            
            min_idx = min(indices_for_doc)
            max_idx = max(indices_for_doc)
            
            idx_strings = [str(i) for i in range(min_idx, max_idx + 1)]
            result = (
                supabase
                .table(table_name)
                .select("content, metadata")
                .eq("metadata->>document_id", doc_id)
                .in_("metadata->>chunk_index", idx_strings)
                .execute()
            )
            
            for row in result.data:
                meta = row.get("metadata", {})
                row_doc_id = meta.get("document_id", "")
                row_chunk_idx = meta.get("chunk_index")
                if row_chunk_idx is not None:
                    key = (row_doc_id, int(row_chunk_idx))
                    adjacent_lookup[key] = row.get("content", "")
        
        logger.info(
            f"Adjacent Chunk Expansion: fetched {len(adjacent_lookup)} neighboring chunks "
            f"from {len(doc_ids)} document(s) (window={window})"
        )
    
    except Exception as e:
        logger.warning(f"Adjacent Chunk Expansion gagal: {e}. Lanjut tanpa ekspansi.")
        return {}
    
    # STEP 3: Bangun adjacent_map untuk setiap reranked chunk
    for doc_id, chunk_idx in doc_chunk_pairs:
        if doc_id is None:
            continue
        
        map_key = (doc_id, chunk_idx)
        before_contents = []
        after_contents = []
        
        for offset in range(window, 0, -1):
            adj_key = (doc_id, chunk_idx - offset)
            if adj_key in adjacent_lookup:
                before_contents.append(adjacent_lookup[adj_key])
        
        for offset in range(1, window + 1):
            adj_key = (doc_id, chunk_idx + offset)
            if adj_key in adjacent_lookup:
                after_contents.append(adjacent_lookup[adj_key])
        
        if before_contents or after_contents:
            adjacent_map[map_key] = {
                "before": before_contents,
                "after": after_contents
            }
    
    return adjacent_map


def _build_expanded_context_block(
    doc,
    adjacent_map: dict
) -> str:
    metadata = doc.metadata
    doc_id = metadata.get("document_id", "unknown")
    chunk_idx = metadata.get("chunk_index")
    
    meta_header = (
        f"[Sumber: {metadata.get('title', 'Unknown')}] "
        f"[Desa: {metadata.get('village_name', 'unknown')}] "
        f"[Kabupaten: {metadata.get('regency_name', 'unknown')}] "
        f"[Nomor: {metadata.get('perdes_number', '?')}/{metadata.get('perdes_year', '?')}] "
        f"[Halaman: {metadata.get('page', '?')}]"
    )
    
    map_key = (doc_id, int(chunk_idx)) if chunk_idx is not None else None
    adj = adjacent_map.get(map_key) if map_key else None
    
    parts = [meta_header]
    
    if adj and adj["before"]:
        before_text = "\n\n".join(adj["before"])
        parts.append(f"[Konteks Sebelumnya — pasal/butir sebelumnya]\n{before_text}")
    
    parts.append(f"[Bagian Utama]\n{doc.page_content}")
    
    if adj and adj["after"]:
        after_text = "\n\n".join(adj["after"])
        parts.append(f"[Konteks Berikutnya — pasal/butir selanjutnya]\n{after_text}")
    
    return "\n\n".join(parts)


def evaluate_rag_answer(query: str, context: str, answer: str) -> dict:
    """LLM-as-a-Judge untuk mengevaluasi kualitas jawaban RAG."""
    eval_prompt = (
        "Anda adalah juri (evaluator) untuk sistem RAG hukum desa.\n"
        "Tugas Anda adalah menilai JAWABAN berdasarkan KONTEKS dan PERTANYAAN.\n\n"
        "Berikan output HANYA dalam format JSON dengan struktur:\n"
        '{"grounded": true/false, "relevant": true/false, "score": 1-10, "reason": "alasan singkat"}\n\n'
        f"PERTANYAAN: {query}\n"
        f"KONTEKS DOKUMEN:\n{context}\n\n"
        f"JAWABAN SISTEM:\n{answer}\n"
    )
    messages = [{"role": "system", "content": eval_prompt}]
    try:
        # Memanggil API LLM secara langsung (menggunakan gpt-4o-mini)
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1").rstrip('/')
        api_url = f"{base_url}/chat/completions"
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 150
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/7.68.0"
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
    except Exception as e:
        logger.warning(f"LLM-as-a-Judge evaluation failed: {e}")
    
    return {"grounded": None, "relevant": None, "score": None, "reason": "Evaluasi gagal diproses."}

def _extract_final_answer(text: str) -> str:
    """
    Fallback extractor: jika model masih menulis [EKSPLORASI] / [EVALUASI] / [JAWABAN AKHIR],
    ambil hanya teks setelah penanda [JAWABAN AKHIR] (case-insensitive, multi-format).
    Jika tidak ada penanda, kembalikan teks asli.
    """
    import re as _re
    # Cari berbagai varian penanda jawaban akhir
    pattern = _re.compile(
        r'\[JAWABAN AKHIR\]|\*\*\[JAWABAN AKHIR\]\*\*|\*\*JAWABAN AKHIR\*\*|JAWABAN AKHIR\s*:',
        _re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        extracted = text[match.end():].strip()
        # Hilangkan baris kosong berlebih di awal
        return extracted.lstrip('\n').strip()
    return text.strip()


def get_answer_from_rag(query: str, model_id: int = 1, chat_history: List[Dict[str, str]] = None) -> dict:
    """Full pipeline RAG: Query Rewriting → Retrieval → Reranking → Adjacent Expansion → Generation."""
    supabase_table = os.getenv("SUPABASE_TABLE_NAME", "")
    
    # ==========================================
    # TAHAP 0: QUERY REWRITING (Follow-up Resolution)
    # ==========================================
    t0_retrieval = time.time()
    
    retrieval_query = rewrite_query_with_history(
        original_query=query,
        chat_history=chat_history
    )
    
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
    initial_k = 20
    logger.info(f"Tahap 1: Mengambil top-{initial_k} dokumen awal dari Supabase...")
    initial_docs = vector_store.similarity_search(retrieval_query, k=initial_k)
    
    # ==========================================
    # TAHAP 2: RE-RANKING (K=5)
    # ==========================================
    final_k = 5
    logger.info("Tahap 2: Menerapkan metode Re-ranking menggunakan MS Marco Cross-Encoder...")
    reranked_docs, top_score = rerank_documents(query=retrieval_query, documents=initial_docs, top_k=final_k)
    
    CONFIDENCE_THRESHOLD = -5.0
    logger.info(f"Top re-ranking score: {top_score:.4f} (threshold: {CONFIDENCE_THRESHOLD})")
    
    if top_score < CONFIDENCE_THRESHOLD:
        logger.info(f"Confidence too low ({top_score:.4f} < {CONFIDENCE_THRESHOLD}). Query tidak relevan dengan dokumen.")
        return {
            "answer": "Maaf, informasi yang Anda tanyakan tidak ditemukan dalam dokumen peraturan desa yang tersedia. Silakan coba pertanyaan lain terkait peraturan desa.",
            "sources": [],
            "model_used": "ConfidenceFilter",
            "confidence_score": top_score
        }
    
    # Tentukan model info untuk conditional logic
    model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
    is_raft = model_info.get("type") == "raft"
    
    # ==========================================
    # TAHAP 3: ADJACENT CHUNK EXPANSION
    # ==========================================
    # if not is_raft:
    #     logger.info("Tahap 3: Adjacent Chunk Expansion (window=3)...")
    #     adjacent_map = fetch_adjacent_chunks(reranked_docs, window=3)
    # else:
    #     logger.info("Tahap 3: Skip Adjacent Chunk Expansion for RAFT model...")
    #     adjacent_map = {}
    
    # DISABLED TEMPORARILY PER USER REQUEST
    logger.info("Tahap 3: Adjacent Chunk Expansion dinonaktifkan sementara...")
    adjacent_map = {}
        
    t1_retrieval = time.time()
    retrieval_time = t1_retrieval - t0_retrieval
    
    # 4. Ekstrak konteks dan sumber dokumen
    context_texts = []
    raw_doc_chunks = []
    sources = []
    for doc in reranked_docs:
        metadata = doc.metadata
        
        doc_id = metadata.get("document_id")
        chunk_idx = metadata.get("chunk_index")
        map_key = (doc_id, int(chunk_idx)) if doc_id and chunk_idx is not None else None
        adj = adjacent_map.get(map_key) if map_key else None
        
        if not is_raft:
            context_block = _build_expanded_context_block(doc, adjacent_map)
            context_texts.append(context_block)

            sources.append({
                "content": doc.page_content,
                "expanded_content": context_block,
                "neighbor_chunks": {
                    "before": adj["before"] if adj else [],
                    "after": adj["after"] if adj else [],
                },
                "metadata": {
                    "chunk_index": metadata.get("chunk_index"),
                    "document_id": metadata.get("document_id")
                }
            })
            raw_doc_chunks.append(doc.page_content)
        else:
            cleaned_content = re.sub(r'^(\[[^\]]+\]\s*)+\n*', '', doc.page_content, flags=re.IGNORECASE).strip()
            
            sources.append({
                "content": doc.page_content,
                "expanded_content": doc.page_content,
                "neighbor_chunks": {
                    "before": [],
                    "after": [],
                },
                "metadata": {
                    "chunk_index": metadata.get("chunk_index"),
                    "document_id": metadata.get("document_id")
                }
            })
            raw_doc_chunks.append(cleaned_content)
    
    context_joined = "\n\n---\n\n".join(context_texts) if context_texts else ""

    # 5. Generate jawaban
    logger.info(f"Using Model {model_info['name']} for Generation")
    
    raft_metadata_holder = {} if is_raft else None
    
    t0_inference = time.time()
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        model_id=model_id,
        chat_history=chat_history,
        raw_doc_chunks=raw_doc_chunks,
        _raft_metadata_out=raft_metadata_holder
    )
    t1_inference = time.time()
    inference_time = t1_inference - t0_inference
    
    raw_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."
    final_answer = _extract_final_answer(raw_answer)


    raft_analysis = None
    if raft_metadata_holder is not None:
        raft_analysis = raft_metadata_holder.get("analisis")

    # LLM-as-a-Judge
    logger.info("Mengevaluasi jawaban dengan LLM-as-a-Judge...")
    eval_context = context_joined if context_joined else "\n".join(raw_doc_chunks)
    judge_evaluation = evaluate_rag_answer(query, eval_context, final_answer)

    return {
        "answer": final_answer,
        "sources": sources,
        "model_used": model_info["name"],
        "confidence_score": top_score,
        "analysis": raft_analysis,
        "judge_evaluation": judge_evaluation,
        "retrieval_time": retrieval_time,
        "inference_time": inference_time
    }


def test_retrieval(query: str, top_k: int = 5, initial_k: int = 20) -> dict:
    supabase_table = os.getenv("SUPABASE_TABLE_NAME", "")
    
    t0 = time.time()
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=supabase_table,
        query_name="match_documents"
    )
    
    logger.info(f"[Test Retrieval] Mengambil top-{initial_k} dokumen awal untuk query: \"{query[:80]}...\"")
    initial_docs = vector_store.similarity_search(query, k=initial_k)
    
    t1_vector = time.time()
    vector_search_time = t1_vector - t0
    
    logger.info(f"[Test Retrieval] Re-ranking {len(initial_docs)} → top {top_k}...")
    reranked_docs, top_score = rerank_documents(query=query, documents=initial_docs, top_k=top_k)
    
    logger.info("[Test Retrieval] Adjacent Chunk Expansion (window=3)...")
    adjacent_map = fetch_adjacent_chunks(reranked_docs, window=3)
    
    t2_rerank = time.time()
    reranking_time = t2_rerank - t1_vector
    total_time = t2_rerank - t0
    
    retrieved_documents = []
    for rank, doc in enumerate(reranked_docs, start=1):
        metadata = doc.metadata
        
        doc_id = metadata.get("document_id")
        chunk_idx = metadata.get("chunk_index")
        map_key = (doc_id, int(chunk_idx)) if doc_id and chunk_idx is not None else None
        adj = adjacent_map.get(map_key) if map_key else None
        
        expanded_content = _build_expanded_context_block(doc, adjacent_map)

        retrieved_documents.append({
            "rank": rank,
            "content": doc.page_content,
            "metadata": {
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "?"),
                "title": metadata.get("title", "Unknown"),
                "village_name": metadata.get("village_name", ""),
                "regency_name": metadata.get("regency_name", ""),
                "perdes_number": metadata.get("perdes_number", ""),
                "perdes_year": metadata.get("perdes_year", ""),
                "document_id": metadata.get("document_id", ""),
                "chunk_index": metadata.get("chunk_index", ""),
            }
        })
    
    logger.info(
        f"[Test Retrieval] Selesai — {len(retrieved_documents)} dokumen retrieved, "
        f"top_score={top_score:.4f}, total_time={total_time:.2f}s"
    )
    
    return {
        "query": query,
        "top_score": top_score,
        "num_initial_candidates": len(initial_docs),
        "num_reranked": len(retrieved_documents),
        "retrieved_documents": retrieved_documents,
        "timing": {
            "vector_search_seconds": round(vector_search_time, 3),
            "reranking_seconds": round(reranking_time, 3),
            "total_seconds": round(total_time, 3),
        }
    }