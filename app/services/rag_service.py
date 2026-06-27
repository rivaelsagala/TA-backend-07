import os
import requests
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
    base_url=os.getenv("OPENAI_BASE_URL", "")
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
            
            # Ambil metadata RAFT jika ada
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
        """
        Chat dengan context (untuk RAG)
        
        Args:
            user_question: Pertanyaan user
            context: Context dari dokumen (joined text, untuk model non-RAFT)
            system_prompt: Custom system prompt (optional)
            model_id: ID model yang akan digunakan (1-9)
            chat_history: Riwayat percakapan sebelumnya (list of {role, content}).
                          Digunakan agar LLM memahami konteks follow-up questions.
            raw_doc_chunks: List of individual document chunk strings (untuk RAFT model).
                           RAFT menerima dokumen terpisah, bukan joined context.
        """
        # Cek apakah ini RAFT model — RAFT tidak butuh system_prompt,
        # dia menerima dokumen mentah dan melakukan reasoning internal
        model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
        is_raft = model_info.get("type") == "raft"
        
        if is_raft:
            # RAFT: kirim pertanyaan + dokumen mentah, tanpa system_prompt
            messages = [{"role": "user", "content": user_question}]
            
            logger.info(f"Calling RAFT Model ({model_info['name']}) with question: {user_question[:50]}... (docs: {len(raw_doc_chunks or [])})")
            # Gunakan get_completion_with_metadata agar analisis RAFT tidak hilang
            raft_result = self.get_completion_with_metadata(messages, model_id=model_id, raw_doc_chunks=raw_doc_chunks or [], **kwargs)
            # Simpan metadata ke mutable dict agar bisa diakses caller (get_answer_from_rag)
            raft_meta = raft_result.get("raft_metadata")
            if raft_meta and "_raft_metadata_out" in kwargs and isinstance(kwargs["_raft_metadata_out"], dict):
                kwargs["_raft_metadata_out"].update(raft_meta)
            return raft_result.get("content")
        
        # Non-RAFT: bangun system_prompt dengan konteks dokumen
        if not system_prompt:
            system_prompt = (f"""
                    Anda adalah asisten hukum pemerintahan desa.
        
                    Jawab pertanyaan pengguna hanya berdasarkan konteks dokumen yang diberikan.
        
                    Aturan:
                    1. Gunakan hanya informasi dalam konteks dokumen.
                    2. Jangan menggunakan informasi di luar konteks dokumen.
                    3. Jangan menambahkan asumsi, opini pribadi, atau informasi yang tidak ada dalam dokumen.
                    4. Sertakan pasal, ayat, atau bagian yang ada dalam dokumen jika tersedia dalam konteks.
                    5. PENTING: Selalu sebutkan sumber dokumen secara spesifik — termasuk nama desa, nomor peraturan, dan tahun — saat mengutip. Contoh: "Berdasarkan Peraturan Desa Biru No. 07 Tahun 2015, Pasal 14..."
                    6. Jika ada beberapa peraturan dari desa berbeda dengan isi serupa, pastikan Anda merujuk ke peraturan yang tepat sesuai konteks dokumen yang diberikan.
                    7. Gunakan bahasa yang mudah dipahami oleh manusia.
        
                    KONTEKS DOKUMEN:
                    {context}
                    """
            )
        
        # Build messages: system → history → user
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Insert conversation history (agar LLM paham konteks follow-up)
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Current user question (always last)
        messages.append({"role": "user", "content": user_question})
        
        logger.info(f"Calling Model API ({model_info['name']}) with question: {user_question[:50]}... (history: {len(chat_history or [])} messages)")
        return self.get_completion(messages, model_id=model_id, **kwargs)

# Singleton instance
hf_service = HuggingFaceService()


# ==========================================
# QUERY REWRITING (Follow-up Resolution)
# ==========================================
def rewrite_query_with_history(
    original_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_history_turns: int = 5
) -> str:
    """
    Mengubah follow-up question yang ambigu menjadi standalone query
    dengan menggunakan konteks dari riwayat percakapan.
    
    Masalah yang diatasi:
    - User tanya: "dimana peraturan tentang ibu dan bayi" → retrieval berhasil ✅
    - User follow-up: "apa isi pasal satu itu?" → retrieval GAGAL ❌
      karena query ambigu (pasal satu dari dokumen mana?)
    - Setelah rewriting: "apa isi pasal 1 dalam peraturan desa tentang
      kesehatan ibu dan bayi" → retrieval berhasil ✅
    
    Strategi:
    - Jika TIDAK ada chat history → return query asli (tidak perlu rewrite)
    - Jika ADA chat history → gunakan LLM cepat (gpt-3.5-turbo via Maia Router)
      untuk menghasilkan standalone query yang menggabungkan konteks percakapan.
    - Rewritten query dipakai untuk RETRIEVAL (vector search + reranking)
    - Original query tetap dipakai untuk GENERATION (agar LLM menjawab apa yang user tanyakan)

    
    Args:
        original_query: Pertanyaan user saat ini (apa adanya)
        chat_history: Riwayat percakapan (list of {role, content})
        max_history_turns: Maksimal jumlah pasangan percakapan yang dipakai (default 3)
    
    Returns:
        Rewritten query (standalone) jika berhasil, atau original_query jika gagal/tidak perlu.
    """
    # Jika tidak ada history, tidak perlu rewrite
    if not chat_history:
        return original_query
    
    # Ambil N turns terakhir dari history (1 turn = 1 pair user+assistant)
    # Batasi agar prompt tidak terlalu panjang
    recent_history = chat_history[-(max_history_turns * 2):]
    
    # Format history menjadi teks yang mudah dipahami LLM
    history_text = ""
    for msg in recent_history:
        role = "User" if msg.get("role") == "user" else "Asisten"
        history_text += f"{role}: {msg.get('content', '')[:300]}\n"
    
    # Prompt untuk query rewriting
    # Instruksi: gabungkan konteks percakapan ke pertanyaan user agar jadi standalone
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
        # Gunakan model gpt-3.5-turbo (model_id=6) via Maia Router
        # Alasan: cepat (<1 detik), murah, dan cukup untuk tugas sederhana seperti rewrite
        rewritten = hf_service.get_completion(
            rewrite_messages,
            model_id=6,  # openai/gpt-3.5-turbo
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
    """
    Memperluas konteks setiap chunk yang di-retrieve dengan mengambil
    chunk tetangga (chunk[i-1] dan chunk[i+1]) dari dokumen yang sama.
    
    Kenapa?
    - Chunk butir-level bersifat atomic tapi sering kehilangan konteks
      sekitarnya (definisi, pengecualian, syarat).
    - Contoh: chunk "Pasal 14 butir 3" mungkin merujuk ke definisi di
      butir 1-2 yang tidak di-retrieve. Adjacent chunk mengisi celah ini.

    
    Args:
        reranked_docs: List of Document objects hasil re-ranking
        window: Jumlah chunk tetangga di setiap sisi (default 1 = sebelum + sesudah)
    
    Returns:
        Dict mapping (document_id, chunk_index) → list of adjacent content strings.
        Format: {
            ("perdes_biru_07_2015", 5): {
                "before": ["content chunk 4"],  # chunk sebelum (i-1)
                "after": ["content chunk 6"]    # chunk sesudah (i+1)
            }
        }
    """
    table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
    adjacent_map = {}  # key: (doc_id, chunk_idx) → {"before": [], "after": []}
    
    # STEP 1: Kumpulkan semua (document_id, chunk_index) yang perlu di-fetch
    # Gunakan set agar tidak fetch duplikat (misal 2 retrieved chunk bertetangga)
    needed_keys = set()  # set of (document_id, chunk_index)
    doc_chunk_pairs = []  # list of (document_id, chunk_index) untuk setiap reranked doc
    
    for doc in reranked_docs:
        doc_id = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")
        
        # Graceful degradation: skip jika metadata tidak ada (data lama tanpa chunk_index)
        if not doc_id or chunk_idx is None:
            doc_chunk_pairs.append((None, None))
            continue
        
        chunk_idx = int(chunk_idx)
        doc_chunk_pairs.append((doc_id, chunk_idx))
        
        # Chunk sebelum: i-1, i-2, ..., i-window
        for offset in range(1, window + 1):
            adj_idx = chunk_idx - offset
            if adj_idx >= 1:  # chunk_index starts at 1
                needed_keys.add((doc_id, adj_idx))
        
        # Chunk sesudah: i+1, i+2, ..., i+window
        for offset in range(1, window + 1):
            adj_idx = chunk_idx + offset
            needed_keys.add((doc_id, adj_idx))
    
    # Jika tidak ada chunk_index di metadata, return empty
    if not needed_keys:
        logger.info("Adjacent Chunk Expansion: tidak ada chunk_index di metadata, skip.")
        return {}
    
    # STEP 2: Batch query ke Supabase — ambil semua adjacent chunks sekaligus
    # Lebih efisien daripada query per-chunk (1 round-trip vs N round-trips)
    adjacent_lookup = {}  # (doc_id, chunk_idx) → content
    
    # Group by document_id untuk mengurangi jumlah query
    doc_ids = set(doc_id for doc_id, _ in needed_keys)
    
    try:
        for doc_id in doc_ids:
            # Fetch semua chunks dari dokumen ini yang memiliki chunk_index
            # di antara min_index dan max_index yang dibutuhkan
            indices_for_doc = [idx for did, idx in needed_keys if did == doc_id]
            if not indices_for_doc:
                continue
            
            min_idx = min(indices_for_doc)
            max_idx = max(indices_for_doc)
            
            # Query: ambil range chunk dari dokumen ini dalam 1 query
            result = (
                supabase
                .table(table_name)
                .select("content, metadata")
                .eq("metadata->>document_id", doc_id)
                .gte("metadata->>chunk_index", str(min_idx))
                .lte("metadata->>chunk_index", str(max_idx))
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
        
        # Chunk sebelum (urut dari terjauh ke terdekat: i-2, i-1)
        for offset in range(window, 0, -1):
            adj_key = (doc_id, chunk_idx - offset)
            if adj_key in adjacent_lookup:
                before_contents.append(adjacent_lookup[adj_key])
        
        # Chunk sesudah (urut dari terdekat ke terjauh: i+1, i+2)
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
    """
    Bangun context block yang diperluas dengan chunk tetangga.
    
    Format output:
    ┌─────────────────────────────────────────────┐
    │ [Metadata Header]                           │
    │ [Konteks Sebelumnya] ← chunk i-1 (if any)  │
    │ ── BAGIAN UTAMA ──    ← chunk i (retrieved) │
    │ [Konteks Berikutnya] ← chunk i+1 (if any)  │
    └─────────────────────────────────────────────┘
    
    Adjacent chunks diberi label jelas agar LLM tahu mana yang merupakan
    konteks pendukung vs bagian utama yang langsung relevan dengan query.
    """
    metadata = doc.metadata
    doc_id = metadata.get("document_id", "unknown")
    chunk_idx = metadata.get("chunk_index")
    
    # Metadata header (disambiguasi antar peraturan desa)
    meta_header = (
        f"[Sumber: {metadata.get('title', 'Unknown')}] "
        f"[Desa: {metadata.get('village_name', 'unknown')}] "
        f"[Kabupaten: {metadata.get('regency_name', 'unknown')}] "
        f"[Nomor: {metadata.get('perdes_number', '?')}/{metadata.get('perdes_year', '?')}] "
        f"[Halaman: {metadata.get('page', '?')}]"
    )
    
    # Cek apakah ada adjacent chunks untuk dokumen ini
    map_key = (doc_id, int(chunk_idx)) if chunk_idx is not None else None
    adj = adjacent_map.get(map_key) if map_key else None
    
    parts = [meta_header]
    
    if adj and adj["before"]:
        # Gabungkan semua chunk sebelumnya
        before_text = "\n\n".join(adj["before"])
        parts.append(f"[Konteks Sebelumnya — pasal/butir sebelumnya]\n{before_text}")
    
    # Bagian utama: chunk yang di-retrieve
    parts.append(f"[Bagian Utama]\n{doc.page_content}")
    
    if adj and adj["after"]:
        # Gabungkan semua chunk sesudahnya
        after_text = "\n\n".join(adj["after"])
        parts.append(f"[Konteks Berikutnya — pasal/butir selanjutnya]\n{after_text}")
    
    return "\n\n".join(parts)


def get_answer_from_rag(query: str, model_id: int = 1, chat_history: List[Dict[str, str]] = None) -> dict:
    """Full pipeline RAG: Query Rewriting → Retrieval → Reranking → Adjacent Expansion → Generation."""
    # Mengambil pengaturan dari environment variable
    supabase_table = os.getenv("SUPABASE_TABLE_NAME", "")
    
    # ==========================================
    # TAHAP 0: QUERY REWRITING (Follow-up Resolution)
    # ==========================================
    # Jika ada chat history, rewrite query ambigu menjadi standalone.
    # Contoh: "apa isi pasal satu itu?" → "apa isi pasal 1 dalam Perdes Kesehatan
    # Ibu Bayi Baru Lahir Desa Biru No. 07 Tahun 2015?"
    # Rewritten query dipakai untuk RETRIEVAL agar lebih akurat.
    # Original query tetap dipakai untuk GENERATION agar LLM menjawab apa yang user tanyakan.
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
    # Ambil lebih banyak dokumen (misal 20) untuk menjaring semua kemungkinan
    # PENTING: Gunakan retrieval_query (rewritten) bukan query asli
    initial_k = 20
    logger.info(f"Tahap 1: Mengambil top-{initial_k} dokumen awal dari Supabase...")
    initial_docs = vector_store.similarity_search(retrieval_query, k=initial_k)
    
    # ==========================================
    # TAHAP 2: RE-RANKING (K=5)
    # ==========================================
    final_k = 5
    logger.info("Tahap 2: Menerapkan metode Re-ranking menggunakan MS Marco Cross-Encoder...")
    # Gunakan retrieval_query (rewritten) agar cross-encoder menilai relevansi
    # berdasarkan query yang lebih spesifik (bukan query ambigu)
    reranked_docs, top_score = rerank_documents(query=retrieval_query, documents=initial_docs, top_k=final_k)
    
    # ==========================================
    # TAHAP 2.5: CONFIDENCE THRESHOLD CHECK
    # ==========================================
    # MS Marco Cross-Encoder outputs logit scores (can be negative).
    # Typical scores: relevant > 0, irrelevant < -5.
    # Jika skor tertinggi di bawah threshold, query tidak relevan dengan dokumen apapun.
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
    # TAHAP 3: ADJACENT CHUNK EXPANSION (skip untuk RAFT)
    # ==========================================
    adjacent_map = {}
    if not is_raft:
        logger.info("Tahap 3: Adjacent Chunk Expansion — mengambil chunk tetangga...")
        adjacent_map = fetch_adjacent_chunks(reranked_docs, window=1)
    
    # 4. Ekstrak konteks dan sumber dokumen
    context_texts = []
    raw_doc_chunks = []
    sources = []
    for doc in reranked_docs:
        metadata = doc.metadata
        if not is_raft:
            context_block = _build_expanded_context_block(doc, adjacent_map)
            context_texts.append(context_block)
        #     sources.append({"content": context_block, "metadata": metadata})
        # else:
        #     sources.append({"content": doc.page_content, "metadata": metadata})
        raw_doc_chunks.append(doc.page_content)
        sources.append({"content": doc.page_content, "metadata": metadata})
    
    context_joined = "\n\n---\n\n".join(context_texts) if context_texts else ""

    # 5. Generate jawaban
    logger.info(f"Using Model {model_info['name']} for Generation")
    
    raft_metadata_holder = {} if is_raft else None
    
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        model_id=model_id,
        chat_history=chat_history,
        raw_doc_chunks=raw_doc_chunks,
        _raft_metadata_out=raft_metadata_holder  # RAFT metadata akan disimpan di sini
    )
    
    # Fallback jika API gagal atau mengembalikan None
    final_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."

    # Ambil RAFT metadata (analisis dokumen) jika ada
    raft_analysis = None
    if raft_metadata_holder is not None:
        raft_analysis = raft_metadata_holder.get("analisis")

    return {
        "answer": final_answer,
        "sources": sources,
        "model_used": model_info["name"],
        "confidence_score": top_score,
        "analysis": raft_analysis
    }