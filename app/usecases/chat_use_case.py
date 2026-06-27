from typing import Tuple, Dict, Any
from loguru import logger
import re
from app.services.rag_service import get_answer_from_rag
from app.services import chat_service
from app.services.ragas_service import ragas_service


MAX_HISTORY_MESSAGES = 6


def _get_recent_history(session_id: int, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """
    Mengambil riwayat percakapan terakhir dari session untuk konteks LLM.
    Mengembalikan list of {role, content} dalam format yang siap dikirim ke LLM.
    
    Hanya mengambil N pesan terakhir agar tidak memenuhi context window.
    Menggunakan SQL LIMIT agar tidak fetch semua rows dari database.
    """
    try:
        # limit=5 rows dari DB = 10 messages (5 user + 5 assistant)
        # Menghemat DB load: tidak fetch 200 rows jika hanya butuh 5
        max_pairs = max_messages // 2
        history = chat_service.get_formatted_chat_messages(session_id, limit=max_pairs)
        # Safety net: jika limit tidak bekerja di SQL, truncate di Python
        if len(history) > max_messages:
            history = history[-max_messages:]
        return history
    except Exception as e:
        logger.warning(f"Gagal mengambil chat history: {e}")
        return []


def is_chitchat(text: str) -> bool:
    cleaned_text = text.lower().strip().replace(".", "").replace("!", "")
    
    pattern = r'^(h+a+l+o+|h+a+i+|h+a+y+|h+e+l+o+|p+i+n+g+|t+e+s+t*|h+y+)$'
    
    if re.match(pattern, cleaned_text):
        return True
    
    greetings_phrase = {"selamat pagi", "selamat siang", "selamat malam", "assalamualaikum"}
    if cleaned_text in greetings_phrase:
        return True
        
    return False


# Keyword set untuk deteksi topik hukum/peraturan desa
_LEGAL_KEYWORDS = {
    # Struktur dokumen
    'pasal', 'ayat', 'bab', 'bagian', 'huruf', 'angka', 'butir',
    # Jenis peraturan
    'peraturan', 'perdes', 'peraturan desa', 'undang-undang', 'uu',
    'peraturan pemerintah', 'pp', 'perpres', 'perda',
    # Istilah pemerintahan
    'desa', 'kepala desa', 'bpd', 'kecamatan', 'kabupaten', 'pemerintah',
    'pemerintahan', 'sekretaris', 'bendahara', 'musyawarah',
    # Konten hukum
    'kewajiban', 'hak', 'tugas', 'fungsi', 'wewenang', 'sanksi',
    'ketentuan', 'larangan', 'kibbla', 'kesehatan', 'pendidikan',
    'anggaran', 'dana', 'apbdes', 'pengelolaan', 'pengawasan',
    # Pertanyaan umum tentang dokumen
    'isi', 'bunyi', 'menurut', 'berdasarkan', 'dalam dokumen',
    'perdes', 'mengatur', 'ditetapkan', 'berlaku',
    # Definisi
    'pengertian', 'definisi', 'yang dimaksud', 'adalah',
}

# Keyword set untuk deteksi off-topic yang jelas
_OFF_TOPIC_KEYWORDS = {
    # Kuliner
    'resep', 'masak', 'masakan', 'bumbu', 'nasi', 'goreng', 'rendang',
    # Hiburan
    'film', 'lagu', 'musik', 'artis', 'selebriti', 'sinetron',
    # Olahraga
    'sepakbola', 'bola', 'olahraga', 'pemain', 'klub',
    # Teknologi umum
    'coding', 'programming', 'python', 'javascript', 'react',
    # Cuaca
    'cuaca', 'hujan', 'panas',
    # Umum non-legal
    'celebrity', 'gosip', 'viral', 'meme', 'tiktok',
}


def is_off_topic(text: str) -> bool:
    """
    Deteksi apakah pertanyaan user di luar topik peraturan desa.
    
    Strategi:
    - Jika query mengandung LEGAL keywords → BUKAN off-topic (biarkan RAG)
    - Jika query mengandung OFF-TOPIC keywords DAN TIDAK ada legal keywords → OFF-TOPIC
    - Jika tidak ada keduanya → BUKAN off-topic (biarkan RAG + confidence threshold)
    
    Pendekatan ini LEBIH AMAN (lenient): hanya memblokir yang jelas off-topic.
    Pertanyaan ambigu tetap dilewatkan ke RAG, lalu confidence threshold
    yang akan memfilter jika memang tidak relevan.
    """
    lower_text = text.lower()
    
    # Cek apakah ada legal keywords dalam query
    has_legal = any(kw in lower_text for kw in _LEGAL_KEYWORDS)
    if has_legal:
        return False
    
    # Cek apakah ada off-topic keywords
    has_off_topic = any(kw in lower_text for kw in _OFF_TOPIC_KEYWORDS)
    if has_off_topic:
        return True
    
    # Default: bukan off-topic (biarkan confidence threshold yang memfilter)
    return False

def chat_with_history(session_id: int, user_id: int, user_question: str, model_id: int = 1, ground_truth: str = None, evaluate: bool = False) -> Tuple[Dict[str, Any], int]:
    try:
        logger.info(f"Processing chat for user {user_id}, session {session_id} with model_id {model_id}")

        if is_chitchat(user_question):
            answer = "Halo! Saya adalah asisten AI Anda. Ada yang bisa saya bantu terkait penelusuran dokumen atau peraturan hari ini?"
            rag_result = {"answer": answer, "sources": [], "model_used": "RuleBase Router"}

        elif is_off_topic(user_question):
            answer = (
                "Maaf, saya hanya dapat menjawab pertanyaan terkait peraturan desa dan dokumen hukum. "
                "Silakan ajukan pertanyaan tentang isi peraturan desa, pasal, kewajiban, hak, atau topik pemerintahan desa."
            )
            rag_result = {"answer": answer, "sources": [], "model_used": "OffTopicFilter"}

        else:
            # Ambil riwayat percakapan terakhir agar LLM paham konteks follow-up
            recent_history = _get_recent_history(session_id)
            logger.info(f"Sending {len(recent_history)} history messages to RAG for context")
            
            rag_result = get_answer_from_rag(
                user_question,
                model_id=model_id,
                chat_history=recent_history
            )
        
        if not rag_result or not rag_result.get("answer"):
            return {"status": "error", "message": "Gagal mendapatkan respons dari sistem RAG"}, 500
        
        # 4. Evaluasi RAGAS — hanya dijalankan jika evaluate=True
        evaluation_result = None
        if evaluate:
            logger.info(f"Starting RAGAS evaluation for model_id={model_id} (model={rag_result.get('model_used', 'Unknown')})...")
            logger.debug(f"RAGAS input — ground_truth provided: {ground_truth is not None}, contexts count: {len(rag_result.get('sources', []))}")
            
            contexts = ragas_service.format_contexts_from_sources(rag_result.get("sources", []))
            
            evaluation_result = ragas_service.evaluate_single_response(
                question=user_question,  
                answer=rag_result["answer"],
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            logger.info(f"RAGAS Evaluation (model_id={model_id}): {evaluation_result}")
        else:
            logger.info(f"RAGAS evaluation skipped (evaluate=False) for model_id={model_id}")
            
        similarity_score = None
        sources = rag_result.get("sources", [])
        if sources and isinstance(sources, list) and isinstance(sources[0], dict):
            # Mencari key 'score' atau 'similarity' dari metadata vector DB
            similarity_score = sources[0].get("score") or sources[0].get("similarity")
            if similarity_score is not None:
                similarity_score = float(similarity_score)

        # 6. Simpan Pertanyaan, Jawaban, dan Metrik ke dalam PostgreSQL (chat_history)
        metadata = {
            "sources": sources,
            "model_used": rag_result.get("model_used", "Unknown Model"),
            "analysis": rag_result.get("analysis")  # RAFT thought_process
        }
        
        # QA Note: evaluation_result tidak perlu lagi di-dump mentah ke metadata 
        # karena sekarang sudah masuk ke dalam schema kolom yang memiliki tipe data spesifik (FLOAT).
        
        chat_service.save_chat_message(
            session_id=session_id, 
            user_id=user_id, 
            user_query=user_question, 
            llm_response=rag_result["answer"], 
            metadata=metadata,
            evaluation=evaluation_result,         # Kirim dictionary dari RAGAS
        )
        
        # Build response
        response_body = {
            "status": "success",
            "message": "Jawaban berhasil diproses",
            "answer": rag_result["answer"],
            "sources": sources,
            "evaluation": evaluation_result,
            "model_used": rag_result.get("model_used", "Unknown Model")
        }
        
        # Tambahkan field analysis hanya jika ada (model RAFT)
        if rag_result.get("analysis"):
            response_body["analysis"] = rag_result["analysis"]
        
        return response_body, 200

    except Exception as e:
        logger.error(f"Error in chat_with_history: {str(e)}")
        return {"status": "error", "message": f"Server error: {str(e)}"}, 500
    
def create_new_session(user_id: int, session_name: str, evaluate: bool = False) -> Tuple[Dict[str, Any], int]:
    session_id = chat_service.create_chat_session(user_id, session_name, evaluate)
    if session_id:
        return {"status": "success", "data": {"id": session_id, "session_name": session_name, "evaluate": evaluate}}, 200
    return {"status": "error", "message": "Gagal membuat session"}, 500

def get_user_sessions(user_id: int) -> Tuple[Dict[str, Any], int]:
    sessions = chat_service.get_user_sessions(user_id)
    return {"status": "success", "data": sessions}, 200

def get_session_history(session_id: int) -> Tuple[Dict[str, Any], int]:
    history = chat_service.get_session_history(session_id)
    return {"status": "success", "data": history}, 200

def update_session_details(session_id: int, data: dict) -> Tuple[Dict[str, Any], int]:
    new_name = data.get('session_name')
    evaluate = data.get('evaluate')
    
    success = chat_service.update_session(session_id, new_name, evaluate)
    if success:
        return {"status": "success", "message": "Session berhasil diupdate"}, 200
    return {"status": "error", "message": "Gagal mengupdate session"}, 500

def delete_session(session_id: int) -> Tuple[Dict[str, Any], int]:
    success = chat_service.delete_session(session_id)
    if success:
        return {"status": "success", "message": "Session berhasil dihapus"}, 200
    return {"status": "error", "message": "Gagal menghapus session"}, 500