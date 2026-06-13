from typing import Tuple, Dict, Any
from loguru import logger
import re
from app.services.rag_service import get_answer_from_rag
from app.services import chat_service
from app.services.ragas_service import ragas_service


def is_chitchat(text: str) -> bool:
    cleaned_text = text.lower().strip().replace(".", "").replace("!", "")
    
    pattern = r'^(h+a+l+o+|h+a+i+|h+a+y+|h+e+l+o+|p+i+n+g+|t+e+s+t*|h+y+)$'
    
    if re.match(pattern, cleaned_text):
        return True
    
    greetings_phrase = {"selamat pagi", "selamat siang", "selamat malam", "assalamualaikum"}
    if cleaned_text in greetings_phrase:
        return True
        
    return False

def chat_with_history(session_id: int, user_id: int, user_question: str, model_id: int = 1, ground_truth: str = None) -> Tuple[Dict[str, Any], int]:
    try:
        logger.info(f"Processing chat for user {user_id}, session {session_id} with model_id {model_id}")

        if is_chitchat(user_question):
            answer = "Halo! Saya adalah asisten AI Anda. Ada yang bisa saya bantu terkait penelusuran dokumen atau peraturan hari ini?"
            rag_result = {"answer": answer, "sources": [], "model_used": "RuleBase Router"}

        else:
            rag_result = get_answer_from_rag(user_question, model_id=model_id)
        
        if not rag_result or not rag_result.get("answer"):
            return {"status": "error", "message": "Gagal mendapatkan respons dari sistem RAG"}, 500
        
        # # 4. Evaluasi respons menggunakan RAGAS
        # # Catatan untuk model_id=4 (fine-tuned):
        # # - Evaluasi TETAP dijalankan seperti model lainnya
        # # - System prompt + konteks dokumen sudah dikirim ke fine-tuned model (diperbaiki)
        # # - ground_truth digunakan jika dikirim dari client, jika tidak → fallback ke answer
        # logger.info(f"Starting RAGAS evaluation for model_id={model_id} (model={rag_result.get('model_used', 'Unknown')})...")
        # logger.debug(f"RAGAS input — ground_truth provided: {ground_truth is not None}, contexts count: {len(rag_result.get('sources', []))}")
        
        # contexts = ragas_service.format_contexts_from_sources(rag_result.get("sources", []))
        
        # evaluation_result = ragas_service.evaluate_single_response(
        #     question=user_question,  
        #     answer=rag_result["answer"],
        #     contexts=contexts,
        #     ground_truth=ground_truth  # None jika tidak dikirim dari client (akan trigger warning)
        # )
        
        # logger.info(f"RAGAS Evaluation (model_id={model_id}): {evaluation_result}")
            
        # 5. Simpan Pertanyaan dan Jawaban ke dalam PostgreSQL (chat_history)
        metadata = {
            "sources": rag_result.get("sources", []),
            # "evaluation": evaluation_result,
            "model_used": rag_result.get("model_used", "Unknown Model")
        }
        chat_service.save_chat_message(
            session_id=session_id, 
            user_id=user_id, 
            user_query=user_question, 
            llm_response=rag_result["answer"], 
            metadata=metadata
        )
        
        return {
            "status": "success",
            "message": "Jawaban berhasil diproses",
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
            # "evaluation": evaluation_result,
            "model_used": rag_result.get("model_used", "Unknown Model")
        }, 200

    except Exception as e:
        logger.error(f"Error in chat_with_history: {str(e)}")
        return {"status": "error", "message": f"Server error: {str(e)}"}, 500
    
def create_new_session(user_id: int, session_name: str) -> Tuple[Dict[str, Any], int]:
    session_id = chat_service.create_chat_session(user_id, session_name)
    if session_id:
        return {"status": "success", "data": {"id": session_id, "session_name": session_name}}, 200
    return {"status": "error", "message": "Gagal membuat session"}, 500

def get_user_sessions(user_id: int) -> Tuple[Dict[str, Any], int]:
    sessions = chat_service.get_user_sessions(user_id)
    return {"status": "success", "data": sessions}, 200

def get_session_history(session_id: int) -> Tuple[Dict[str, Any], int]:
    history = chat_service.get_session_history(session_id)
    return {"status": "success", "data": history}, 200

def update_session_name(session_id: int, new_name: str) -> Tuple[Dict[str, Any], int]:
    success = chat_service.update_session_name(session_id, new_name)
    if success:
        return {"status": "success", "message": "Nama session berhasil diupdate"}, 200
    return {"status": "error", "message": "Gagal mengupdate session"}, 500

def delete_session(session_id: int) -> Tuple[Dict[str, Any], int]:
    success = chat_service.delete_session(session_id)
    if success:
        return {"status": "success", "message": "Session berhasil dihapus"}, 200
    return {"status": "error", "message": "Gagal menghapus session"}, 500