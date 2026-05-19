from typing import Tuple, Dict, Any
from loguru import logger
from app.services.rag_service import get_answer_from_rag
from app.services import chat_history_service
from app.services.ragas_service import ragas_service

def chat_with_history(session_id: int, user_id: int, user_question: str, use_finetuned_model: bool = False) -> Tuple[Dict[str, Any], int]:
    """
    Chat dengan history dan pilihan model
    
    Args:
        session_id: ID sesi chat
        user_id: ID user
        user_question: Pertanyaan user
        use_finetuned_model: True untuk model fine-tuned, False untuk model original
    """
    try:
        model_type = "Fine-tuned" if use_finetuned_model else "Original"
        logger.info(f"Processing chat for user {user_id}, session {session_id} with {model_type} model")
        
        # 1. Ambil riwayat chat sebelumnya agar AI mengingat percakapan sesi ini
        history_messages = chat_history_service.get_formatted_chat_messages(session_id)
        
        # 2. Jika ada history, gabungkan dengan pertanyaan saat ini
        if history_messages:
            # Format pertanyaan dengan konteks history untuk RAG
            contextual_question = f"Riwayat percakapan sebelumnya:\n"
            for msg in history_messages[-4:]:  # Ambil 4 pesan terakhir saja
                role = "User" if msg["role"] == "user" else "Assistant"
                contextual_question += f"{role}: {msg['content']}\n"
            contextual_question += f"\nPertanyaan saat ini: {user_question}"
        else:
            contextual_question = user_question
        
        # 3. Gunakan get_answer_from_rag dengan pilihan model
        rag_result = get_answer_from_rag(contextual_question, use_finetuned_model=use_finetuned_model)
        
        if not rag_result or not rag_result.get("answer"):
            return {"status": "error", "message": "Gagal mendapatkan respons dari sistem RAG"}, 500
        
        # # 4. Evaluasi respons menggunakan RAGAS
        # logger.info("Starting RAGAS evaluation...")
        # contexts = ragas_service.format_contexts_from_sources(rag_result.get("sources", []))
        
        # evaluation_result = ragas_service.evaluate_single_response(
        #     question=user_question,  
        #     answer=rag_result["answer"],
        #     contexts=contexts,
        #     reference=rag_result["answer"] 
        # )
        
        # logger.info(f"RAGAS Evaluation: {evaluation_result}")
            
        # 5. Simpan Pertanyaan dan Jawaban ke dalam PostgreSQL (chat_history)
        metadata = {
            "sources": rag_result.get("sources", []),
            # "evaluation": evaluation_result,
            "model_used": rag_result.get("model_used", model_type)
        }
        chat_history_service.save_chat_message(
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
            "model_used": rag_result.get("model_used", model_type)
        }, 200

    except Exception as e:
        logger.error(f"Error in chat_with_history: {str(e)}")
        return {"status": "error", "message": f"Server error: {str(e)}"}, 500
    
def create_new_session(user_id: int, session_name: str) -> Tuple[Dict[str, Any], int]:
    session_id = chat_history_service.create_chat_session(user_id, session_name)
    if session_id:
        return {"status": "success", "data": {"id": session_id, "session_name": session_name}}, 200
    return {"status": "error", "message": "Gagal membuat session"}, 500

def get_user_sessions(user_id: int) -> Tuple[Dict[str, Any], int]:
    sessions = chat_history_service.get_user_sessions(user_id)
    return {"status": "success", "data": sessions}, 200

def get_session_history(session_id: int) -> Tuple[Dict[str, Any], int]:
    history = chat_history_service.get_session_history(session_id)
    return {"status": "success", "data": history}, 200

def update_session_name(session_id: int, new_name: str) -> Tuple[Dict[str, Any], int]:
    success = chat_history_service.update_session_name(session_id, new_name)
    if success:
        return {"status": "success", "message": "Nama session berhasil diupdate"}, 200
    return {"status": "error", "message": "Gagal mengupdate session"}, 500

def delete_session(session_id: int) -> Tuple[Dict[str, Any], int]:
    success = chat_history_service.delete_session(session_id)
    if success:
        return {"status": "success", "message": "Session berhasil dihapus"}, 200
    return {"status": "error", "message": "Gagal menghapus session"}, 500