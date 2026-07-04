from flask import request, jsonify
from app.services.rag_service import test_retrieval
from loguru import logger


def handle_test_retrieval():
    
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "error": "Format request salah. Butuh field 'question'."
        }), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({
            "error": "Field 'question' tidak boleh kosong."
        }), 400
    
    top_k = data.get('top_k', 5)
    initial_k = data.get('initial_k', 20)
    
    try:
        logger.info(f"[API] Test Retrieval — question: \"{question[:80]}...\", top_k={top_k}, initial_k={initial_k}")
        
        result = test_retrieval(
            query=question,
            top_k=top_k,
            initial_k=initial_k
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"[API] Test Retrieval error: {str(e)}")
        return jsonify({
            "error": f"Terjadi kesalahan saat melakukan retrieval: {str(e)}"
        }), 500
