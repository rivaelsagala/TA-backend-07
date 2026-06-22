from flask import request, jsonify
from app.usecases.chat_use_case import (
    chat_with_history,
    create_new_session,
    get_user_sessions,
    get_session_history,
    update_session_name,
    delete_session
)

def handle_chat():
    data = request.get_json()
    if not data or 'message' not in data or 'session_id' not in data or 'user_id' not in data:
        return jsonify({"error": "Format request salah. Butuh message, session_id, dan user_id"}), 400
    
    # Ambil model_id dari request. Jika kosong, default gunakan 1 (Llama-3.1-8B)
    # 1: meta-llama/Llama-3.1-8B-Instruct
    # 2: Qwen/Qwen2.5-7B-Instruct
    # 3: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # 4: model_merged_legal (fine-tuned)
    # 5: openai/gpt-4o-mini
    # 6: openai/gpt-3.5-turbo
    # 7: maia/gemini-2.0-flash
    # 8: rivaelsagala/TA-llama-3-1-8-b-finetune
    # 9: model_merged_raft_perdes (RAFT)
    model_id = data.get('model_id', 1)
    
    # Ambil ground_truth (jawaban pakar) dari request body (opsional).
    # Jika diisi, hasil evaluasi RAGAS akan lebih akurat (terutama context_recall,
    # context_entity_recall, noise_sensitivity). Jika tidak diisi, evaluasi tetap
    # berjalan tapi metrik berbasis ground_truth tidak valid.
    ground_truth = data.get('ground_truth', None)
    
    # Flag untuk mengontrol apakah evaluasi RAGAS dijalankan atau tidak.
    # Default: False (tidak dijalankan) agar response lebih cepat.
    # Set True jika ingin mengevaluasi kualitas RAG pada request tertentu.
    evaluate = data.get('evaluate', False)
    
    response_data, status_code = chat_with_history(
        session_id=data['session_id'],
        user_id=data['user_id'],
        user_question=data['message'],
        model_id=model_id,
        ground_truth=ground_truth,
        evaluate=evaluate
    )
    return jsonify(response_data), status_code

def handle_create_session():
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({"error": "Butuh field user_id"}), 400
        
    session_name = data.get('session_name', 'New Chat')
    response_data, status_code = create_new_session(data['user_id'], session_name)
    return jsonify(response_data), status_code

def handle_get_sessions():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Butuh query parameter user_id"}), 400
        
    response_data, status_code = get_user_sessions(int(user_id))
    return jsonify(response_data), status_code

def handle_get_history(session_id):
    response_data, status_code = get_session_history(session_id)
    return jsonify(response_data), status_code

def handle_update_session(session_id):
    data = request.get_json()
    if not data or 'session_name' not in data:
        return jsonify({"error": "Butuh field session_name"}), 400
        
    response_data, status_code = update_session_name(session_id, data['session_name'])
    return jsonify(response_data), status_code

def handle_delete_session(session_id):
    response_data, status_code = delete_session(session_id)
    return jsonify(response_data), status_code