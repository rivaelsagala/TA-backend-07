from flask import request, jsonify
from app.usecases.chat_use_case import (
    chat_with_history,
    create_new_session,
    get_user_sessions,
    get_session_history,
update_session_details,
    delete_session
)

def handle_chat():
    data = request.get_json()
    if not data or 'message' not in data or 'session_id' not in data or 'user_id' not in data:
        return jsonify({"error": "Format request salah. Butuh message, session_id, dan user_id"}), 400
    
    model_id = data.get('model_id', 1)
    ground_truth = data.get('ground_truth', None)
    evaluate = data.get('evaluate', False)
    
    if 'evaluate' in data:
        from app.services.chat_service import update_session
        update_session(data['session_id'], evaluate=evaluate)
    
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
    evaluate = data.get('evaluate', False) 
    
    response_data, status_code = create_new_session(data['user_id'], session_name, evaluate)
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
    if not data:
        return jsonify({"error": "Body request kosong"}), 400
        
    response_data, status_code = update_session_details(session_id, data)
    return jsonify(response_data), status_code

def handle_delete_session(session_id):
    response_data, status_code = delete_session(session_id)
    return jsonify(response_data), status_code