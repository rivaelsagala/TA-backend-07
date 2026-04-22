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
        
    response_data, status_code = chat_with_history(
        session_id=data['session_id'],
        user_id=data['user_id'],
        user_question=data['message']
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