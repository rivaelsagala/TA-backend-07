from flask import jsonify
from app.usecases.user_use_case import get_user

def handle_get_user(id):
    response_data, status_code = get_user(id)
    return jsonify(response_data), status_code