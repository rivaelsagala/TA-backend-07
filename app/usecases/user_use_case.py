from typing import Tuple, Dict, Any
from app.services import user_service

def get_user(user_id: int) -> Tuple[Dict[str, Any], int]:
    user = user_service.get_user_by_id(user_id)
    if user:
        return {"status": "success", "data": user}, 200
    
    return {"status": "error", "message": "User tidak ditemukan"}, 404