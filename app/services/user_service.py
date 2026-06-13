from psycopg2.extras import RealDictCursor
from loguru import logger
from app.services.chat_service import get_db_connection

def get_user_by_id(user_id: int):
    """Mengambil data user berdasarkan ID"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Sesuaikan query ini dengan struktur tabel users kamu
                cur.execute("""
                    SELECT id, username, created_at 
                    FROM users 
                    WHERE id = %s
                """, (user_id,))
                
                user = cur.fetchone()
                
                # Format datetime agar bisa di-parse jadi JSON
                if user and user.get('created_at'):
                    user['created_at'] = user['created_at'].isoformat()
                    
                return user
    except Exception as e:
        logger.error(f"Error get_user_by_id: {e}")
        return None