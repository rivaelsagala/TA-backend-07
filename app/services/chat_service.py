import os
import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger
from dotenv import load_dotenv
import json

load_dotenv()

def get_db_connection():
    """Membuat koneksi ke database PostgreSQL"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", ""),
        port=int(os.getenv("DB_PORT", "")),
        database=os.getenv("DB_NAME", ""),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", "")
    )

def create_chat_session(user_id: int, session_name: str):
    """Membuat session chat baru"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (user_id, session_name, created_at, updated_at)
                    VALUES (%s, %s, NOW(), NOW()) RETURNING id
                """, (user_id, session_name))
                session_id = cur.fetchone()[0]
                conn.commit()
                return session_id
    except Exception as e:
        logger.error(f"Error create_chat_session: {e}")
        return None

def get_user_sessions(user_id: int):
    """Mengambil semua session milik user"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, session_name, created_at, updated_at 
                    FROM chat_sessions 
                    WHERE user_id = %s 
                    ORDER BY updated_at DESC
                """, (user_id,))
                
                results = cur.fetchall()
                # Format datetime agar bisa di-parse jadi JSON
                for r in results:
                    if r.get('created_at'): r['created_at'] = r['created_at'].isoformat()
                    if r.get('updated_at'): r['updated_at'] = r['updated_at'].isoformat()
                return results
    except Exception as e:
        logger.error(f"Error get_user_sessions: {e}")
        return []

def get_session_history(session_id: int, limit: int = None):
    """Mengambil history chat dalam session tertentu.
    
    Args:
        session_id: ID session yang ingin diambil history-nya
        limit: Jumlah row terakhir yang diambil. None = ambil semua (untuk frontend).
               Set limit=5 untuk konteks LLM (5 rows = 10 messages: 5 user + 5 assistant).
               Menghemat DB load: tidak fetch 200 rows jika hanya butuh 5.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if limit:
                    # Subquery: ambil N rows terakhir (DESC), lalu urutkan balik ke ASC
                    # agar hasilnya tetap kronologis (pesan lama → baru)
                    cur.execute("""
                        SELECT * FROM (
                            SELECT id, user_query, llm_response, created_at, faithfulness, answer_relevance, context_precision, context_recall, noise_sensitivity, similarity_score
                            FROM chat_history 
                            WHERE session_id = %s 
                            ORDER BY created_at DESC
                            LIMIT %s
                        ) sub
                        ORDER BY created_at ASC
                    """, (session_id, limit))
                else:
                    # Ambil SEMUA history (untuk ditampilkan di frontend)
                    cur.execute("""
                        SELECT id, user_query, llm_response, created_at, faithfulness, answer_relevance, context_precision, context_recall, noise_sensitivity, similarity_score
                        FROM chat_history 
                        WHERE session_id = %s 
                        ORDER BY created_at ASC
                    """, (session_id,))
                
                results = cur.fetchall()
                for r in results:
                    if r.get('created_at'): r['created_at'] = r['created_at'].isoformat()
                return results
    except Exception as e:
        logger.error(f"Error get_session_history: {e}")
        return []

def get_formatted_chat_messages(session_id: int, limit: int = None):
    """Mengambil history lalu memformatnya untuk konteks LLM
    
    Args:
        session_id: ID session
        limit: Jumlah row terakhir. None = semua (frontend), N = untuk LLM context.
    """
    history = get_session_history(session_id, limit=limit)
    formatted_messages = []
    for msg in history:
        formatted_messages.append({"role": "user", "content": msg["user_query"]})
        formatted_messages.append({"role": "assistant", "content": msg["llm_response"]})
    return formatted_messages

def update_session_name(session_id: int, new_name: str):
    """Update nama session"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE chat_sessions 
                    SET session_name = %s, updated_at = NOW() 
                    WHERE id = %s
                """, (new_name, session_id))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error update_session_name: {e}")
        return False

def delete_session(session_id: int):
    """Hapus session beserta history chat-nya"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error delete_session: {e}")
        return False

def save_chat_message(
    session_id: int, 
    user_id: int, 
    user_query: str, 
    llm_response: str, 
    metadata: dict, 
    evaluation: dict = None, 
    similarity_score: float = None
):
    """Simpan pesan chat baru beserta metrik evaluasi RAGAS dan Similarity Score"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ubah dictionary metadata menjadi JSON string
                metadata_json = json.dumps(metadata) if metadata else None
                
                # Ekstrak nilai evaluasi jika evaluate=True (menghindari error jika evaluate=False/None)
                faithfulness = evaluation.get("faithfulness") if evaluation else None
                answer_relevance = evaluation.get("answer_relevancy") if evaluation else None # Mapping key Ragas ke DB
                context_precision = evaluation.get("context_precision") if evaluation else None
                context_recall = evaluation.get("context_recall") if evaluation else None
                noise_sensitivity = evaluation.get("noise_sensitivity") if evaluation else None
                
                cur.execute("""
                    INSERT INTO chat_history (
                        session_id, user_id, user_query, llm_response, metadata, created_at,
                        faithfulness, answer_relevance, context_precision, context_recall, 
                        noise_sensitivity, similarity_score
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    session_id, user_id, user_query, llm_response, metadata_json,
                    faithfulness, answer_relevance, context_precision, context_recall, 
                    noise_sensitivity, similarity_score
                ))
                
                # Update waktu session agar selalu muncul paling atas saat chat aktif
                cur.execute("UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s", (session_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error save_chat_message: {e}")
        return False