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

def create_chat_session(user_id: int, session_name: str, evaluate: bool = False):
    """Membuat session chat baru beserta flag evaluate"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (user_id, session_name, evaluate, created_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW()) RETURNING id
                """, (user_id, session_name, evaluate))
                session_id = cur.fetchone()[0]
                conn.commit()
                return session_id
    except Exception as e:
        logger.error(f"Error create_chat_session: {e}")
        return None

def get_user_sessions(user_id: int):
    """Mengambil semua session milik user termasuk status evaluate"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Tambahkan 'evaluate' di dalam klausa SELECT
                cur.execute("""
                    SELECT id, session_name, evaluate, created_at, updated_at 
                    FROM chat_sessions 
                    WHERE user_id = %s 
                    ORDER BY updated_at DESC
                """, (user_id,))
                
                results = cur.fetchall()
                for r in results:
                    if r.get('created_at'): r['created_at'] = r['created_at'].isoformat()
                    if r.get('updated_at'): r['updated_at'] = r['updated_at'].isoformat()
                return results
    except Exception as e:
        logger.error(f"Error get_user_sessions: {e}")
        return []

def update_session(session_id: int, new_name: str = None, evaluate: bool = None):
    """Update nama session dan/atau status evaluate"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if new_name is not None and evaluate is not None:
                    cur.execute("""
                        UPDATE chat_sessions 
                        SET session_name = %s, evaluate = %s, updated_at = NOW() 
                        WHERE id = %s
                    """, (new_name, evaluate, session_id))
                elif new_name is not None:
                    cur.execute("""
                        UPDATE chat_sessions 
                        SET session_name = %s, updated_at = NOW() 
                        WHERE id = %s
                    """, (new_name, session_id))
                elif evaluate is not None:
                    cur.execute("""
                        UPDATE chat_sessions 
                        SET evaluate = %s, updated_at = NOW() 
                        WHERE id = %s
                    """, (evaluate, session_id))
                
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error update_session: {e}")
        return False

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
                    cur.execute("""
                        SELECT * FROM (
                            SELECT id, user_query, llm_response, created_at, faithfulness, answer_relevance, context_precision, context_recall, noise_sensitivity, semantic_similarity, metadata
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
                        SELECT id, user_query, llm_response, created_at, faithfulness, answer_relevance, context_precision, context_recall, noise_sensitivity, semantic_similarity, metadata
                        FROM chat_history 
                        WHERE session_id = %s 
                        ORDER BY created_at ASC
                    """, (session_id,))
                
                results = cur.fetchall()
                formatted_results = []
                for r in results:
                    if r.get('created_at'): r['created_at'] = r['created_at'].isoformat()
                    
                    if r.get('metadata'):
                        if isinstance(r['metadata'], str):
                            try:
                                metadata_dict = json.loads(r['metadata'])
                                sources = metadata_dict.get('sources', [])
                            except json.JSONDecodeError:
                                sources = []
                        elif isinstance(r['metadata'], dict):
                            sources = r['metadata'].get('sources', [])
                        else:
                            sources = []
                        
                        cleaned_sources = []
                        for s in sources:
                            if isinstance(s, dict):
                                cleaned_source = {k: v for k, v in s.items() if k != 'metadata'}
                                cleaned_sources.append(cleaned_source)
                            else:
                                cleaned_sources.append(s)
                                
                        r['sources'] = cleaned_sources
                        del r['metadata']
                    else:
                        r['sources'] = []
                    
                    ordered_row = {
                        "user_query": r.get("user_query"),
                        "llm_response": r.get("llm_response"),
                        "sources": r.get("sources", []),
                        "id": r.get("id"),
                        "created_at": r.get("created_at"),
                        "faithfulness": r.get("faithfulness"),
                        "answer_relevance": r.get("answer_relevance"),
                        "context_precision": r.get("context_precision"),
                        "context_recall": r.get("context_recall"),
                        "noise_sensitivity": r.get("noise_sensitivity"),
                        "semantic_similarity": r.get("semantic_similarity")
                    }
                    formatted_results.append(ordered_row)
                    
                return formatted_results
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
):
    """Simpan pesan chat baru beserta metrik evaluasi RAGAS dan Similarity Score"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                metadata_json = json.dumps(metadata) if metadata else None
                
                faithfulness = evaluation.get("faithfulness") if evaluation else None
                answer_relevance = evaluation.get("answer_relevancy") if evaluation else None
                context_precision = evaluation.get("context_precision") if evaluation else None
                context_recall = evaluation.get("context_recall") if evaluation else None
                noise_sensitivity = evaluation.get("noise_sensitivity") if evaluation else None
                semantic_similarity = evaluation.get("semantic_similarity") if evaluation else None
                
                cur.execute("""
                    INSERT INTO chat_history (
                        session_id, user_id, user_query, llm_response, metadata, created_at,
                        faithfulness, answer_relevance, context_precision, context_recall, 
                        noise_sensitivity, semantic_similarity
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    session_id, user_id, user_query, llm_response, metadata_json,
                    faithfulness, answer_relevance, context_precision, context_recall, 
                    noise_sensitivity, semantic_similarity
                ))
                
                cur.execute("UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s", (session_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error save_chat_message: {e}")
        return False