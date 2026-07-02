from flask import Blueprint
from app.handler.pdf_ingest_handler import handle_generate_embedding
# from app.handler.evaluation_handler import handle_evaluate_response
from app.handler.chat_handler import (
    handle_chat,
    handle_create_session,
    handle_get_sessions,
    handle_get_history,
    handle_update_session,
    handle_delete_session
)
from app.handler.model_handler import (
    handle_load_finetuned_model,
    handle_get_model_info
)

from app.handler.user_hendler import(
    handle_get_user
)

from app.handler.retrieval_handler import handle_test_retrieval

bp = Blueprint("routes", __name__)

# Endpoint RAG (Dokumen & Embedding)
bp.add_url_rule('/api/generate-embedding', 'generate_embedding', handle_generate_embedding, methods=['POST'])

# Endpoint Evaluasi RAGAS
# bp.add_url_rule('/api/evaluate-response', 'evaluate_response', handle_evaluate_response, methods=['POST'])

# Endpoint Model Management
bp.add_url_rule('/api/models', 'get_model_info', handle_get_model_info, methods=['GET'])

# Endpoint Chat dan Riwayat Session (Tersimpan di DB)
bp.add_url_rule('/api/chat', 'chat', handle_chat, methods=['POST'])
bp.add_url_rule('/api/chat-sessions', 'create_session', handle_create_session, methods=['POST'])
bp.add_url_rule('/api/chat-sessions', 'get_sessions', handle_get_sessions, methods=['GET'])
bp.add_url_rule('/api/chat-history/<int:session_id>', 'get_history', handle_get_history, methods=['GET'])
bp.add_url_rule('/api/chat-sessions/<int:session_id>', 'update_session', handle_update_session, methods=['PUT'])
bp.add_url_rule('/api/chat-sessions/<int:session_id>', 'delete_session', handle_delete_session, methods=['DELETE'])

# Endpoint User
bp.add_url_rule('/api/user/<int:id>', 'get_user', handle_get_user, methods=['GET'])

# Endpoint Test Retrieval (tanpa LLM generation)
bp.add_url_rule('/api/test-retrieval', 'test_retrieval', handle_test_retrieval, methods=['POST'])