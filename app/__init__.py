import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():
    """
    Entry point aplikasi Flask - RAG Peraturan Desa.
    Menginisialisasi Flask, CORS, dan registrasi Blueprint routes.
    """
    app = Flask(__name__)

    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'rag-peraturan-desa-secret')

    # CORS — izinkan semua origin, expose header custom
    CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=["X-Chatbot-Text"])

    with app.app_context():
        from app.routes import bp
        app.register_blueprint(bp)

        # Validasi OPENAI_API_KEY
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("⚠️  WARNING: OPENAI_API_KEY tidak ditemukan. Pastikan .env sudah dimuat dengan benar.")
        else:
            print(f"✅ OPENAI_API_KEY ditemukan (***...{openai_key[-8:]})")
                
    return app
