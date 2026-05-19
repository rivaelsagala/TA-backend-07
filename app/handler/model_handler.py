from flask import jsonify
from app.services.rag_service import hf_service
from loguru import logger

def handle_load_finetuned_model():
    """Handler untuk load model fine-tuned ke memori server B200"""
    try:
        logger.info("Attempting to load fine-tuned model...")
        success = hf_service.load_finetuned_model()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model fine-tuned berhasil dimuat ke memori"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Gagal memuat model fine-tuned"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in handle_load_finetuned_model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

def handle_get_model_info():
    """Handler untuk mendapatkan informasi model yang tersedia"""
    try:
        return jsonify({
            "status": "success",
            "data": {
                "original_model": {
                    "name": hf_service.model,
                    "description": "Llama 3.1 8B Instruct (Original)",
                    "use_finetuned_model": False
                },
                "finetuned_model": {
                    "name": hf_service.finetuned_model_name,
                    "description": "Llama 3.1 8B Instruct Fine-tuned for Legal Documents",
                    "use_finetuned_model": True
                }
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in handle_get_model_info: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500
