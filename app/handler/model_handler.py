from flask import jsonify
from app.services.rag_service import hf_service, AVAILABLE_MODELS
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
        models = {}
        for model_id, info in AVAILABLE_MODELS.items():
            models[str(model_id)] = {
                "name": info["name"],
                "type": info["type"],
                "description": f"{info['name']} ({info['type']})"
            }
        
        return jsonify({
            "status": "success",
            "data": models
        }), 200
    except Exception as e:
        logger.error(f"Error in handle_get_model_info: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500
