import os
from flask import request, jsonify
from werkzeug.utils import secure_filename
from app.usecases.embedding_use_case import ingest_pdf_to_vector_db

def handle_generate_embedding():
    if 'file' not in request.files:
        return jsonify({"error": "Key 'file' tidak ditemukan dalam request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file PDF yang dipilih"}), 400

    save_to_db_str = request.form.get('save_to_db', 'true').lower()
    save_to_db = save_to_db_str in ['true', '1', 'yes']
        
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        
        file.save(filepath)
        
        result = ingest_pdf_to_vector_db(filepath, filename, save_to_db)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            
        status_code = 200 if result["status"] == "success" else 500
        return jsonify(result), status_code
    
    return jsonify({"error": "Format file tidak didukung. Harap unggah format PDF."}), 400