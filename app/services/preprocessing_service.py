import os
import fitz  # PyMuPDF
import pytesseract
import io
import re
import json
import cv2         
import numpy as np
import psycopg2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_legal_text(text: str) -> str:
    if not text:
        return ""
    
    text = text.replace('\x00', '').replace('\xa0', ' ')
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # --- FIX ERROR KHUSUS OCR ---
    # 1. Fix "Menimbang $ a." menjadi "Menimbang : a."
    text = re.sub(r'(Menimbang|Mengingat|Memperhatikan|Menetapkan)\s*[$|S]\s*', r'\1 : ', text, flags=re.IGNORECASE)
    
    # 2. Fix "BAB !!" menjadi "BAB II"
    text = re.sub(r'BAB\s+!!', 'BAB II', text)
    
    # 3. Fix "p S" atau karakter aneh yang seharusnya angka "1." di awal list
    text = re.sub(r'(?m)^p\s*S\s*', '1. ', text)
    text = re.sub(r'(?m)^/\s*', '7. ', text) # Seringkali angka 7 miring dibaca garis miring
    
    # Rapatkan titik dua (:)
    text = re.sub(r'\n\s*:\s*', ' : ', text)
    text = re.sub(r'\s+:\s+', ' : ', text)
    
    # Atasi list yang terpisah enter
    text = re.sub(r'(?m)^([a-z]|\d{1,2})\.\s*\n+', r'\1. ', text)
    text = re.sub(r'(,\d*)\s*(\d{1,2}\.\s+[A-Z])', r'\1\n\n\2', text)
    
    # Normalisasi enter & spasi
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Bersihkan noise garis tabel yang terbaca Tesseract
    text = re.sub(r'[|_\-\[\]{}><]+', ' ', text)
    
    # Hapus nomor halaman yang berdiri sendiri
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r'(?m)^[a-zA-Z]\s*$', '', text)
    
    return text.strip()


def extract_text_from_pdf(file_path: str):
    try:
        pdf_doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error membuka file: {e}")
        return []

    documents = []

    # ========================================================
    # 1. LOGIKA UNTUK MENGEKSTRAK NAMA DESA DAN KABUPATEN
    # ========================================================
    village_name = "Unknown"
    regency_name = "Unknown"
    
    try:
        first_page = pdf_doc.load_page(0)
        first_page_text = first_page.get_text("text").strip()
        
        if first_page_text:
            lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
            
            # Cek maksimal 10 baris pertama untuk mencari KEPALA DESA dan KABUPATEN
            for i, line in enumerate(lines[:10]):
                upper_line = line.upper()
                
                # Cari pola "KEPALA DESA <NAMA_DESA>"
                if "KEPALA DESA" in upper_line:
                    # Ambil teks setelah "KEPALA DESA"
                    parts = upper_line.split("KEPALA DESA")
                    if len(parts) > 1:
                        village_name = parts[1].strip()
                
                # Cari pola "KABUPATEN <NAMA_KABUPATEN>" atau "KOTA <NAMA_KOTA>"
                if "KABUPATEN" in upper_line:
                    parts = upper_line.split("KABUPATEN")
                    if len(parts) > 1:
                        regency_name = parts[1].strip()
                elif "KOTA" in upper_line and i < 3:  # KOTA biasanya dekat dengan awal
                    parts = upper_line.split("KOTA")
                    if len(parts) > 1:
                        regency_name = "KOTA " + parts[1].strip()
        else:
            logger.warning("Halaman pertama kosong, menggunakan nama file sebagai fallback")
            
    except Exception as e:
        logger.warning(f"Gagal mengekstrak nama desa dan kabupaten: {e}")
    
    # Format title: "Desa <NAMA_DESA> - Kabupaten <NAMA_KABUPATEN>"
    document_title = f"Desa {village_name} - {regency_name}"

    # ========================================================
    # 2. LOOP EKSTRAKSI TEKS SEPERTI BIASA
    # ========================================================
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text("text", sort=True).strip()

        if not text:
            logger.info(f"Halaman {page_num + 1}: Menjalankan OCR...")
            
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
            img_data = pix.tobytes("png")
            img_pil = Image.open(io.BytesIO(img_data)).convert('L') 
            
            img_cv = np.array(img_pil)
            _, img_bin = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            final_img = Image.fromarray(img_bin)
            
            custom_ocr_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(final_img, lang='ind', config=custom_ocr_config)

        if text:
            clean_text = clean_legal_text(text)
            
            if clean_text:
                # TAMBAHKAN METADATA TITLE KE SINI
                doc = Document(
                    page_content=clean_text, 
                    metadata={
                        "source": os.path.basename(file_path), 
                        "title": document_title,                      # <--- Title ditambahkan di sini
                        "page": page_num + 1,
                        "total_pages": len(pdf_doc)
                    }
                )
                documents.append(doc)

    pdf_doc.close()
    return documents


def chunk_documents(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


def save_results_to_folder(file_path: str, extracted_docs: list, chunks: list):
    """
    Menyimpan hasil ekstraksi (.txt) dan hasil chunking (.json) ke dalam folder data/processed
    """
    # Buat direktori penyimpanan utama (misal di root_project/data/processed)
    output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Ambil nama file asli tanpa ekstensi .pdf
    base_filename = os.path.basename(file_path).replace('.pdf', '')
    
    # 1. Simpan Full Text (Ekstraksi) ke file .txt
    txt_path = os.path.join(output_dir, f"{base_filename}_extracted.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for doc in extracted_docs:
            f.write(f"--- HALAMAN {doc.metadata.get('page')} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")
            
    # 2. Simpan Data Chunking ke file .json
    json_path = os.path.join(output_dir, f"{base_filename}_chunks.json")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "chunk_index": i + 1,
            "metadata": chunk.metadata,
            "character_count": len(chunk.page_content),
            "content": chunk.page_content
        })
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Berhasil menyimpan hasil ekstraksi dan chunking ke folder: {output_dir}")



def save_chunks_to_postgres(chunks: list) -> bool:
    """
    Menyimpan data hasil chunking ke dalam tabel chunks_perdes di PostgreSQL.
    
    Schema tabel:
        CREATE TABLE chunks_perdes (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255),
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    
    Args:
        chunks: List berisi Document objects dengan page_content dan metadata
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    conn = None
    cursor = None
    
    try:
        # 1. Buka Koneksi ke Database menggunakan Settings Pydantic
        logger.info("🔗 Menghubungkan ke PostgreSQL database...")
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT", "")),
            database=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", "")
        )
        logger.debug(f"✅ Koneksi berhasil ke {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
        
        cursor = conn.cursor()
        
        # 2. Validasi tabel chunks_perdes ada
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chunks_perdes'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.error("❌ Tabel 'chunks_perdes' tidak ditemukan di database!")
            return False
        
        logger.info("✅ Tabel 'chunks_perdes' ditemukan")

        # 3. Siapkan Query Insert (sesuai schema tabel baru)
        insert_query = """
        INSERT INTO chunks_perdes (file_name, content)
        VALUES (%s, %s)
        """
        
        # 4. Eksekusi Query untuk Setiap Chunk (dengan validasi)
        inserted_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Ambil nilai dari metadata yang sudah kita buat saat ekstraksi
                file_name = chunk.metadata.get("source", "Unknown")
                content = chunk.page_content
                
                # Masukkan data ke kolom tabel
                cursor.execute(insert_query, (file_name, content))
                inserted_count += 1
                logger.debug(f"Chunk {i+1} inserted: {file_name}")
                
            except Exception as chunk_error:
                logger.error(f"❌ Error menyimpan chunk {i+1}: {chunk_error}")
                # Rollback dan stop jika terjadi error
                conn.rollback()
                return False

        # 5. Simpan Perubahan (COMMIT) dan Tutup Koneksi
        conn.commit()
        logger.info(f"✅ COMMIT berhasil! Total {inserted_count} chunks tersimpan di PostgreSQL")
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Gagal terhubung ke database PostgreSQL: {e}")
        logger.error(f"   Pastikan kredensial database benar: host={os.getenv('DB_HOST')}, port={os.getenv('DB_PORT')}, db={os.getenv('DB_NAME')}, user={os.getenv('DB_USER')}")
        return False
        
    except psycopg2.ProgrammingError as e:
        logger.error(f"❌ Error SQL di PostgreSQL: {e}")
        logger.error(f"   Mungkin nama kolom atau tipe data tidak sesuai")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error tidak terduga saat menyimpan ke PostgreSQL: {e}")
        return False
        
    finally:
        # 6. Pastikan koneksi ditutup
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        logger.debug("🔌 Koneksi PostgreSQL ditutup")


def extract_and_chunk_pdf(file_path: str):

    documents = extract_text_from_pdf(file_path)
    chunks = chunk_documents(documents)
    save_results_to_folder(file_path, documents, chunks)
    # save_chunks_to_postgres(chunks)

    return chunks

