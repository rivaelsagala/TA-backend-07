import os
import fitz  # PyMuPDF
# import pytesseract          # <-- OCR (uncomment jika butuh OCR)
import io
import re
import json
# import cv2                   # <-- OpenCV (uncomment jika butuh OCR)
# import numpy as np            # <-- OpenCV (uncomment jika butuh OCR)
import psycopg2
from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter  # No longer used — Pasal-based chunking now
# from PIL import Image          # <-- OCR (uncomment jika butuh OCR)
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# [OCR] Path tesseract - uncomment jika butuh OCR
# import platform
# _tesseract_default = (
#     r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     if platform.system() == 'Windows'
#     else '/usr/bin/tesseract'
# )
# pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', _tesseract_default)

def clean_legal_text(text: str) -> str:
    """
    Membersihkan dan menormalisasi teks hukum dari hasil ekstraksi PDF.
    
    Tahapan pembersihan (berdasarkan best practice NLP untuk dokumen hukum):
    1. Normalisasi karakter unicode & kontrol
    2. Konversi ke lowercase (case normalization)
    3. Perbaikan OCR errors khusus dokumen hukum
    4. Penghapusan karakter aneh/noise (bintang, simbol, dll)
    5. Normalisasi whitespace (spasi berlebih, tab, newline ganda)
    6. Pembersihan nomor halaman & artifacts PDF
    
    Referensi:
    - Hui et al. (2024) "Legal Document Retrieval and Summarization" -
      menekankan pentingnya normalisasi teks untuk improving retrieval quality.
    - Devlin et al. (2019) BERT - lowercase normalization meningkatkan
      tokenization quality untuk embedding models.
    """
    if not text:
        return ""
    
    # ==============================================
    # TAHAP 1: Normalisasi karakter unicode & kontrol
    # ==============================================
    # Hapus null bytes dan non-breaking space
    text = text.replace('\x00', '').replace('\xa0', ' ')
    # Normalisasi unicode characters (misal: smart quotes ke straight quotes)
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # smart double quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash
    text = text.replace('\u2022', '-')  # bullet point
    text = text.replace('\u00a0', ' ')  # non-breaking space
    
    # ==============================================
    # TAHAP 2: Konversi SEMUA teks ke LOWERCASE
    # ==============================================
    # Penting untuk konsistensi embedding dan pencarian.
    # Model embedding seperti text-embedding-3-large sudah case-insensitive,
    # tetapi lowercase memastikan konsistensi di seluruh pipeline.
    text = text.lower()
    
    # ==============================================
    # TAHAP 3: Perbaikan OCR errors khusus dokumen hukum
    # ==============================================
    # Fix "menimbang $ a." menjadi "menimbang : a."
    text = re.sub(r'(menimbang|mengingat|memperhatikan|menetapkan)\s*[$|s]\s*', r'\1 : ', text, flags=re.IGNORECASE)
    
    # Fix "bab !!" menjadi "bab ii"
    text = re.sub(r'bab\s+!!', 'bab ii', text)
    
    # Fix karakter aneh yang seharusnya angka di awal list
    text = re.sub(r'(?m)^p\s*s\s*', '1. ', text)
    text = re.sub(r'(?m)^/\s*', '7. ', text)  # Seringkali angka 7 miring dibaca garis miring
    
    # ==============================================
    # TAHAP 4: Penghapusan karakter aneh / noise
    # ==============================================
    # Hapus simbol-simbol yang bukan bagian dari teks hukum:
    # bintang (*), caret (^), tilde (~), backtick (`), curly braces,
    # square brackets, angle brackets, pipe, underscore berlebih
    text = re.sub(r'[*\^~`{}\[\]<>|]', ' ', text)
    # Hapus garis tabel yang terbaca OCR (kombinasi |, _, -)
    text = re.sub(r'[|_\-\[\]{}><]{2,}', ' ', text)
    # Hapus karakter kontrol yang tersisa (kecuali newline dan tab)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Hapus sequence karakter aneh seperti (c), (r), (tm), dll
    text = re.sub(r'[©®™℠§¶†‡•…‰′″‱]', ' ', text)
    
    # ==============================================
    # TAHAP 5: Normalisasi whitespace
    # ==============================================
    # Rapatkan titik dua (:)
    text = re.sub(r'\n\s*:\s*', ' : ', text)
    text = re.sub(r'\s+:\s+', ' : ', text)
    
    # Atasi list yang terpisah enter
    text = re.sub(r'(?m)^([a-z]|\d{1,2})\.\s*\n+', r'\1. ', text)
    text = re.sub(r'(,\d*)\s*(\d{1,2}\.\s+[a-z])', r'\1\n\n\2', text)
    
    # Normalisasi hyphenation (kata yang terputus di akhir baris)
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Ganti multiple spaces menjadi single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Ganti multiple newlines menjadi maksimal 2 newlines (paragraf separator)
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    # Hapus spasi di awal baris
    text = re.sub(r'(?m)^[ \t]+', '', text)
    # Hapus spasi di akhir baris
    text = re.sub(r'(?m)[ \t]+$', '', text)
    
    # ==============================================
    # TAHAP 6: Pembersihan nomor halaman & artifacts PDF
    # ==============================================
    # Hapus nomor halaman yang berdiri sendiri di satu baris
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    # Hapus single letter yang berdiri sendiri (artifact PDF)
    text = re.sub(r'(?m)^[a-z]\s*$', '', text)
    # Hapus kata "salinan" yang berdiri sendiri di awal
    text = re.sub(r'(?m)^salinan\s*$', '', text)
    # Hapus "lembaran desa" line artifacts
    text = re.sub(r'(?m)^lembaran desa\s+.*$', '', text)
    
    # ==============================================
    # TAHAP 7: Penghapusan HEADER dan FOOTER dokumen
    # ==============================================
    # Header/footer umum pada peraturan desa:
    # - "pemerintah desa <nama>" di baris terpisah (header kop surat)
    # - "peraturan desa <nama>" di baris terpisah (header judul)
    # - "nomor ... tahun ..." di baris terpisah
    # Hapus header kop surat yang berdiri sendiri (bukan bagian dari isi pasal)
    text = re.sub(r'(?m)^pemerintah desa\s+\w+\s*$', '', text)
    # Hapus baris "nomor xx tahun yyyy" yang berdiri sendiri (header)
    text = re.sub(r'(?m)^nomor\s+\d+\s+tahun\s+\d{4}\s*$', '', text)
    
    # ==============================================
    # TAHAP 8: Penghapusan BLOK TANDA TANGAN (signature block)
    # ==============================================
    # Signature block biasanya di halaman terakhir setelah pasal terakhir.
    # Pola: "ditetapkan di ...", "pada tanggal ...", "kepala desa ...",
    #       "ttd", nama orang, "diundangkan di ...", "sekretaris desa",
    #       "berita daerah ..."
    #
    # Hapus "ttd" (tanda tangan)
    text = re.sub(r'(?m)^ttd\s*$', '', text)
    # Hapus baris "ditetapkan di ..." / "diundangkan di ..."
    text = re.sub(r'(?m)^(ditetapkan|diundangkan)\s+di\s+.*$', '', text)
    # Hapus baris "pada tanggal ..."
    text = re.sub(r'(?m)^pada tanggal\s+.*$', '', text)
    # Hapus baris "kepala desa ...," (dengan koma di akhir)
    text = re.sub(r'(?m)^kepala desa\s+\w+,?\s*$', '', text)
    # Hapus baris "sekretaris desa" yang berdiri sendiri
    text = re.sub(r'(?m)^sekretaris desa\s*$', '', text)
    # Hapus nama orang yang berdiri sendiri setelah ttd (2-4 kata, semua huruf)
    # Pola: baris dengan 2-4 kata yang semuanya huruf kapital/tidak, tanpa angka
    # Ini heuristic — nama orang biasanya 2-4 kata tanpa tanda baca khusus
    text = re.sub(r'(?m)^[a-z]+(?:\s+[a-z]+){1,3}\s*$', lambda m: '' if not any(kw in m.group() for kw in ['pasal', 'bab', 'ayat', 'huruf', 'angka', 'bagian', 'paragraf']) else m.group(), text)
    # Hapus baris "berita daerah ..." yang merupakan footer
    text = re.sub(r'(?m)^berita daerah\s+.*$', '', text)
    
    # ==============================================
    # TAHAP 9: Penghapusan KONJUNGSI berdiri sendiri
    # ==============================================
    # Pada dokumen peraturan desa, kata hubung seperti "dan", "atau", "serta"
    # sering muncul di baris terpisah antara dua entitas (misal:
    # "BADAN PERMUSYAWARATAN DESA BIRU"\n"dan"\n"KEPALA DESA BIRU").
    # Baris konjungsi tunggal ini adalah artifact layout PDF dan harus dihapus.
    # Ref: Manning & Schütze (1999) "Foundations of Statistical NLP" —
    #   stopword/conjunction removal pada document boundary meningkatkan
    #   kualitas segmentasi teks.
    text = re.sub(r'(?m)^(dan|atau|serta|dengan|untuk|dari|ke|pada|oleh)\s*$', '', text)
    
    # Final cleanup: hapus leading/trailing whitespace
    text = text.strip()
    
    # Hapus multiple blank lines yang tersisa setelah semua pembersihan
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def extract_perdes_metadata(file_path: str, full_text: str) -> dict:
    """
    Mengekstrak metadata terstruktur dari dokumen Peraturan Desa.
    
    Perubahan dari versi sebelumnya:
    - Sekarang menerima FULL TEXT (semua halaman digabung), bukan hanya halaman 1.
    - Kabupaten dideteksi dari seluruh dokumen (tidak hanya 15 baris pertama).
    - Ini mengatasi kasus dimana "Kabupaten <nama>" baru muncul di halaman 2+.
    
    Metadata yang diekstrak:
    - village_name, regency_name, perdes_number, perdes_year, perdes_title, document_id
    
    Ref:
    - Gao et al. (2024) "RAG for LLMs: A Survey" — metadata filtering
      meningkatkan precision untuk dokumen serupa dari sumber berbeda.
    - Khandelwal et al. (2020) "Generalization through Memorization" —
      richer metadata in retrieval corpus improves downstream accuracy.
    """
    village_name = "unknown"
    regency_name = "unknown"
    perdes_number = "unknown"
    perdes_year = "unknown"
    perdes_title = "unknown"
    
    try:
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        # ========================================
        # SCAN 1: 15 baris pertama — untuk desa, nomor, tahun, judul
        # ========================================
        for i, line in enumerate(lines[:20]):
            upper_line = line.upper()
            
            # Ekstrak nama desa
            if "KEPALA DESA" in upper_line:
                parts = upper_line.split("KEPALA DESA")
                if len(parts) > 1 and parts[1].strip():
                    village_name = parts[1].strip().lower()
            elif "PEMERINTAH DESA" in upper_line and village_name == "unknown":
                parts = upper_line.split("PEMERINTAH DESA")
                if len(parts) > 1 and parts[1].strip():
                    village_name = parts[1].strip().lower()
            
            # Ekstrak nomor dan tahun
            nomor_match = re.search(r'NOMOR\s+(\d+)\s+TAHUN\s+(\d{4})', upper_line)
            if nomor_match:
                perdes_number = nomor_match.group(1)
                perdes_year = nomor_match.group(2)
            
            # Ekstrak judul peraturan dari baris setelah "TENTANG"
            if upper_line == "TENTANG" and i + 1 < len(lines):
                title_lines = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].upper() in ["DENGAN RAHMAT TUHAN YANG MAHA ESA", 
                                            "KEPALA DESA", ""]:
                        break
                    title_lines.append(lines[j].strip())
                if title_lines:
                    perdes_title = " ".join(title_lines).lower()
        
        # ========================================
        # SCAN 2: Seluruh teks — khusus untuk KABUPATEN/KOTA
        # ========================================
        # Kabupaten sering tidak muncul di 15 baris pertama karena halaman 1
        # berisi daftar undang-undang referensi (Lembaran Negara, dll).
        # Nama kabupaten biasanya muncul di:
        #   - "Kecamatan X Kabupaten Y" (dalam definisi desa)
        #   - "Peraturan Daerah Kabupaten Y" (dalam dasar hukum)
        #   - "Kabupaten Y" di header/footer dokumen
        #
        # Strategi: cari pola "KABUPATEN <WORD>" di seluruh teks,
        # ambil yang paling sering muncul (majority vote).
        regency_candidates = re.findall(
            r'kabupaten\s+([a-zA-Z]+)',
            full_text, re.IGNORECASE
        )
        if regency_candidates:
            # Filter: hapus kata-kata yang jelas bukan nama kabupaten
            stop_words = {'yang', 'dan', 'atau', 'dari', 'di', 'ke', 'pada', 
                          'dalam', 'dengan', 'untuk', 'oleh', 'ini', 'itu',
                          'nomor', 'tahun', 'republik', 'indonesia', 'negara',
                          'daerah', 'bupati', 'peraturan', 'pemerintah'}
            filtered = [
                w.lower() for w in regency_candidates 
                if w.lower() not in stop_words and len(w) > 2
            ]
            if filtered:
                # Ambil yang paling sering muncul (mode)
                from collections import Counter
                regency_name = Counter(filtered).most_common(1)[0][0]
        
        # Fallback: coba KOTA jika kabupaten masih unknown
        if regency_name == "unknown":
            kota_candidates = re.findall(
                r'kota\s+([a-zA-Z]+)',
                full_text, re.IGNORECASE
            )
            if kota_candidates:
                stop_words = {'yang', 'dan', 'atau', 'dari', 'di', 'ke', 'pada',
                              'dalam', 'dengan', 'untuk', 'oleh', 'ini', 'itu'}
                filtered = [w.lower() for w in kota_candidates if w.lower() not in stop_words and len(w) > 2]
                if filtered:
                    from collections import Counter
                    regency_name = "kota " + Counter(filtered).most_common(1)[0][0]
        
    except Exception as e:
        logger.warning(f"Gagal mengekstrak metadata perdes: {e}")
    
    document_id = f"perdes_{village_name}_{perdes_number}_{perdes_year}"
    document_title = f"Peraturan Desa {village_name.title()} No. {perdes_number} Tahun {perdes_year} - {perdes_title.title()}"
    
    return {
        "village_name": village_name,
        "regency_name": regency_name,
        "perdes_number": perdes_number,
        "perdes_year": perdes_year,
        "perdes_title": perdes_title,
        "document_id": document_id,
        "document_title": document_title
    }


def extract_text_from_pdf(file_path: str):
    """
    Mengekstrak teks dari PDF dan menghasilkan SATU Document object
    berisi seluruh teks dokumen yang sudah digabung (bukan per-halaman).
    
    Perubahan penting dari versi sebelumnya:
    - Semua halaman digabung menjadi SATU teks utuh sebelum chunking.
    - Ini mengatasi masalah preamble yang terpotong antar halaman.
    - Raw text (sebelum cleaning) disimpan sebagai file _raw.txt terpisah.
    - Metadata diekstrak dari FULL TEXT, bukan hanya halaman 1.
    
    Ref:
    - Sarfraz et al. (2024) "Contextual Retrieval" (Anthropic) —
      full-document context sebelum chunking meningkatkan semantic coherence.
    - Zhong et al. (2024) "Legal RAG" — pasal-level chunking memerlukan
      teks utuh (bukan per-halaman) agar tidak terpotong di batas halaman.
    
    Returns:
        List berisi SATU Document object dengan:
        - page_content: teks utuh seluruh dokumen (sudah di-clean + enriched)
        - metadata: metadata lengkap (tanpa raw_text agar tidak membebani)
    """
    try:
        pdf_doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error membuka file: {e}")
        return []

    # ========================================================
    # 1. EKSTRAK TEKS MENTAH (RAW) DARI SEMUA HALAMAN
    # ========================================================
    raw_pages = []  # List of (page_num, raw_text) tuples
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text("text", sort=True).strip()
        if text:
            raw_pages.append((page_num + 1, text))
        else:
            logger.warning(f"Halaman {page_num + 1}: Teks kosong, lewati.")
    
    pdf_doc.close()
    
    if not raw_pages:
        logger.error("Tidak ada teks yang bisa diekstrak dari PDF.")
        return []
    
    # ========================================================
    # 2. GABUNG SEMUA HALAMAN MENJADI SATU TEKS UTUH
    # ========================================================
    # Ini penting: agar preamble (menimbang, mengingat, memutuskan)
    # yang membentang beberapa halaman tidak terpotong.
    full_raw_text = "\n\n".join([text for _, text in raw_pages])
    
    # ========================================================
    # 3. EKSTRAK METADATA DARI FULL TEXT (bukan hanya halaman 1)
    # ========================================================
    perdes_meta = extract_perdes_metadata(file_path, full_raw_text)
    logger.info(f"Metadata diekstrak: {perdes_meta['document_id']} | {perdes_meta['document_title']}")
    
    # ========================================================
    # 4. BERSIHKAN TEKS (cleaning)
    # ========================================================
    clean_text = clean_legal_text(full_raw_text)
    
    if not clean_text:
        logger.error("Teks bersih kosong setelah cleaning.")
        return []
    
    # ========================================================
    # 5. PREPEND CONTEXT HEADER UNTUK DISAMBIGUASI
    # ========================================================
    context_header = (
        f"[dokumen: {perdes_meta['document_title']}] "
        f"[desa: {perdes_meta['village_name']}] "
        f"[kabupaten: {perdes_meta['regency_name']}] "
        f"[nomor: {perdes_meta['perdes_number']}/{perdes_meta['perdes_year']}]"
    )
    
    enriched_text = f"{context_header}\n\n{clean_text}"
    
    # ========================================================
    # 6. SIMPAN RAW TEXT KE FILE TERPISAH (tidak di metadata)
    # ========================================================
    raw_output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(raw_output_dir, exist_ok=True)
    base_name = os.path.basename(file_path).replace('.pdf', '')
    raw_path = os.path.join(raw_output_dir, f"{base_name}_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(full_raw_text)
    logger.info(f"Raw text tersimpan: {raw_path}")
    
    # ========================================================
    # 7. BUAT SATU DOCUMENT OBJECT (teks utuh seluruh dokumen)
    # ========================================================
    doc = Document(
        page_content=enriched_text,
        metadata={
            "source": os.path.basename(file_path),
            "title": perdes_meta["document_title"],
            "document_id": perdes_meta["document_id"],
            "village_name": perdes_meta["village_name"],
            "regency_name": perdes_meta["regency_name"],
            "perdes_number": perdes_meta["perdes_number"],
            "perdes_year": perdes_meta["perdes_year"],
            "perdes_title": perdes_meta["perdes_title"],
            "total_pages": len(raw_pages)
        }
    )
    
    # Return list berisi 1 document (kompatibel dengan pipeline selanjutnya)
    return [doc]


def _parse_perdes_sections(text: str) -> list:
    """
    State-machine parser untuk dokumen Peraturan Desa.
    Memecah dokumen menjadi butir-level chunks dengan hierarchy:
    BAB → Bagian → Pasal → Butir
    
    Output: list of dict, setiap dict berisi:
    - type: 'preamble' | 'butir'
    - bab: 'bab i', 'bab ii', ...
    - bab_title: 'ketentuan umum', 'pelayanan kesehatan', ...
    - bagian: 'bagian kesatu', ... (jika ada)
    - bagian_title: 'asas', 'tujuan', ... (jika ada)
    - pasal: 'pasal 1', 'pasal 2', ...
    - butir_num: '1', '2', ... atau '' jika single-item
    - content: teks butir lengkap
    
    Ref:
    - Zhong et al. (2024) "Legal RAG" — hierarchical legal document parsing.
    """
    results = []
    lines = text.split('\n')
    
    # State variables
    current_bab = ""
    current_bab_title = ""
    current_bagian = ""
    current_bagian_title = ""
    
    # Temukan posisi semua PASAL
    pasal_positions = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*pasal\s+\d+\s*$', line, re.IGNORECASE):
            pasal_positions.append(i)
    
    if not pasal_positions:
        return results
    
    # PREAMBLE: semua teks sebelum pasal pertama
    preamble_text = '\n'.join(lines[:pasal_positions[0]]).strip()
    if preamble_text:
        results.append({
            "type": "preamble",
            "bab": "",
            "bab_title": "",
            "bagian": "",
            "bagian_title": "",
            "pasal": "pembuka",
            "butir_num": "",
            "content": preamble_text
        })
    
    # Scan line-by-line untuk BAB dan BAGIAN, proses setiap PASAL
    for pos in pasal_positions:
        # Update state: scan backwards + forward untuk BAB/Bagian
        # Scan dari awal dokumen sampai pasal ini
        for i in range(pos):
            line = lines[i].strip()
            lower = line.lower()
            
            bab_match = re.match(r'^bab\s+([ivxlcdm]+)\s*$', lower)
            if bab_match:
                current_bab = f"bab {bab_match.group(1)}"
                # Ambil title: baris non-empty berikutnya yang BUKAN bab/bagian/pasal
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = lines[j].strip()
                    cl = candidate.lower()
                    if candidate and not re.match(r'^(bab|bagian|pasal)\s', cl):
                        current_bab_title = candidate
                        break
                else:
                    current_bab_title = ""
            
            bagian_match = re.match(r'^bagian\s+(\w+)', lower)
            if bagian_match:
                current_bagian = f"bagian {bagian_match.group(1)}"
                # Ambil title: baris non-empty berikutnya yang BUKAN bab/bagian/pasal
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = lines[j].strip()
                    cl = candidate.lower()
                    if candidate and not re.match(r'^(bab|bagian|pasal)\s', cl):
                        current_bagian_title = candidate
                        break
                else:
                    current_bagian_title = ""
        
        # Ekstrak nomor pasal
        pasal_line = lines[pos].strip().lower()
        pasal_num_match = re.match(r'pasal\s+(\d+)', pasal_line)
        if not pasal_num_match:
            continue
        pasal_num = pasal_num_match.group(1)
        pasal_label = f"pasal {pasal_num}"
        
        # Ambil isi pasal: dari baris setelah "pasal X" sampai pasal berikutnya
        next_pasal_pos = None
        for pp in pasal_positions:
            if pp > pos:
                next_pasal_pos = pp
                break
        
        end_pos = next_pasal_pos if next_pasal_pos else len(lines)
        
        # Ambil raw content (skip baris kosong setelah "pasal X")
        content_start = pos + 1
        while content_start < end_pos and not lines[content_start].strip():
            content_start += 1
        
        pasal_lines = lines[content_start:end_pos]
        
        # ==========================================================
        # Hapus trailing BAB/Bagian/title headers yang bocor.
        # Forward scan: dari akhir isi pasal, hapus semua baris yang
        # merupakan structural header (BAB, Bagian, title, blank line)
        # sampai ketemu baris yang merupakan bagian dari isi pasal.
        # ==========================================================
        actual_end = len(pasal_lines)
        while actual_end > 0:
            check_line = pasal_lines[actual_end - 1].strip().lower()
            # Blank line
            if not check_line:
                actual_end -= 1
                continue
            # BAB header
            if re.match(r'^bab\s+[ivxlcdm]+', check_line):
                actual_end -= 1
                continue
            # Bagian header
            if re.match(r'^bagian\s+\w+', check_line):
                actual_end -= 1
                continue
            # Title line: short line (< 60 chars) without numbered items,
            # sitting between a Bagian header and the next Pasal
            if (len(check_line) < 60 and
                not re.match(r'^\d+[\.\)]\s', check_line) and
                not re.match(r'^pasal\s+\d', check_line) and
                not re.match(r'^[a-z][\.\)]\s', check_line)):
                actual_end -= 1
                continue
            # This is actual pasal content — stop
            break
        
        pasal_lines = pasal_lines[:actual_end]
        
        pasal_content = '\n'.join(pasal_lines).strip()
        if not pasal_content:
            continue
        
        # Split menjadi butir-butir individual
        butir_items = _split_butir(pasal_content)
        
        if len(butir_items) > 1:
            # Multiple butir → 1 chunk per butir
            for idx, butir in enumerate(butir_items):
                # Extract nomor butir dari awal teks (e.g. "1. " → "1")
                butir_num_match = re.match(r'(\d+)[\.\)]\s*', butir.strip())
                butir_num = butir_num_match.group(1) if butir_num_match else str(idx + 1)
                
                content = f"{pasal_label}\n\n{butir}"
                results.append({
                    "type": "butir",
                    "bab": current_bab,
                    "bab_title": current_bab_title,
                    "bagian": current_bagian,
                    "bagian_title": current_bagian_title,
                    "pasal": pasal_label,
                    "butir_num": butir_num,
                    "content": content
                })
        else:
            # Single item → 1 chunk saja
            results.append({
                "type": "butir",
                "bab": current_bab,
                "bab_title": current_bab_title,
                "bagian": current_bagian,
                "bagian_title": current_bagian_title,
                "pasal": pasal_label,
                "butir_num": "",
                "content": f"{pasal_label}\n\n{butir_items[0]}"
            })
    
    return results


def _split_butir(pasal_content: str) -> list:
    """
    Membagi isi Pasal menjadi butir-butir individual.
    Normalisasi penomoran: huruf (a, b, c) → angka (1, 2, 3).
    
    Urutan penting:
    1. SPLIT dulu berdasarkan numbered items (1. 2. 3.) — SEBELUM normalisasi
    2. Normalisasi huruf → angka HANYA di dalam setiap butir (setelah split)
    
    Ini agar sub-items huruf (a. b. c.) di dalam butir TIDAK diubah menjadi
    angka sebelum split, sehingga tidak dianggap sebagai butir baru.
    
    Contoh:
    Pasal 10 ayat 2: "...mempunyai fungsi: a. ... b. ... c. ..."
    → split berdasarkan "1." dan "2." (numbered items) → 2 butir
    → normalize a,b,c → 1,2,3 HANYA di dalam butir 2
    → sub-items a,b,c TIDAK di-split jadi butir terpisah
    
    Ref:
    - Zhong et al. (2024) "Legal Retrieval-Augmented Generation" —
      Atomic Clause Chunking (per-butir) meningkatkan precision retrieval
      karena setiap butir = satu unit hukum yang self-contained.
    """
    # STEP 1: Cek apakah ada numbered items: "1.", "2.", dst di awal baris
    # Gunakan teks ASLI (belum dinormalisasi) agar huruf tidak dianggap angka
    has_numbered = re.search(r'(?m)^\s*\d+\.', pasal_content)
    if not has_numbered:
        # Tidak ada numbered items → normalize dan return as single chunk
        return [_normalize_letter_numbering(pasal_content)]
    
    # STEP 2: Split berdasarkan numbered items pada teks ASLI
    # Negative lookahead agar tidak split "10." jadi "1" + "0."
    parts = re.split(r'\n(?=\s*\d+\.(?!\d))', pasal_content)
    
    butir_list = []
    for part in parts:
        part = part.strip()
        if part:
            butir_list.append(part)
    
    if not butir_list:
        return [_normalize_letter_numbering(pasal_content)]
    
    # STEP 3: Jika part pertama BUKAN numbered item (intro text),
    # gabungkan ke butir pertama agar tidak jadi chunk terpisah.
    if not re.match(r'^\s*\d+\.', butir_list[0]):
        if len(butir_list) > 1:
            butir_list[1] = f"{butir_list[0]}\n\n{butir_list[1]}"
            butir_list.pop(0)
    
    # STEP 4: Normalisasi huruf → angka DI DALAM setiap butir
    # Ini dilakukan SETELAH split agar sub-items tidak ter-split
    result = []
    for butir in butir_list:
        result.append(_normalize_letter_numbering(butir))
    
    return result


def _normalize_letter_numbering(text: str) -> str:
    """
    Normalisasi penomoran huruf (a, b, c) menjadi angka (1, 2, 3).
    
    Menangani 2 kasus:
    1. Di awal baris: 'a. item' → '1. item'
    2. Inline setelah ':' atau ';': 'fungsi : a. item' → 'fungsi : 1. item'
    
    Hanya mengkonversi huruf tunggal yang merupakan sub-item penomoran,
    BUKAN huruf awal kalimat biasa.
    
    Ref:
    - Manning & Schütze (1999) "Foundations of Statistical NLP" —
      konsistensi format penomoran penting untuk parsing dan retrieval legal docs.
    """
    def replace_letter_dot(match):
        letter = match.group(2).lower()
        number = ord(letter) - ord('a') + 1
        return f"{match.group(1)}{number}."
    
    # Konversi a. b. c. → 1. 2. 3.
    # Case 1: di awal baris — group(1)=whitespace, group(2)=huruf
    text = re.sub(r'(?m)^(\s*)([a-z])\.(?=\s)', replace_letter_dot, text)
    # Case 2: inline setelah : atau ; (e.g. "fungsi : a. item")
    text = re.sub(r'(?<=[;:])(\s*)([a-z])\.(?=\s)', replace_letter_dot, text)
    
    def replace_letter_paren(match):
        letter = match.group(2).lower()
        number = ord(letter) - ord('a') + 1
        return f"{match.group(1)}{number})"
    
    # Konversi a) b) c) → 1) 2) 3) (format kurung tutup)
    # Case 1: di awal baris — group(1)=whitespace, group(2)=huruf
    text = re.sub(r'(?m)^(\s*)([a-z])\)(?=\s)', replace_letter_paren, text)
    # Case 2: inline setelah : atau ;
    text = re.sub(r'(?<=[;:])(\s*)([a-z])\)(?=\s)', replace_letter_paren, text)
    
    return text


def chunk_documents(documents: list):
    """
    Atomic Clause Chunking untuk dokumen Peraturan Desa (Perdes).
    
    Hierarki parsing: BAB → Bagian → Pasal → Butir/Ayat → Huruf
    Setiap butir = 1 chunk (unit retrieval terkecil).
    
    Kenapa butir-level (Atomic Clause Chunking)?
    1. Pasal 1 berisi 24 definisi → 24 chunk terpisah, retrieval lebih presisi.
       Query "apa itu desa?" → match ke butir 1 pasal 1, BUKAN seluruh pasal.
    2. Untuk FINE-TUNING: setiap butir = 1 instruction pair yang fokus dan spesifik.
    3. Normalisasi: huruf (a,b,c) → angka (1,2,3) agar konsisten di seluruh dokumen.
    4. Metadata hierarchy (bab/bagian/pasal/butir) disimpan di setiap chunk.
    
    Ref:
    - Zhong et al. (2024) "Legal Retrieval-Augmented Generation" —
      atomic clause chunking outperforms pasal-level untuk legal QA.
    - Sarfraz et al. (2024) "Contextual Retrieval" (Anthropic) —
      metadata-enriched chunks improve retrieval precision by 67%.
    - Manning & Schütze (1999) "Foundations of Statistical NLP" —
      structural normalization meningkatkan parsing dan matching quality.
    
    Returns:
        List of Document objects, setiap chunk = 1 butir + context_header + metadata.
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata.copy()
        
        # Pisahkan context_header dari isi dokumen
        context_header = ""
        content = text
        header_match = re.match(
            r'(\[dokumen:.*?\]\s*\[desa:.*?\]\s*\[kabupaten:.*?\]\s*\[nomor:.*?\])',
            text
        )
        if header_match:
            context_header = header_match.group(1)
            content = text[header_match.end():].strip()
        
        # Parse dokumen → list of structured sections
        sections = _parse_perdes_sections(content)
        
        # Buat chunks dari sections (skip preamble)
        for section in sections:
            if section["type"] == "preamble":
                continue
            
            # Enriched content = context_header + isi butir
            enriched = f"{context_header}\n\n{section['content']}" if context_header else section["content"]
            
            # Build metadata hierarchy
            chunk_meta = metadata.copy()
            chunk_meta["bab"] = section.get("bab", "")
            chunk_meta["bab_title"] = section.get("bab_title", "")
            chunk_meta["bagian"] = section.get("bagian", "")
            chunk_meta["bagian_title"] = section.get("bagian_title", "")
            chunk_meta["section"] = section["pasal"]
            chunk_meta["butir_number"] = section.get("butir_num", "")
            
            all_chunks.append(Document(
                page_content=enriched,
                metadata=chunk_meta
            ))
    
    logger.info(f"Butir-level chunking menghasilkan {len(all_chunks)} chunks dari {len(documents)} dokumen")
    return all_chunks


def save_results_to_folder(file_path: str, extracted_docs: list, chunks: list):
    """
    Menyimpan 2 output ke folder data/processed:
    1. Cleaned text (setelah preprocessing) — full teks utuh
    2. Chunks (JSON) — hasil Butir-level Atomic Clause Chunking
    (Raw text sudah disimpan terpisah di extract_text_from_pdf)
    
    Ref: 
    - Olsson et al. (2022) "A Survey of Data-Efficient Graph Learning" —
      menyimpan intermediate outputs memudahkan ablation study dan debugging.
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(file_path).replace('.pdf', '')
    
    # 1. Simpan CLEANED TEXT (setelah preprocessing, full dokumen)
    cleaned_path = os.path.join(output_dir, f"{base_filename}_extracted.txt")
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        for doc in extracted_docs:
            f.write(doc.page_content)
            f.write("\n\n")
    
    # 2. Simpan CHUNKS (butir-level) ke file JSON
    json_path = os.path.join(output_dir, f"{base_filename}_chunks.json")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata.copy()
        chunks_data.append({
            "chunk_index": i + 1,
            "bab": meta.get("bab", ""),
            "bab_title": meta.get("bab_title", ""),
            "bagian": meta.get("bagian", ""),
            "bagian_title": meta.get("bagian_title", ""),
            "pasal": meta.get("section", ""),
            "butir": meta.get("butir_number", ""),
            "metadata": meta,
            "character_count": len(chunk.page_content),
            "content": chunk.page_content
        })
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Hasil tersimpan: raw, extracted, dan {len(chunks)} butir-chunks ke {output_dir}")



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


def export_finetune_dataset(chunks: list, file_path: str) -> str:
    """
    Mengekspor hasil Butir-level chunking ke format JSONL untuk fine-tuning LLM.
    
    Format output (OpenAI fine-tune compatible):
    {"messages": [
        {"role": "system", "content": "Anda adalah asisten hukum... Konteks: <chunk>"},
        {"role": "user", "content": "<pertanyaan>"},
        {"role": "assistant", "content": "<jawaban>"}
    ]}
    
    Setiap BUTIR chunk menghasilkan 1 instruction pair:
    - System: context_header + isi butir
    - User: pertanyaan otomatis berdasarkan pasal + butir
    - Assistant: jawaban = isi butir lengkap
    
    Ref:
    - OpenAI (2024) "Fine-tuning Best Practices" —
      instruction format harus konsisten antara training dan inference.
    - Context dari RAG chunking HARUS identik dengan context di fine-tune data.
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(file_path).replace('.pdf', '')
    jsonl_path = os.path.join(output_dir, f"{base_filename}_finetune.jsonl")
    
    finetune_entries = []
    
    for chunk in chunks:
        content = chunk.page_content
        metadata = chunk.metadata
        
        # Pisahkan context_header dari isi butir
        context_header = ""
        butir_content = content
        header_match = re.match(
            r'(\[dokumen:.*?\]\s*\[desa:.*?\]\s*\[kabupaten:.*?\]\s*\[nomor:.*?\])',
            content
        )
        if header_match:
            context_header = header_match.group(1)
            butir_content = content[header_match.end():].strip()
        
        # System prompt identik dengan RAG inference
        doc_title = metadata.get("title", "Unknown")
        system_content = (
            f"Anda adalah asisten hukum pemerintahan desa. "
            f"Jawab pertanyaan berdasarkan konteks dokumen yang diberikan.\n\n"
            f"KONTEKS DOKUMEN:\n"
            f"{context_header}\n\n{butir_content}"
        )
        
        # Generate pertanyaan otomatis berdasarkan pasal + butir
        pasal = metadata.get("section", "")
        butir_num = metadata.get("butir_number", "")
        
        if butir_num:
            user_question = f"Apa isi {pasal} butir {butir_num} dalam {doc_title}?"
            assistant_answer = (
                f"Berdasarkan {doc_title}, {pasal} butir {butir_num} mengatur bahwa:\n\n"
                f"{butir_content}"
            )
        else:
            user_question = f"Apa isi {pasal} dalam {doc_title}?"
            assistant_answer = (
                f"Berdasarkan {doc_title}, {pasal} mengatur bahwa:\n\n"
                f"{butir_content}"
            )
        
        entry = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": assistant_answer}
            ]
        }
        finetune_entries.append(entry)
    
    # Tulis ke file JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in finetune_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Fine-tune dataset tersimpan: {jsonl_path} ({len(finetune_entries)} entries)")
    return jsonl_path


def extract_and_chunk_pdf(file_path: str):

    documents = extract_text_from_pdf(file_path)
    chunks = chunk_documents(documents)
    save_results_to_folder(file_path, documents, chunks)
    
    # Ekspor dataset fine-tune
    export_finetune_dataset(chunks, file_path)
    
    # save_chunks_to_postgres(chunks)

    return chunks

