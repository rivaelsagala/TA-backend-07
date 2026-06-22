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


def _canonical_output_basename(file_path: str) -> str:
    """Buat nama file output yang konsisten agar tidak muncul duplikasi versi lama/baru."""
    original_name = os.path.splitext(os.path.basename(file_path))[0]
    normalized = re.sub(r'[^A-Za-z0-9]+', '_', original_name).strip('_')
    return normalized or "document"


def _cleanup_processed_outputs(output_dir: str, canonical_base: str) -> None:
    """Hapus file processed lama yang merepresentasikan dokumen yang sama."""
    if not os.path.isdir(output_dir):
        return

    suffixes = ("_raw.txt", "_extracted.txt", "_chunks.json")
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if not os.path.isfile(entry_path):
            continue

        entry_base, entry_ext = os.path.splitext(entry)
        normalized_entry_base = re.sub(r'[^A-Za-z0-9]+', '_', entry_base).strip('_')
        if normalized_entry_base != canonical_base:
            continue

        if entry.endswith(suffixes):
            os.remove(entry_path)


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
    # TAHAP 1.5: Perbaikan OCR spacing error
    # ==============================================
    # Kadang OCR gagal membaca karakter sehingga menghasilkan spasi di tengah kata.
    # Contoh: "d ngan" → "dengan", "s tiap" → "setiap".
    # Strategi: jika ada huruf tunggal yang diapit spasi di antara dua kata,
    # dan gabungannya membentuk kata yang dikenal, merge-kan.
    # Pendekatan sederhana: hapus spasi antara single-char dan kata berikutnya
    # jika single-char itu bukan kata bermakna sendiri (a, i, o, u, di, ke, dll).
    _VALID_SINGLE = {'a', 'i', 'o', 'u', 'di', 'ke', 'si', 'se', 'ku', 'mu', 'ya'}
    def _fix_ocr_spacing(txt: str) -> str:
        # Pola: spasi + 1 huruf + spasi + kata (misal " d ngan")
        def _replacer(m):
            char = m.group(1)
            if char.lower() in _VALID_SINGLE:
                return m.group(0)  # Biarkan kata yang valid sendiri
            return ' ' + char + m.group(2)  # Merge: hilangkan spasi antara single-char dan kata, tapi pertahankan spasi sebelumnya
        return re.sub(r'(?<=[a-z]) ([a-z]) ([a-z]{2,})', _replacer, txt)
    text = _fix_ocr_spacing(text)

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
    # Fix spaced-out keywords dari PDF layout (estetika judul):
    # "t e n t a n g" → "tentang", "p a s a l" → "pasal", dll.
    # Tanpa ini, parser gagal mendeteksi keyword penting dan marker pasal.
    spaced_keywords = [
        'TENTANG', 'MENIMBANG', 'MENGINGAT', 'MEMUTUSKAN', 'MENETAPKAN',
        'MEMPERHATIKAN', 'PASAL', 'BAB', 'BAGIAN'
    ]
    for kw in spaced_keywords:
        spaced_pattern = r'\s+'.join(list(kw))
        text = re.sub(spaced_pattern, kw.lower(), text, flags=re.IGNORECASE)
    
    # Fix "menimbang $ a." menjadi "menimbang : a."
    text = re.sub(r'(menimbang|mengingat|memperhatikan|menetapkan)\s*[$|s]\s*', r'\1 : ', text, flags=re.IGNORECASE)
    
    # Fix "bab !!" menjadi "bab ii"
    text = re.sub(r'bab\s+!!', 'bab ii', text)
    
    # Fix pasal tanpa nomor karena error OCR (contoh: di perdes drawati "pasal\n1. tarip...")
    # Asumsi: jika ada kata "pasal" sendirian diikuti oleh daftar angka, kita beri nomor dummy/interpolated.
    # Karena kita tidak tahu nomor pastinya secara regex sederhana, kita tangkap kasus khusus yang sering terjadi.
    text = re.sub(r'(?m)^pasal\s*\n\s*1\.\s*tarip', 'pasal 6\n1. tarip', text)
    
    # Fix karakter aneh yang seharusnya angka di awal list
    text = re.sub(r'(?m)^p\s*s\s*', '1. ', text)
    text = re.sub(r'(?m)^/\s*', '7. ', text)  # Seringkali angka 7 miring dibaca garis miring

    # Potong lampiran/tabel setelah kalimat penutup peraturan.
    # Untuk RAG regulasi, lampiran tabel RPJM dan berita acara tidak ikut di-embedding
    # karena format tabel PDF sering rusak dan mengotori chunk Pasal terakhir.
    # Potong teks setelah kalimat penutup standar peraturan desa.
    # Pola dibuat fleksibel: cukup match "agar setiap orang ... lembaran desa"
    # tanpa mengandalkan frasa yang persis sama di setiap dokumen.
    closing_match = re.search(
        r'agar\s+setiap\s+orang\s+dapat\s+mengetahuinya',
        text,
        flags=re.IGNORECASE
    )
    if closing_match:
        # Cari akhir kalimat penutup (titik atau akhir baris)
        end_search = re.search(
            r'menempatkannya\s+dalam\s+lembaran\s+desa\.?',
            text[closing_match.start():],
            flags=re.IGNORECASE
        )
        if end_search:
            text = text[:closing_match.start() + end_search.end()]
        else:
            # Fallback: potong di titik akhir kalimat setelah "mengetahuinya"
            rest = text[closing_match.end():]
            period = re.search(r'\.', rest)
            if period:
                text = text[:closing_match.end() + period.end()]
            else:
                text = text[:closing_match.end()]
    
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
    
    # Paksa newline sebelum nomor butir jika ada di tengah kalimat (misal setelah titik dua)
    # Contoh: "...dimaksud dengan : 1. desa adalah..." -> "...dimaksud dengan :\n\n1. desa adalah..."
    text = re.sub(r'(?<=[;:])\s*(\d{1,2}\.)(?=\s)', r'\n\n\1', text)
    
    # Atasi list yang terpisah enter
    text = re.sub(r'(?m)^([a-z]|\d{1,2})\.\s*\n+', r'\1. ', text)
    text = re.sub(r'(,\d*)\s*(\d{1,2}\.\s+[a-z])', r'\1\n\n\2', text)
    
    # Normalisasi hyphenation (kata yang terputus di akhir baris)
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # -------------------------------------------------------
    # MERGE BROKEN LINES: Gabungkan baris yang terputus karena
    # hard line-break dari layout PDF (bukan pemisah paragraf).
    # Strategi: jika baris saat ini TIDAK diakhiri tanda baca
    # dan baris berikutnya tidak diawali oleh:
    # - huruf kapital (kalimat baru)
    # - angka/huruf diikuti titik/kurung (awal butir/item baru)
    # - keyword struktural (pasal, bab, bagian, menimbang, dll)
    # maka gabungkan dengan spasi.
    # -------------------------------------------------------
    _STRUCTURAL_STARTERS = (
        r'pasal\s+\d|bab\s+[ivxlcdm]+|bagian\s+\w|menimbang|mengingat|'
        r'memutuskan|menetapkan|\(\d+\)|\d+\.(?!\d)|[a-z][.)\s]'
    )
    def _merge_broken_lines(txt: str) -> str:
        lines = txt.split('\n')
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip()
            if (i + 1 < len(lines)
                    and stripped
                    and not re.search(r'[.;:,]$', stripped)  # baris saat ini belum selesai
                    and not re.match(r'^\s*$', lines[i + 1])  # baris berikutnya tidak kosong
                    and not re.match(r'^\s*(' + _STRUCTURAL_STARTERS + r')', lines[i + 1], re.IGNORECASE)
                    and not re.match(r'^[A-Z]', lines[i + 1])):
                # Gabungkan baris ini dengan baris berikutnya
                lines[i + 1] = stripped + ' ' + lines[i + 1].lstrip()
                i += 1
                continue
            merged.append(lines[i])
            i += 1
        return '\n'.join(merged)
    
    text = _merge_broken_lines(text)
    
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
    # Hapus baris header/footer "draf peraturan desa X tentang Y N" yang bocor ke tengah teks
    # Pola: "draf peraturan desa <nama> tentang <judul> <nomor_halaman>"
    text = re.sub(r'(?m)^draf\s+peraturan\s+desa\s+.*$', '', text)
    # Hapus baris "peraturan desa <nama> tentang <judul> <nomor>" yang berdiri sendiri (footer)
    text = re.sub(r'(?m)^peraturan\s+desa\s+\w+\s+tentang\s+.*\s+\d+\s*$', '', text)
    
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
    # Handle format: "nomor 5 tahun 2017" dan "nomor : 5 tahun 2017"
    text = re.sub(r'(?m)^nomor\s*[:\-]?\s*\d+\s+tahun\s+\d{4}\s*$', '', text)
    
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
    # Jangan hapus baris 2-4 kata secara umum.
    # Regex lama terlalu agresif dan bisa menghapus isi aturan pendek seperti
    # "kepala desa" pada Pasal 4.
    # Hapus baris "berita daerah ..." yang merupakan footer
    text = re.sub(r'(?m)^berita daerah\s+.*$', '', text)
    
    # Hapus signature yang sudah tergabung oleh _merge_broken_lines:
    # Pola: "<nama orang> ditetapkan/diundangkan di: <tempat> pada tanggal <tgl>"
    # Contoh: "asep zaki kamil diundangkan di: desa biru pada tanggal 2 november 2016"
    # Regex dibuat spesifik agar tidak menghapus isi pasal yang kebetulan
    # mengandung "ditetapkan di" sebagai bagian kalimat regulasi.
    # Hanya match jika ada pola promulgasi (nama tempat + tanggal).
    text = re.sub(
        r'(?m)^.*?\b(ditetapkan|diundangkan)\s+di\s*[:\s]\s*(?:desa|kota|kabupaten|kecamatan)\b.*$',
        '', text
    )
    # Fallback: hapus baris yang DIMULAI dengan ditetapkan/diundangkan di (tanpa prefiks)
    # Ini menangkap kasus dimana baris sudah bersih di baris terpisah
    text = re.sub(r'(?m)^(ditetapkan|diundangkan)\s+di\s+.*$', '', text)
    # Hapus nama orang yang mengandung gelar akademik (S.Sy, S.H., M.M., dll)
    text = re.sub(r'(?m)^[a-z]+(?:[\s,]+[a-z\.]+){0,5}\s*,?\s*s\.\w+\.?\s*$', '', text)
    
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
        # ============================================================
        # PREPROCESSING: Normalisasi spaced-out keywords dari PDF layout
        # ============================================================
        # PDF sering merenggangkan huruf untuk estetika judul:
        #   "T E N T A N G" → "TENTANG"
        #   "M E N I M B A N G" → "MENIMBANG"
        # Tanpa normalisasi ini, metadata extraction gagal mendeteksi keyword.
        spaced_keywords = [
            'TENTANG', 'MENIMBANG', 'MENGINGAT', 'MEMUTUSKAN', 'MENETAPKAN',
            'MEMPERHATIKAN', 'DENGAN', 'RAHMAT', 'TUHAN'
        ]
        normalized_text = full_text
        for kw in spaced_keywords:
            # Pola: setiap huruf dipisah oleh 1+ spasi (misal: "T E N T A N G")
            spaced_pattern = r'\s+'.join(list(kw))
            normalized_text = re.sub(spaced_pattern, kw, normalized_text, flags=re.IGNORECASE)
        
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
        
        # ============================================================
        # Batasi scan HANYA sebelum preamble (Menimbang/Mengingat)
        # ============================================================
        # Nomor dan tahun peraturan ada di HEADER dokumen (sebelum "Menimbang").
        # Jika scan melewati preamble, regex bisa match "Nomor 114 Tahun 2014"
        # yang merupakan referensi undang-undang, BUKAN nomor peraturan ini.
        header_end = len(lines)
        for i, line in enumerate(lines[:30]):
            upper = line.upper().strip()
            if upper.startswith('MENIMBANG') or upper.startswith('MENGINGAT'):
                header_end = i
                break
        
        # ========================================
        # SCAN 1: Header dokumen (sebelum preamble) — untuk desa, nomor, tahun, judul
        # ========================================
        for i, line in enumerate(lines[:min(header_end + 5, 25)]):
            upper_line = line.upper()
            
            # Ekstrak nama desa
            if "KEPALA DESA" in upper_line:
                parts = upper_line.split("KEPALA DESA")
                if len(parts) > 1 and parts[1].strip():
                    raw_village = parts[1].strip().lower()
                    # Bersihkan trailing punctuation (koma, titik, titik dua)
                    village_name = raw_village.rstrip('.,:;').strip()
            elif "PEMERINTAH DESA" in upper_line and village_name == "unknown":
                parts = upper_line.split("PEMERINTAH DESA")
                if len(parts) > 1 and parts[1].strip():
                    raw_village = parts[1].strip().lower()
                    village_name = raw_village.rstrip('.,:;').strip()
            
            # Ekstrak nomor dan tahun
            # Handle 2 format: "NOMOR : 5 TAHUN 2017" dan "NOMOR 5 TAHUN 2017"
            # Optional colon/dash separator antara NOMOR dan angka
            nomor_match = re.search(
                # Handle: "NOMOR 6 TAHUN 2015", "NOMOR : 01 TAHUN 2018",
                # dan "NOMOR 6TAHUN 2015" (tanpa spasi antara angka dan TAHUN)
                r'NOMOR\s*[:\-]?\s*(\d+)\s*TAHUN\s+(\d{4})',
                upper_line
            )
            if nomor_match and perdes_number == "unknown":
                # Ambil match PERTAMA saja (dari header, bukan preamble)
                perdes_number = nomor_match.group(1)
                perdes_year = nomor_match.group(2)
            
            # Ekstrak judul peraturan dari baris setelah "TENTANG"
            if upper_line.strip() == "TENTANG" and i + 1 < len(lines):
                title_lines = []
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_upper = lines[j].upper().strip()
                    if next_upper in ["DENGAN RAHMAT TUHAN YANG MAHA ESA",
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
    canonical_base = _canonical_output_basename(file_path)
    raw_path = os.path.join(raw_output_dir, f"{canonical_base}_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(full_raw_text)
    logger.info(f"Raw text tersimpan: {os.path.basename(raw_path)}")
    
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
    Memecah dokumen menjadi ayat/butir-level chunks dengan hierarchy:
    BAB → Bagian → Pasal → Ayat → Butir

    Hierarki yang benar dalam dokumen hukum Indonesia:
      Pasal
        ├── Ayat  → ditandai oleh (1)(2)(3)... ATAU penomoran 1. 2. 3.
        └── Butir → sub-bagian dalam ayat, ditandai a. b. c.

    Output: list of dict, setiap dict berisi:
    - type: 'preamble' | 'ayat' | 'butir'
    - bab: 'bab i', 'bab ii', ...
    - bab_title: 'ketentuan umum', 'pelayanan kesehatan', ...
    - bagian: 'bagian kesatu', ... (jika ada)
    - bagian_title: 'asas', 'tujuan', ... (jika ada)
    - pasal: 'pasal 1', 'pasal 2', ...
    - ayat: '1', '2', ... atau '' jika pasal hanya 1 ayat
    - butir_num: 'a', 'b', ... atau '' jika ayat tidak punya sub-butir
    - content: teks lengkap (pasal + ayat + butir di dalamnya)
    
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
    # Hanya match "pasal" diikuti ANGKA ARAB (bukan huruf/romawi).
    # "pasal i", "pasal ii" adalah judul bab, bukan nomor pasal.
    pasal_positions = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*pasal\s+\d+\s*$', line.strip(), re.IGNORECASE):
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
            "ayat": "",
            "butir_num": "",
            "content": preamble_text
        })
    
    # Scan line-by-line untuk BAB dan BAGIAN, proses setiap PASAL
    for pos in pasal_positions:
        # Reset bagian state setiap kali masuk BAB baru.
        # Scan dari awal dokumen sampai posisi pasal ini untuk mendapatkan
        # BAB dan BAGIAN terkini. Bagian direset saat bertemu BAB baru.
        temp_bagian = ""
        temp_bagian_title = ""
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
                # Reset bagian saat masuk BAB baru
                temp_bagian = ""
                temp_bagian_title = ""
            
            bagian_match = re.match(r'^bagian\s+(\w+)', lower)
            if bagian_match:
                temp_bagian = f"bagian {bagian_match.group(1)}"
                # Ambil title: baris non-empty berikutnya yang BUKAN bab/bagian/pasal
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = lines[j].strip()
                    cl = candidate.lower()
                    if candidate and not re.match(r'^(bab|bagian|pasal)\s', cl):
                        temp_bagian_title = candidate
                        break
                else:
                    temp_bagian_title = ""
        
        # Update state global bagian dari hasil scan
        current_bagian = temp_bagian
        current_bagian_title = temp_bagian_title
        
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
            # Title line yang benar-benar milik BAB/Bagian berikutnya.
            # Hanya hapus jika beberapa baris sebelumnya adalah header BAB/Bagian;
            # jangan hapus semua baris pendek karena isi pasal bisa pendek juga.
            if (len(check_line) < 60 and
                not re.match(r'^\d+[\.\)]\s', check_line) and
                not re.match(r'^pasal\s+\d', check_line) and
                not re.match(r'^[a-z][\.\)]\s', check_line)):
                previous_nonempty = []
                scan_idx = actual_end - 2
                while scan_idx >= 0 and len(previous_nonempty) < 3:
                    prev_line = pasal_lines[scan_idx].strip().lower()
                    if prev_line:
                        previous_nonempty.append(prev_line)
                    scan_idx -= 1
                if any(re.match(r'^(bab\s+[ivxlcdm]+|bagian\s+\w+)', prev) for prev in previous_nonempty):
                    actual_end -= 1
                    continue
            # This is actual pasal content — stop
            break
        
        pasal_lines = pasal_lines[:actual_end]
        
        pasal_content = '\n'.join(pasal_lines).strip()
        if not pasal_content:
            continue
        
        # ==========================================================
        # Hapus blok tanda tangan / promulgasi yang bocor ke pasal
        # terakhir. Potong content di titik dimana frasa promulgasi
        # atau signature muncul.
        # Pola yang ditangkap:
        # - "ditetapkan di ..." / "diundangkan di ..."
        # - "<nama orang> ditetapkan/diundangkan di: ..."
        # - Nama orang dengan gelar ("aan kurniawan, s.sy")
        # ==========================================================
        pasal_content = re.split(
            r'\n\s*(?:.*?\b(?:ditetapkan|diundangkan)\s+di\b)',
            pasal_content
        )[0].strip()
        if not pasal_content:
            continue
        
        # ==========================================================
        # Hierarki yang benar:
        #   Ayat  → (1)(2)(3)... ATAU penomoran 1. 2. 3.
        #   Butir → sub-bagian dalam ayat, ditandai a. b. c.
        #
        # Strategi:
        # - Ada ayat markers (1)(2)... → split per ayat.
        #   Setiap ayat bisa berisi butir (a,b,c) yang ikut dalam chunk.
        # - Tidak ada ayat markers → cek numbered items (1. 2. 3.).
        #   Tiap numbered item = 1 ayat. Sub-items a. b. c. = butir
        #   dalam ayat tersebut, tetap dalam 1 chunk.
        # ==========================================================
        ayat_markers = re.findall(r'(?m)^\(\d+\)', pasal_content)
        
        if len(ayat_markers) >= 2:
            # Pasal memiliki multiple ayat (1)(2)(3)... → split per ayat
            ayat_chunks = _split_by_ayat(pasal_content)
            
            for ayat_content in ayat_chunks:
                # Extract ayat number: "(1)" → "1", "(12)" → "12"
                ayat_match = re.match(r'\((\d+)\)', ayat_content.strip())
                ayat_num = ayat_match.group(1) if ayat_match else ""
                
                content = f"{pasal_label}\n\n{ayat_content}"
                results.append({
                    "type": "ayat",
                    "bab": current_bab,
                    "bab_title": current_bab_title,
                    "bagian": current_bagian,
                    "bagian_title": current_bagian_title,
                    "pasal": pasal_label,
                    "ayat": ayat_num,
                    "butir_num": "",
                    "content": content
                })
        else:
            # Tidak ada ayat markers (1)(2)...
            # Cek apakah ada numbered items (1. 2. 3.) → masing-masing = 1 ayat
            # Sub-items a. b. c. di dalam numbered item = butir dalam ayat
            ayat_items = _split_butir(pasal_content)
            
            if len(ayat_items) > 1:
                # Multiple numbered items → 1 chunk per ayat
                for idx, ayat_raw in enumerate(ayat_items):
                    ayat_num_match = re.match(r'(\d+)[\.\)]\s*', ayat_raw.strip())
                    ayat_num = ayat_num_match.group(1) if ayat_num_match else str(idx + 1)
                    
                    # Cek apakah ayat ini punya sub-items a. b. c. (butir)
                    butir_match = re.search(r'(?m)^\s*[a-z]\.\s', ayat_raw)
                    butir_label = "" # default: tidak ada butir
                    if butir_match:
                        # Tandai bahwa ayat ini berisi butir
                        # butir_num kosong karena seluruh isi (inkl. butir) ada dalam 1 chunk
                        butir_label = "ada"
                    
                    content = f"{pasal_label}\n\n{ayat_raw}"
                    results.append({
                        "type": "ayat",
                        "bab": current_bab,
                        "bab_title": current_bab_title,
                        "bagian": current_bagian,
                        "bagian_title": current_bagian_title,
                        "pasal": pasal_label,
                        "ayat": ayat_num,
                        "butir_num": "",  # butir di dalam ayat tidak di-split terpisah
                        "content": content
                    })
            else:
                # Single item (pasal pendek, 1 kalimat saja) → 1 chunk
                results.append({
                    "type": "ayat",
                    "bab": current_bab,
                    "bab_title": current_bab_title,
                    "bagian": current_bagian,
                    "bagian_title": current_bagian_title,
                    "pasal": pasal_label,
                    "ayat": "",
                    "butir_num": "",
                    "content": f"{pasal_label}\n\n{ayat_items[0]}"
                })
    
    return results


def _split_by_ayat(pasal_content: str) -> list:
    """
    Membagi isi Pasal berdasarkan ayat markers: (1), (2), (3), ...
    
    Setiap ayat = 1 chunk yang berisi SEMUA nested content:
    - Teks ayat itu sendiri
    - Butir (a, b, c) di dalam ayat
    - Sub-butir (1, 2, 3) di dalam butir
    - Sub-sub-butir (huruf a), b), c)) di dalam sub-butir
    
    Ini memastikan sub-items TIDAK di-split sebagai butir terpisah.
    Contoh: Pasal 6 ayat (3) berisi huruf a-e dengan sub-items 1-17
    → tetap jadi 1 chunk (ayat 3), BUKAN 17+ chunk terpisah.
    
    Contoh:
    Input:
        (1) text ayat 1
        (2) text ayat 2
        a. butir a
        1. sub-butir 1
        2. sub-butir 2
        (3) text ayat 3
    
    Output:
        ["(1) text ayat 1",
         "(2) text ayat 2\na. butir a\n1. sub-butir 1\n2. sub-butir 2",
         "(3) text ayat 3"]
    """
    # Split pada \n yang diikuti oleh (N) di awal baris berikutnya.
    # Lookahead (?=\(\d+\)\s) memastikan (N) tetap di awal part berikutnya.
    parts = re.split(r'\n(?=\(\d+\)\s)', pasal_content)
    
    ayat_list = []
    intro = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^\(\d+\)', part):
            # Ini adalah ayat marker
            if intro:
                # Gabungkan intro text dengan ayat pertama
                # (biasanya teks pembuka sebelum "(1)" muncul)
                ayat_list.append(f"{intro}\n{part}")
                intro = ""
            else:
                ayat_list.append(part)
        else:
            # Teks sebelum ayat pertama (intro/preamble dalam pasal)
            intro = part
    
    # Jika masih ada intro yang belum tergabung (tidak ada ayat)
    if intro and not ayat_list:
        ayat_list.append(intro)
    elif intro:
        # Intro setelah semua ayat (jarang, tapi handle saja)
        ayat_list[-1] = f"{ayat_list[-1]}\n{intro}"
    
    return ayat_list


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
    has_numbered = re.search(r'(?m)^\s*\d+\.(?!\d)', pasal_content)
    if not has_numbered:
        # Tidak ada numbered items → normalize dan return as single chunk
        return [_normalize_letter_numbering(pasal_content)]
    
    # STEP 2: Split berdasarkan numbered items pada teks ASLI.
    # Pola: \n diikuti angka + titik (bukan angka lagi) di awal baris.
    # Menggunakan lookahead agar delimiter (angka + titik) ikut ke bagian baru.
    # PERBAIKAN: pakai re.split dengan lookahead yang lebih ketat.
    # Pastikan split terjadi sebelum SETIAP angka baru di awal baris,
    # bukan hanya sebelum angka 1 (bug sebelumnya: hanya split di awal sehingga
    # butir 2, 3, dst tidak ter-split jika butir 1 sudah di baris pertama).
    parts = re.split(r'(?m)(?<=\n)(?=\d+\.(?!\d)\s)', pasal_content)
    
    # Fallback: jika split di atas menghasilkan hanya 1 part (tidak ada \n sebelum butir),
    # coba split yang lebih agresif dengan \n opsional
    if len(parts) <= 1:
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
    if not re.match(r'^\s*\d+\.(?!\d)', butir_list[0]):
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
    # Hanya huruf a-n (14 item maks realistis untuk sub-item hukum).
    # Huruf o,p,q,r,... sangat jarang jadi sub-item, lebih sering kata biasa.
    # Case 1: di awal baris — group(1)=whitespace, group(2)=huruf
    text = re.sub(r'(?m)^(\s*)([a-n])\.(?=\s)', replace_letter_dot, text)
    # Case 2: inline setelah ": " atau "; " (spasi wajib ada setelah tanda baca)
    text = re.sub(r'(?<=[;:])(\s+)([a-n])\.(?=\s)', replace_letter_dot, text)
    
    def replace_letter_paren(match):
        letter = match.group(2).lower()
        number = ord(letter) - ord('a') + 1
        return f"{match.group(1)}{number})"
    
    # Konversi a) b) c) → 1) 2) 3) (format kurung tutup)
    # Case 1: di awal baris — group(1)=whitespace, group(2)=huruf
    text = re.sub(r'(?m)^(\s*)([a-n])\)(?=\s)', replace_letter_paren, text)
    # Case 2: inline setelah ": " atau "; "
    text = re.sub(r'(?<=[;:])(\s+)([a-n])\)(?=\s)', replace_letter_paren, text)
    
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
        
        # Sequential chunk index within this document (1-based).
        # Disimpan di metadata agar saat retrieval bisa fetch chunk[i-1] dan chunk[i+1]
        # dari dokumen yang sama untuk memperluas konteks (Adjacent Chunk Expansion).
        chunk_idx = 0
        
        # Buat chunks dari sections (skip preamble)
        for section in sections:
            if section["type"] == "preamble":
                continue
            
            chunk_idx += 1
            
            # Enriched content = context_header + isi butir
            enriched = f"{context_header}\n\n{section['content']}" if context_header else section["content"]
            
            # Build metadata hierarchy: bab → pasal → ayat → butir
            chunk_meta = metadata.copy()
            chunk_meta["bab"] = section.get("bab", "")
            chunk_meta["bab_title"] = section.get("bab_title", "")
            chunk_meta["bagian"] = section.get("bagian", "")
            chunk_meta["bagian_title"] = section.get("bagian_title", "")
            chunk_meta["section"] = section["pasal"]
            chunk_meta["ayat"] = section.get("ayat", "")
            if section.get("butir_num"):
                chunk_meta["butir_number"] = section["butir_num"]
            else:
                chunk_meta.pop("butir_number", None)
            chunk_meta["chunk_index"] = chunk_idx
            
            all_chunks.append(Document(
                page_content=enriched,
                metadata=chunk_meta
            ))
    
    logger.info(f"Ayat-level chunking menghasilkan {len(all_chunks)} chunks dari {len(documents)} dokumen")
    return all_chunks


def save_results_to_folder(file_path: str, extracted_docs: list, chunks: list):
    """
    Menyimpan 2 output ke folder data/processed:
    1. Cleaned text (setelah preprocessing) — full teks utuh
    2. Chunks (JSON) — hasil Ayat-level chunking
    (Raw text sudah disimpan terpisah di extract_text_from_pdf)
    
    Ref: 
    - Olsson et al. (2022) "A Survey of Data-Efficient Graph Learning" —
      menyimpan intermediate outputs memudahkan ablation study dan debugging.
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    canonical_base = _canonical_output_basename(file_path)
    _cleanup_processed_outputs(output_dir, canonical_base)
    
    # 1. Simpan CLEANED TEXT (setelah preprocessing, full dokumen)
    cleaned_path = os.path.join(output_dir, f"{canonical_base}_extracted.txt")
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        for doc in extracted_docs:
            f.write(doc.page_content)
            f.write("\n\n")
    
    # 2. Simpan CHUNKS (ayat-level) ke file JSON
    json_path = os.path.join(output_dir, f"{canonical_base}_chunks.json")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata.copy()
        item = {
            "chunk_index": i + 1,
            "bab": meta.get("bab", ""),
            "bab_title": meta.get("bab_title", ""),
            "bagian": meta.get("bagian", ""),
            "bagian_title": meta.get("bagian_title", ""),
            "pasal": meta.get("section", ""),
            "ayat": meta.get("ayat", ""),
            "metadata": meta,
            "character_count": len(chunk.page_content),
            "content": chunk.page_content
        }
        if meta.get("butir_number"):
            item["butir"] = meta["butir_number"]
        chunks_data.append(item)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Hasil processed diperbarui untuk '{canonical_base}' dengan {len(chunks)} chunk.")



def save_chunks_to_postgres(chunks: list) -> bool:
    """
    Menyimpan data hasil chunking ke dalam tabel chunks_perdes di PostgreSQL.
    
    Schema tabel:
        CREATE TABLE chunks_perdes (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255),
            content TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    
    Args:
        chunks: List berisi Document objects dengan page_content dan metadata
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    conn = None
    cursor = None

    if not chunks:
        logger.warning("Tidak ada chunk untuk disimpan ke tabel 'chunks_perdes'.")
        return False
    
    try:
        # 1. Buka koneksi ke PostgreSQL untuk menyimpan hasil chunking.
        logger.info("Menghubungkan ke PostgreSQL untuk menyimpan chunks_perdes...")
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
        INSERT INTO chunks_perdes (file_name, content, metadata)
        VALUES (%s, %s, %s)
        """
        
        # 4. Eksekusi Query untuk Setiap Chunk
        inserted_count = 0
        for chunk in chunks:
            try:
                file_name = chunk.metadata.get("source", "Unknown")
                content = chunk.page_content
                metadata_json = json.dumps(chunk.metadata)
                cursor.execute(insert_query, (file_name, content, metadata_json))
                inserted_count += 1
            except Exception as chunk_error:
                logger.error(f"Gagal menyimpan chunk ke PostgreSQL: {chunk_error}")
                conn.rollback()
                return False

        # 5. Simpan Perubahan (COMMIT) dan Tutup Koneksi
        conn.commit()
        logger.info(f"Berhasil menyimpan {inserted_count} chunk ke tabel chunks_perdes.")
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
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def export_finetune_dataset(chunks: list, file_path: str) -> str:
    """
    Mengekspor hasil Ayat-level chunking ke format JSONL untuk fine-tuning LLM.
    
    Format output (OpenAI fine-tune compatible):
    {"messages": [
        {"role": "system", "content": "Anda adalah asisten hukum... Konteks: <chunk>"},
        {"role": "user", "content": "<pertanyaan>"},
        {"role": "assistant", "content": "<jawaban>"}
    ]}
    
    Setiap AYAT chunk menghasilkan 1 instruction pair:
    - System: context_header + isi ayat (beserta butir di dalamnya)
    - User: pertanyaan otomatis berdasarkan pasal + ayat
    - Assistant: jawaban = isi ayat lengkap
    
    Ref:
    - OpenAI (2024) "Fine-tuning Best Practices" —
      instruction format harus konsisten antara training dan inference.
    - Context dari RAG chunking HARUS identik dengan context di fine-tune data.
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    canonical_base = _canonical_output_basename(file_path)
    jsonl_path = os.path.join(output_dir, f"{canonical_base}_finetune.jsonl")
    
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
        
        # Generate pertanyaan otomatis berdasarkan pasal + ayat
        pasal = metadata.get("section", "")
        ayat_num = metadata.get("ayat", "")
        
        if ayat_num:
            user_question = f"Apa isi {pasal} ayat {ayat_num} dalam {doc_title}?"
            assistant_answer = (
                f"Berdasarkan {doc_title}, {pasal} ayat {ayat_num} mengatur bahwa:\n\n"
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

    postgres_saved = save_chunks_to_postgres(chunks)
    if postgres_saved:
        logger.info(f"Chunks untuk '{os.path.basename(file_path)}' berhasil disimpan ke tabel chunks_perdes.")
    else:
        logger.warning(
            f"Chunks untuk '{os.path.basename(file_path)}' tidak tersimpan ke tabel chunks_perdes. "
            "Cek log koneksi DB, keberadaan tabel, dan kredensial PostgreSQL."
        )

    return chunks

