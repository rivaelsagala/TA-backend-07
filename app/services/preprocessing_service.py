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

        entry_base, _ = os.path.splitext(entry)
        normalized_entry_base = re.sub(r'[^A-Za-z0-9]+', '_', entry_base).strip('_')
        if normalized_entry_base != canonical_base:
            continue

        if entry.endswith(suffixes):
            os.remove(entry_path)


def clean_legal_text(text: str) -> str:
    """
    Membersihkan dan menormalisasi teks hukum dari hasil ekstraksi PDF.
    """
    if not text:
        return ""
    
    # ==============================================
    # TAHAP 1: Normalisasi karakter unicode & kontrol
    # ==============================================
    text = text.replace('\x00', '').replace('\xa0', ' ')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2022', '-')
    text = text.replace('\u00a0', ' ')
    
    # ==============================================
    # TAHAP 1.5: Perbaikan OCR spacing error
    # ==============================================
    _VALID_SINGLE = {'a', 'i', 'o', 'u', 'di', 'ke', 'si', 'se', 'ku', 'mu', 'ya'}
    def _fix_ocr_spacing(txt: str) -> str:
        def _replacer(m):
            char = m.group(1)
            if char.lower() in _VALID_SINGLE:
                return m.group(0)
            return ' ' + char + m.group(2)
        return re.sub(r'(?<=[a-z]) ([a-z]) ([a-z]{2,})', _replacer, txt)
    text = _fix_ocr_spacing(text)

    # ==============================================
    # TAHAP 2: Konversi ke LOWERCASE
    # ==============================================
    text = text.lower()
    
    # ==============================================
    # TAHAP 3: Perbaikan OCR errors - KHUSUS jangan rusak BAB/BAGIAN/PASAL
    # ==============================================
    # Fix spaced-out keywords (tapi HANYA untuk kata-kata non-struktural)
    spaced_keywords = [
        'tentang', 'menimbang', 'mengingat', 'memutuskan', 'menetapkan',
        'memperhatikan', 'dengan', 'rahmat', 'tuhan', 'yang', 'maha', 'esa'
    ]
    for kw in spaced_keywords:
        spaced_pattern = r'\s+'.join(list(kw))
        text = re.sub(spaced_pattern, kw, text, flags=re.IGNORECASE)
    
    # FIX: Perbaiki spaced-out "B A B" → "bab" tapi PERTAHANKAN spasi setelahnya
    # Pattern: "B A B" (spasi antar huruf) → "bab"
    text = re.sub(r'B\s+A\s+B', 'bab', text, flags=re.IGNORECASE)
    # Pattern: "P A S A L" → "pasal"
    text = re.sub(r'P\s+A\s+S\s+A\s+L', 'pasal', text, flags=re.IGNORECASE)
    # Pattern: "B A G I A N" → "bagian"
    text = re.sub(r'B\s+A\s+G\s+I\s+A\s+N', 'bagian', text, flags=re.IGNORECASE)
    
    # Fix "menimbang $ a." menjadi "menimbang : a."
    text = re.sub(r'(menimbang|mengingat|memperhatikan|menetapkan)\s*[$|s]\s*', r'\1 : ', text)
    
    # Fix "bab !!" menjadi "bab ii"
    text = re.sub(r'bab\s+!!', 'bab ii', text)
    
    # Fix karakter aneh
    text = re.sub(r'(?m)^p\s*s\s*', '1. ', text)
    text = re.sub(r'(?m)^/\s*', '7. ', text)

    # Potong setelah kalimat penutup
    closing_match = re.search(
        r'agar\s+setiap\s+orang\s+dapat\s+mengetahuinya',
        text,
        flags=re.IGNORECASE
    )
    if closing_match:
        end_search = re.search(
            r'menempatkannya\s+dalam\s+lembaran\s+desa\.?',
            text[closing_match.start():],
            flags=re.IGNORECASE
        )
        if end_search:
            text = text[:closing_match.start() + end_search.end()]
        else:
            rest = text[closing_match.end():]
            period = re.search(r'\.', rest)
            if period:
                text = text[:closing_match.end() + period.end()]
            else:
                text = text[:closing_match.end()]
    
    # ==============================================
    # TAHAP 4: Penghapusan karakter aneh
    # ==============================================
    text = re.sub(r'[*\^~`{}\[\]<>|]', ' ', text)
    text = re.sub(r'[|_\-\[\]{}><]{2,}', ' ', text)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[©®™℠§¶†‡•…‰′″‱]', ' ', text)
    
    # ==============================================
    # TAHAP 5: Normalisasi whitespace
    # ==============================================
    text = re.sub(r'\n\s*:\s*', ' : ', text)
    text = re.sub(r'\s+:\s+', ' : ', text)
    text = re.sub(r'(?<=[;:])\s*(\d{1,2}\.)(?=\s)', r'\n\n\1', text)
    text = re.sub(r'(?m)^([a-z]|\d{1,2})\.\s*\n+', r'\1. ', text)
    text = re.sub(r'(,\d*)\s*(\d{1,2}\.\s+[a-z])', r'\1\n\n\2', text)
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # MERGE BROKEN LINES - TAPI jangan merge BAB/BAGIAN/PASAL
    _STRUCTURAL_STARTERS = (
        r'pasal\s+\d|bab\s+[ivxlcdm]+|\d+|bagian\s+\w|menimbang|mengingat|'
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
                    and not re.search(r'[.;:,]$', stripped)
                    and not re.match(r'^\s*$', lines[i + 1])
                    and not re.match(r'^\s*(' + _STRUCTURAL_STARTERS + r')', lines[i + 1], re.IGNORECASE)
                    and not re.match(r'^[A-Z]', lines[i + 1])):
                lines[i + 1] = stripped + ' ' + lines[i + 1].lstrip()
                i += 1
                continue
            merged.append(lines[i])
            i += 1
        return '\n'.join(merged)
    
    text = _merge_broken_lines(text)
    
    # Normalisasi whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    text = re.sub(r'(?m)^[ \t]+', '', text)
    text = re.sub(r'(?m)[ \t]+$', '', text)
    
    # ==============================================
    # TAHAP 6: Pembersihan nomor halaman & artifacts
    # ==============================================
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r'(?m)^[a-z]\s*$', '', text)
    text = re.sub(r'(?m)^salinan\s*$', '', text)
    text = re.sub(r'(?m)^lembaran desa\s+.*$', '', text)
    text = re.sub(r'(?m)^draf\s+peraturan\s+desa\s+.*$', '', text)
    text = re.sub(r'(?m)^peraturan\s+desa\s+\w+\s+tentang\s+.*\s+\d+\s*$', '', text)
    
    # ==============================================
    # TAHAP 7: Penghapusan HEADER dan FOOTER
    # ==============================================
    text = re.sub(r'(?m)^pemerintah desa\s+\w+\s*$', '', text)
    text = re.sub(r'(?m)^nomor\s*[:\-]?\s*\d+\s+tahun\s+\d{4}\s*$', '', text)
    
    # ==============================================
    # TAHAP 8: Penghapusan BLOK TANDA TANGAN
    # ==============================================
    text = re.sub(r'(?m)^ttd\s*$', '', text)
    text = re.sub(r'(?m)^(ditetapkan|diundangkan)\s+di\s+.*$', '', text)
    text = re.sub(r'(?m)^pada tanggal\s+.*$', '', text)
    text = re.sub(r'(?m)^kepala desa\s+\w+,?\s*$', '', text)
    text = re.sub(r'(?m)^sekretaris desa\s*$', '', text)
    text = re.sub(r'(?m)^berita daerah\s+.*$', '', text)
    
    text = re.sub(
        r'(?m)^.*?\b(ditetapkan|diundangkan)\s+di\s*[:\s]\s*(?:desa|kota|kabupaten|kecamatan)\b.*$',
        '', text
    )
    text = re.sub(r'(?m)^(ditetapkan|diundangkan)\s+di\s+.*$', '', text)
    text = re.sub(r'(?m)^[a-z]+(?:[\s,]+[a-z\.]+){0,5}\s*,?\s*s\.\w+\.?\s*$', '', text)
    
    # ==============================================
    # TAHAP 9: Penghapusan KONJUNGSI berdiri sendiri
    # ==============================================
    text = re.sub(r'(?m)^(dan|atau|serta|dengan|untuk|dari|ke|pada|oleh)\s*$', '', text)
    
    # Final cleanup
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def extract_perdes_metadata(file_path: str, full_text: str) -> dict:
    """
    Mengekstrak metadata terstruktur dari dokumen Peraturan Desa.
    """
    village_name = "unknown"
    regency_name = "unknown"
    perdes_number = "unknown"
    perdes_year = "unknown"
    perdes_title = "unknown"
    
    try:
        # Normalisasi spaced-out keywords
        spaced_keywords = [
            'TENTANG', 'MENIMBANG', 'MENGINGAT', 'MEMUTUSKAN', 'MENETAPKAN',
            'MEMPERHATIKAN', 'DENGAN', 'RAHMAT', 'TUHAN'
        ]
        normalized_text = full_text
        for kw in spaced_keywords:
            spaced_pattern = r'\s+'.join(list(kw))
            normalized_text = re.sub(spaced_pattern, kw, normalized_text, flags=re.IGNORECASE)
        
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
        
        # Batasi scan sebelum preamble
        header_end = len(lines)
        for i, line in enumerate(lines[:30]):
            upper = line.upper().strip()
            if upper.startswith('MENIMBANG') or upper.startswith('MENGINGAT'):
                header_end = i
                break
        
        # SCAN 1: Header dokumen
        for i, line in enumerate(lines[:min(header_end + 5, 25)]):
            upper_line = line.upper()
            
            if "KEPALA DESA" in upper_line:
                parts = upper_line.split("KEPALA DESA")
                if len(parts) > 1 and parts[1].strip():
                    raw_village = parts[1].strip().lower()
                    village_name = raw_village.rstrip('.,:;').strip()
            elif "PEMERINTAH DESA" in upper_line and village_name == "unknown":
                parts = upper_line.split("PEMERINTAH DESA")
                if len(parts) > 1 and parts[1].strip():
                    raw_village = parts[1].strip().lower()
                    village_name = raw_village.rstrip('.,:;').strip()
            
            nomor_match = re.search(
                r'NOMOR\s*[:\-]?\s*(\d+)\s*TAHUN\s+(\d{4})',
                upper_line
            )
            if nomor_match and perdes_number == "unknown":
                perdes_number = nomor_match.group(1)
                perdes_year = nomor_match.group(2)
            
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
        
        # SCAN 2: Seluruh teks untuk KABUPATEN
        regency_candidates = re.findall(
            r'kabupaten\s+([a-zA-Z]+)',
            full_text, re.IGNORECASE
        )
        if regency_candidates:
            stop_words = {'yang', 'dan', 'atau', 'dari', 'di', 'ke', 'pada', 
                          'dalam', 'dengan', 'untuk', 'oleh', 'ini', 'itu',
                          'nomor', 'tahun', 'republik', 'indonesia', 'negara',
                          'daerah', 'bupati', 'peraturan', 'pemerintah'}
            filtered = [
                w.lower() for w in regency_candidates 
                if w.lower() not in stop_words and len(w) > 2
            ]
            if filtered:
                from collections import Counter
                regency_name = Counter(filtered).most_common(1)[0][0]
        
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
    Mengekstrak teks dari PDF dan menghasilkan SATU Document object.
    """
    try:
        pdf_doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error membuka file: {e}")
        return []

    raw_pages = []
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
    
    full_raw_text = "\n\n".join([text for _, text in raw_pages])
    
    perdes_meta = extract_perdes_metadata(file_path, full_raw_text)
    logger.info(f"Metadata diekstrak: {perdes_meta['document_id']} | {perdes_meta['document_title']}")
    
    clean_text = clean_legal_text(full_raw_text)
    
    if not clean_text:
        logger.error("Teks bersih kosong setelah cleaning.")
        return []
    
    context_header = (
        f"[dokumen: {perdes_meta['document_title']}] "
        f"[desa: {perdes_meta['village_name']}] "
        f"[kabupaten: {perdes_meta['regency_name']}] "
        f"[nomor: {perdes_meta['perdes_number']}/{perdes_meta['perdes_year']}]"
    )
    
    enriched_text = f"{context_header}\n\n{clean_text}"
    
    raw_output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(raw_output_dir, exist_ok=True)
    canonical_base = _canonical_output_basename(file_path)
    raw_path = os.path.join(raw_output_dir, f"{canonical_base}_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(full_raw_text)
    logger.info(f"Raw text tersimpan: {os.path.basename(raw_path)}")
    
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
    
    return [doc]


def _parse_perdes_sections(text: str) -> list:
    """
    State-machine parser untuk dokumen Peraturan Desa.
    Memecah dokumen menjadi ayat/butir-level chunks dengan hierarchy:
    BAB → Bagian → Pasal → Ayat → Butir
    """
    results = []
    lines = text.split('\n')
    
    # ==========================================================
    # STEP 1: Temukan posisi semua BAB, BAGIAN, dan PASAL
    # ==========================================================
    bab_positions = []  # (index, bab_num, title)
    bagian_positions = []  # (index, bagian_name, title)
    pasal_positions = []  # (index, pasal_num)
    
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        
        # DETECT BAB: "bab i", "bab i ketentuan umum", "bab 1", dll.
        bab_match = re.match(r'^bab\s+([ivxlcdm]+|\d+)(?:\s+(.+))?$', line_lower)
        if bab_match:
            bab_num = bab_match.group(1)
            bab_title = bab_match.group(2) or ""
            # Jika title kosong, cari di baris berikutnya
            if not bab_title:
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    candidate_lower = candidate.lower()
                    if re.match(r'^(bab|bagian|pasal)\s', candidate_lower):
                        break
                    bab_title = candidate
                    break
            bab_positions.append((i, bab_num, bab_title))
            continue
        
        # DETECT BAGIAN: "bagian kesatu", "bagian pertama", dll.
        bagian_match = re.match(r'^bagian\s+(\w+)(?:\s+(.+))?$', line_lower)
        if bagian_match:
            bagian_name = bagian_match.group(1)
            bagian_title = bagian_match.group(2) or ""
            if not bagian_title:
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    candidate_lower = candidate.lower()
                    if re.match(r'^(bab|bagian|pasal)\s', candidate_lower):
                        break
                    bagian_title = candidate
                    break
            bagian_positions.append((i, bagian_name, bagian_title))
            continue
        
        # DETECT PASAL: "pasal 1", "pasal 2", dll. (hanya angka arab)
        pasal_match = re.match(r'^pasal\s+(\d+)\s*$', line_lower)
        if pasal_match:
            pasal_positions.append((i, pasal_match.group(1)))
    
    # ==========================================================
    # STEP 2: Bangun hierarki untuk setiap pasal
    # ==========================================================
    for pos, pasal_num in pasal_positions:
        # Cari BAB terakhir sebelum posisi pasal ini
        bab_for_pasal = ""
        bab_title_for_pasal = ""
        for bab_pos, bab_num, bab_title in bab_positions:
            if bab_pos < pos:
                bab_for_pasal = f"bab {bab_num}"
                bab_title_for_pasal = bab_title
            else:
                break
        
        # Cari BAGIAN terakhir sebelum posisi pasal ini
        bagian_for_pasal = ""
        bagian_title_for_pasal = ""
        for bagian_pos, bagian_name, bagian_title in bagian_positions:
            if bagian_pos < pos:
                bagian_for_pasal = f"bagian {bagian_name}"
                bagian_title_for_pasal = bagian_title
            else:
                break
        
        pasal_label = f"pasal {pasal_num}"
        
        # Ambil isi pasal
        next_pasal_pos = None
        for pp in pasal_positions:
            if pp[0] > pos:
                next_pasal_pos = pp[0]
                break
        
        end_pos = next_pasal_pos if next_pasal_pos else len(lines)
        
        content_start = pos + 1
        while content_start < end_pos and not lines[content_start].strip():
            content_start += 1
        
        pasal_lines = lines[content_start:end_pos]
        
        # Hapus trailing structural headers yang bocor
        actual_end = len(pasal_lines)
        while actual_end > 0:
            check_line = pasal_lines[actual_end - 1].strip().lower()
            if not check_line:
                actual_end -= 1
                continue
            if re.match(r'^bab\s+[ivxlcdm]+|\d+', check_line):
                actual_end -= 1
                continue
            if re.match(r'^bagian\s+\w+', check_line):
                actual_end -= 1
                continue
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
                if any(re.match(r'^(bab\s+[ivxlcdm]+|\d+|bagian\s+\w+)', prev) for prev in previous_nonempty):
                    actual_end -= 1
                    continue
            break
        
        pasal_lines = pasal_lines[:actual_end]
        pasal_content = '\n'.join(pasal_lines).strip()
        if not pasal_content:
            continue
        
        # Hapus blok tanda tangan yang bocor
        pasal_content = re.split(
            r'\n\s*(?:.*?\b(?:ditetapkan|diundangkan)\s+di\b)',
            pasal_content
        )[0].strip()
        if not pasal_content:
            continue
        
        # ==========================================================
        # STEP 3: Pecah menjadi ayat/butir
        # ==========================================================
        ayat_markers = re.findall(r'(?m)^\(\d+\)', pasal_content)
        
        if len(ayat_markers) >= 2:
            ayat_chunks = _split_by_ayat(pasal_content)
            
            for ayat_content in ayat_chunks:
                ayat_match = re.match(r'\((\d+)\)', ayat_content.strip())
                ayat_num = ayat_match.group(1) if ayat_match else ""
                
                content = f"{pasal_label}\n\n{ayat_content}"
                results.append({
                    "type": "ayat",
                    "bab": bab_for_pasal,
                    "bab_title": bab_title_for_pasal,
                    "bagian": bagian_for_pasal,
                    "bagian_title": bagian_title_for_pasal,
                    "pasal": pasal_label,
                    "ayat": ayat_num,
                    "butir_num": "",
                    "content": content
                })
        else:
            ayat_items = _split_butir(pasal_content)
            
            if len(ayat_items) > 1:
                for idx, ayat_raw in enumerate(ayat_items):
                    ayat_num_match = re.match(r'(\d+)[\.\)]\s*', ayat_raw.strip())
                    ayat_num = ayat_num_match.group(1) if ayat_num_match else str(idx + 1)
                    
                    content = f"{pasal_label}\n\n{ayat_raw}"
                    results.append({
                        "type": "ayat",
                        "bab": bab_for_pasal,
                        "bab_title": bab_title_for_pasal,
                        "bagian": bagian_for_pasal,
                        "bagian_title": bagian_title_for_pasal,
                        "pasal": pasal_label,
                        "ayat": ayat_num,
                        "butir_num": "",
                        "content": content
                    })
            else:
                results.append({
                    "type": "ayat",
                    "bab": bab_for_pasal,
                    "bab_title": bab_title_for_pasal,
                    "bagian": bagian_for_pasal,
                    "bagian_title": bagian_title_for_pasal,
                    "pasal": pasal_label,
                    "ayat": "",
                    "butir_num": "",
                    "content": f"{pasal_label}\n\n{ayat_items[0]}"
                })
    
    return results


def _split_by_ayat(pasal_content: str) -> list:
    """
    Membagi isi Pasal berdasarkan ayat markers: (1), (2), (3), ...
    """
    parts = re.split(r'\n(?=\(\d+\)\s)', pasal_content)
    
    ayat_list = []
    intro = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^\(\d+\)', part):
            if intro:
                ayat_list.append(f"{intro}\n{part}")
                intro = ""
            else:
                ayat_list.append(part)
        else:
            intro = part
    
    if intro and not ayat_list:
        ayat_list.append(intro)
    elif intro:
        ayat_list[-1] = f"{ayat_list[-1]}\n{intro}"
    
    return ayat_list


def _split_butir(pasal_content: str) -> list:
    """
    Membagi isi Pasal menjadi butir-butir individual.
    """
    has_numbered = re.search(r'(?m)^\s*\d+\.(?!\d)', pasal_content)
    if not has_numbered:
        return [_normalize_letter_numbering(pasal_content)]
    
    parts = re.split(r'(?m)(?<=\n)(?=\d+\.(?!\d)\s)', pasal_content)
    
    if len(parts) <= 1:
        parts = re.split(r'\n(?=\s*\d+\.(?!\d))', pasal_content)
    
    butir_list = []
    for part in parts:
        part = part.strip()
        if part:
            butir_list.append(part)
    
    if not butir_list:
        return [_normalize_letter_numbering(pasal_content)]
    
    if not re.match(r'^\s*\d+\.(?!\d)', butir_list[0]):
        if len(butir_list) > 1:
            butir_list[1] = f"{butir_list[0]}\n\n{butir_list[1]}"
            butir_list.pop(0)
    
    result = []
    for butir in butir_list:
        result.append(_normalize_letter_numbering(butir))
    
    return result


def _normalize_letter_numbering(text: str) -> str:
    """
    Normalisasi penomoran huruf (a, b, c) menjadi angka (1, 2, 3).
    """
    def replace_letter_dot(match):
        letter = match.group(2).lower()
        number = ord(letter) - ord('a') + 1
        return f"{match.group(1)}{number}."
    
    text = re.sub(r'(?m)^(\s*)([a-n])\.(?=\s)', replace_letter_dot, text)
    text = re.sub(r'(?<=[;:])(\s+)([a-n])\.(?=\s)', replace_letter_dot, text)
    
    def replace_letter_paren(match):
        letter = match.group(2).lower()
        number = ord(letter) - ord('a') + 1
        return f"{match.group(1)}{number})"
    
    text = re.sub(r'(?m)^(\s*)([a-n])\)(?=\s)', replace_letter_paren, text)
    text = re.sub(r'(?<=[;:])(\s+)([a-n])\)(?=\s)', replace_letter_paren, text)
    
    return text


def chunk_documents(documents: list):
    """
    Atomic Clause Chunking untuk dokumen Peraturan Desa (Perdes).
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata.copy()
        
        context_header = ""
        content = text
        header_match = re.match(
            r'(\[dokumen:.*?\]\s*\[desa:.*?\]\s*\[kabupaten:.*?\]\s*\[nomor:.*?\])',
            text
        )
        if header_match:
            context_header = header_match.group(1)
            content = text[header_match.end():].strip()
        
        sections = _parse_perdes_sections(content)
        
        chunk_idx = 0
        
        for section in sections:
            if section["type"] == "preamble":
                continue
            
            chunk_idx += 1
            
            enriched = f"{context_header}\n\n{section['content']}" if context_header else section["content"]
            
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
    Menyimpan output ke folder data/processed.
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    canonical_base = _canonical_output_basename(file_path)
    _cleanup_processed_outputs(output_dir, canonical_base)
    
    cleaned_path = os.path.join(output_dir, f"{canonical_base}_extracted.txt")
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        for doc in extracted_docs:
            f.write(doc.page_content)
            f.write("\n\n")
    
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
    """
    conn = None
    cursor = None

    if not chunks:
        logger.warning("Tidak ada chunk untuk disimpan ke tabel 'chunks_perdes'.")
        return False
    
    try:
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

        insert_query = """
        INSERT INTO chunks_perdes (file_name, content, metadata)
        VALUES (%s, %s, %s)
        """
        
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

        conn.commit()
        logger.info(f"Berhasil menyimpan {inserted_count} chunk ke tabel chunks_perdes.")
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Gagal terhubung ke database PostgreSQL: {e}")
        return False
        
    except psycopg2.ProgrammingError as e:
        logger.error(f"❌ Error SQL di PostgreSQL: {e}")
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
    """
    output_dir = os.path.join(os.getcwd(), 'data', 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    canonical_base = _canonical_output_basename(file_path)
    jsonl_path = os.path.join(output_dir, f"{canonical_base}_finetune.jsonl")
    
    finetune_entries = []
    
    for chunk in chunks:
        content = chunk.page_content
        metadata = chunk.metadata
        
        context_header = ""
        butir_content = content
        header_match = re.match(
            r'(\[dokumen:.*?\]\s*\[desa:.*?\]\s*\[kabupaten:.*?\]\s*\[nomor:.*?\])',
            content
        )
        if header_match:
            context_header = header_match.group(1)
            butir_content = content[header_match.end():].strip()
        
        doc_title = metadata.get("title", "Unknown")
        system_content = (
            f"Anda adalah asisten hukum pemerintahan desa. "
            f"Jawab pertanyaan berdasarkan konteks dokumen yang diberikan.\n\n"
            f"KONTEKS DOKUMEN:\n"
            f"{context_header}\n\n{butir_content}"
        )
        
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
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in finetune_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Fine-tune dataset tersimpan: {jsonl_path} ({len(finetune_entries)} entries)")
    return jsonl_path


def extract_and_chunk_pdf(file_path: str, save_to_db: bool = True):
    """
    Fungsi utama: ekstrak teks dari PDF dan chunk menjadi ayat/butir.
    
    Args:
        file_path: Path ke file PDF
        save_to_db: Jika True, simpan ke PostgreSQL dan export finetune dataset
    
    Returns:
        List of Document objects (chunks)
    """
    documents = extract_text_from_pdf(file_path)
    chunks = chunk_documents(documents)
    
    # Simpan ke folder lokal (txt & json)
    save_results_to_folder(file_path, documents, chunks)
    
    # Simpan ke DB dan finetune dataset jika diminta
    if save_to_db:
        export_finetune_dataset(chunks, file_path)
        postgres_saved = save_chunks_to_postgres(chunks)
        if postgres_saved:
            logger.info(f"Chunks untuk '{os.path.basename(file_path)}' berhasil disimpan ke tabel chunks_perdes.")
        else:
            logger.warning(
                f"Chunks untuk '{os.path.basename(file_path)}' tidak tersimpan ke tabel chunks_perdes."
            )
    else:
        logger.info(f"Preview Mode: Ekstraksi '{os.path.basename(file_path)}' selesai. Melewati penyimpanan ke DB.")

    return chunks