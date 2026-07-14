import os
import fitz  # PyMuPDF
import pytesseract
import io
import re
import json
import psycopg2
from langchain_core.documents import Document
from PIL import Image
from dotenv import load_dotenv
from loguru import logger
import platform

load_dotenv()

_tesseract_default = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if platform.system() == 'Windows'
    else '/usr/bin/tesseract'
)
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', _tesseract_default)

def _canonical_output_basename(file_path: str) -> str:
    original_name = os.path.splitext(os.path.basename(file_path))[0]
    normalized = re.sub(r'[^A-Za-z0-9]+', '_', original_name).strip('_')
    return normalized or "document"

def _cleanup_processed_outputs(output_dir: str, canonical_base: str) -> None:
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

def clean_legal_text(text: str) -> str:
    if not text:
        return ""
    
    # TAHAP 1: Normalisasi karakter unicode & kontrol
    text = text.replace('\x00', '').replace('\xa0', ' ')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2022', '-')
    text = text.replace('\u00a0', ' ')
    
    # TAHAP 1.5: Perbaikan OCR spacing error & Typo
    _VALID_SINGLE = {'a', 'i', 'o', 'u', 'di', 'ke', 'si', 'se', 'ku', 'mu', 'ya'}
    def _fix_ocr_spacing(txt: str) -> str:
        def _replacer(m):
            char = m.group(1)
            if char.lower() in _VALID_SINGLE:
                return m.group(0)
            return ' ' + char + m.group(2)
        return re.sub(r'(?<=[a-z]) ([a-z]) ([a-z]{2,})', _replacer, txt)
    
    text = _fix_ocr_spacing(text)
    
    text = re.sub(r'(?m)(^|\s)([a-z]|\d{1,2})\.\s*\n\s*', r'\1\2. ', text)
    
    text = re.sub(r'\bmenginat\b', 'mengingat', text, flags=re.IGNORECASE)

    # TAHAP 2: Konversi ke LOWERCASE
    text = text.lower()
    
    # TAHAP 3: Perbaikan OCR errors khusus dokumen hukum
    spaced_keywords = [
        'TENTANG', 'MENIMBANG', 'MENGINGAT', 'MEMUTUSKAN', 'MENETAPKAN',
        'MEMPERHATIKAN', 'PASAL', 'BAB', 'BAGIAN'
    ]
    for kw in spaced_keywords:
        spaced_pattern = r'\s+'.join(list(kw))
        text = re.sub(spaced_pattern, kw.lower(), text, flags=re.IGNORECASE)
    
    text = re.sub(r'(menimbang|mengingat|memperhatikan|menetapkan)\s*[$|s]\s*', r'\1 : ', text, flags=re.IGNORECASE)
    text = re.sub(r'bab\s+!!', 'bab ii', text)
    text = re.sub(r'(?m)^pasal\s*\n\s*1\.\s*tarip', 'pasal 6\n1. tarip', text)
    text = re.sub(r'(?m)^p\s*s\s*', '1. ', text)
    text = re.sub(r'(?m)^/\s*', '7. ', text)

    closing_match = re.search(
        r'agar\s+(?:setiap|semua)\s+orang\s+(?:dapat\s+)?mengetahui(?:nya)?',
        text,
        flags=re.IGNORECASE
    )
    if closing_match:
        end_search = re.search(
            r'(?:m|p)enempatkannya\s+dalam\s+lembaran\s+(?:desa|daerah|kabupaten|kota)[^;.]*[;.]?',
            text[closing_match.start():],
            flags=re.IGNORECASE
        )
        if end_search:
            text = text[:closing_match.start() + end_search.end()]
        else:
            rest = text[closing_match.end():]
            period = re.search(r'[;.]', rest)
            if period:
                text = text[:closing_match.end() + period.end()]
            else:
                text = text[:closing_match.end()]
    
    # TAHAP 4: Penghapusan karakter noise
    text = re.sub(r'[*\^~`{}\[\]<>|]', ' ', text)
    text = re.sub(r'[|_\-\[\]{}><]{2,}', ' ', text)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[©®™℠§¶†‡•…‰′″‱]', ' ', text)
    
    # TAHAP 5: Normalisasi whitespace
    text = re.sub(r'\n\s*:\s*', ' : ', text)
    text = re.sub(r'\s+:\s+', ' : ', text)
    text = re.sub(r'(?<=[;:])\s*(\d{1,2}\.)(?=\s)', r'\n\n\1', text)
    text = re.sub(r'(?m)^([a-z]|\d{1,2})\.\s*\n+', r'\1. ', text)
    text = re.sub(r'(,\d*)\s*(\d{1,2}\.\s+[a-z])', r'\1\n\n\2', text)
    text = re.sub(r'-\s*\n\s*', '', text)
    
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
    
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    text = re.sub(r'(?m)^[ \t]+', '', text)
    text = re.sub(r'(?m)[ \t]+$', '', text)
    
    # TAHAP 6 & 7: Pembersihan artifacts PDF & Header/Footer
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r'(?m)^[a-z]\s*$', '', text)
    text = re.sub(r'(?m)^salinan\s*$', '', text)
    text = re.sub(r'(?m)^lembaran desa\s+.*$', '', text)
    text = re.sub(r'(?m)^draf\s+peraturan\s+desa\s+.*$', '', text)
    text = re.sub(r'(?m)^peraturan\s+desa\s+\w+\s+tentang\s+.*\s+\d+\s*$', '', text)
    text = re.sub(r'(?m)^pemerintah desa\s+\w+\s*$', '', text)
    text = re.sub(r'(?m)^nomor\s*[:\-]?\s*\d+\s+tahun\s+\d{4}\s*$', '', text)
    
    # TAHAP 8: Penghapusan Signature Block Umum
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
    text = re.sub(r'(?m)^[a-z]+(?:[\s,]+[a-z\.]+){0,5}\s*,?\s*s\.\w+\.?\s*$', '', text)
    
    # TAHAP 9: Penghapusan Konjungsi Tunggal
    text = re.sub(r'(?m)^(dan|atau|serta|dengan|untuk|dari|ke|pada|oleh)\s*$', '', text)
    
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
    
def extract_perdes_metadata(file_path: str, full_text: str) -> dict:
    village_name = "unknown"
    regency_name = "unknown"
    perdes_number = "unknown"
    perdes_year = "unknown"
    perdes_title = "unknown"
    
    try:
        spaced_keywords = [
            'TENTANG', 'MENIMBANG', 'MENGINGAT', 'MEMUTUSKAN', 'MENETAPKAN',
            'MEMPERHATIKAN', 'DENGAN', 'RAHMAT', 'TUHAN'
        ]
        normalized_text = full_text
        for kw in spaced_keywords:
            spaced_pattern = r'\s+'.join(list(kw))
            normalized_text = re.sub(spaced_pattern, kw, normalized_text, flags=re.IGNORECASE)
        
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
        
        header_end = len(lines)
        for i, line in enumerate(lines[:30]):
            upper = line.upper().strip()
            if upper.startswith('MENIMBANG') or upper.startswith('MENGINGAT'):
                header_end = i
                break
        
        header_lines = lines[:min(header_end + 5, 25)]
        

        header_block = " ".join(header_lines).upper()
        
        # 1. Ekstrak nama desa (Lebih tangguh terhadap akhiran NOMOR, NOMOR., atau NO)
        desa_match = re.search(r'PERATURAN\s+DESA\s+(.*?)\s+(?:NOMOR|NO\b)', header_block)
        if desa_match:
            potensi_desa = desa_match.group(1).strip().lower()
            # Hapus karakter baca/titik/koma jika OCR tidak sengaja menyatukannya
            potensi_desa = re.sub(r'[\.\:\,\-\;]$', '', potensi_desa).strip()
            if len(potensi_desa) < 50:
                village_name = potensi_desa
                
        # 2. Fallback "KEPALA DESA" / "PEMERINTAH DESA"
        if village_name == "unknown":
            fallback_match = re.search(r'(?:KEPALA|PEMERINTAH)\s+DESA\s+(.*?)(?:\s+KECAMATAN|\s+KABUPATEN|\s+PERATURAN|\s+NOMOR|\s+NO\b|$)', header_block)
            if fallback_match:
                potensi_desa = fallback_match.group(1).strip().lower().rstrip('.,:;')
                if potensi_desa and len(potensi_desa) < 50:
                    village_name = potensi_desa

        # 3. [PERBAIKAN UTAMA] Ekstrak Nomor dan Tahun 
        nomor_match = re.search(r'(?:NOMOR|NO)\s*[\.\:\-,]?\s*(\d+)\s*TAHUN\s*[\.\:\-,]?\s*(\d{4})', header_block)
        if nomor_match:
            perdes_number = nomor_match.group(1)
            perdes_year = nomor_match.group(2)

        # 4. [PERBAIKAN FITUR] Ekstrak Judul menggunakan RegEx pada header_block
        tentang_match = re.search(r'TENTANG\s+(.*?)(?:\s+DENGAN\s+RAHMAT|\s+KEPALA\s+DESA|\s+MENIMBANG)', header_block)
        if tentang_match:
            potensi_judul = tentang_match.group(1).strip().lower()
            if len(potensi_judul) > 5:
                perdes_title = potensi_judul
        else:
            for i, line in enumerate(header_lines):
                upper_line = line.upper()
                if upper_line.strip() == "TENTANG" and i + 1 < len(lines):
                    title_lines = []
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_upper = lines[j].upper().strip()
                        if next_upper in ["DENGAN RAHMAT TUHAN YANG MAHA ESA", "KEPALA DESA", ""]:
                            break
                        title_lines.append(lines[j].strip())
                    if title_lines:
                        perdes_title = " ".join(title_lines).lower()
                        
        # 5. Ekstrak Kabupaten / Kota (Sama dengan aslinya)
        regency_candidates = re.findall(r'kabupaten\s+([a-zA-Z]+)', full_text, re.IGNORECASE)
        if regency_candidates:
            stop_words = {'yang', 'dan', 'atau', 'dari', 'di', 'ke', 'pada', 
                          'dalam', 'dengan', 'untuk', 'oleh', 'ini', 'itu',
                          'nomor', 'tahun', 'republik', 'indonesia', 'negara',
                          'daerah', 'bupati', 'peraturan', 'pemerintah'}
            filtered = [w.lower() for w in regency_candidates if w.lower() not in stop_words and len(w) > 2]
            if filtered:
                from collections import Counter
                regency_name = Counter(filtered).most_common(1)[0][0]
        
        if regency_name == "unknown":
            kota_candidates = re.findall(r'kota\s+([a-zA-Z]+)', full_text, re.IGNORECASE)
            if kota_candidates:
                stop_words = {'yang', 'dan', 'atau', 'dari', 'di', 'ke', 'pada',
                              'dalam', 'dengan', 'untuk', 'oleh', 'ini', 'itu'}
                filtered = [w.lower() for w in kota_candidates if w.lower() not in stop_words and len(w) > 2]
                if filtered:
                    from collections import Counter
                    regency_name = "kota " + Counter(filtered).most_common(1)[0][0]
        
    except Exception as e:
        logger.warning(f"Gagal mengekstrak metadata perdes: {e}")
    
    document_id = f"perdes_dis{village_name}_{perdes_number}_{perdes_year}"
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
    try:
        pdf_doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error membuka file: {e}")
        return []

    raw_pages = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text("text", sort=True).strip()
        
        if len(text) < 50:
            logger.info(f"Halaman {page_num + 1} minim teks. Mencoba ekstraksi dengan OCR...")
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                ocr_text = pytesseract.image_to_string(img, lang="ind")
                if ocr_text.strip():
                    text = ocr_text.strip()
            except Exception as e:
                logger.warning(f"OCR gagal di halaman {page_num + 1}: {e}")

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
    
    raw_output_dir = os.path.join(os.getcwd(), 'data', 'processed', "data")
    os.makedirs(raw_output_dir, exist_ok=True)
    canonical_base = _canonical_output_basename(file_path)
    raw_path = os.path.join(raw_output_dir, f"{canonical_base}_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(full_raw_text)
    
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
    results = []
    lines = text.split('\n')
    
    current_bab = ""
    current_bab_title = ""
    current_bagian = ""
    current_bagian_title = ""
    
    pasal_positions = []
    is_content_started = False
    
    for i, line in enumerate(lines):
        lower_line = line.strip().lower()
        
        if not is_content_started:
            if re.match(r'^(menetapkan|memutuskan|bab\s+i|pasal\s+1)\b', lower_line):
                is_content_started = True
                
        if is_content_started:
            if re.match(r'^\s*pasal\s+\d+(?:[\.\:\s]+.*)?$', lower_line):
                if not re.search(r'\b(sampai\s+dengan|sebagaimana|tentang)\b', lower_line):
                    pasal_positions.append(i)

    if not pasal_positions:
        return results
    
    preamble_text = '\n'.join(lines[:pasal_positions[0]]).strip()
    if preamble_text:
        results.append({
            "type": "preamble",
            "bab": "", "bab_title": "", "bagian": "", "bagian_title": "",
            "pasal": "pembuka", "ayat": "", "butir_num": "",
            "content": preamble_text
        })
    
    for pos in pasal_positions:
        temp_bagian = ""
        temp_bagian_title = ""
        for i in range(pos):
            line = lines[i].strip()
            lower = line.lower()
            
            bab_match = re.match(r'^\s*bab\s+([ivxlcdm]+|\d+)(?:[\.\:\s]+(.*))?$', lower)
            if bab_match:
                current_bab = f"bab {bab_match.group(1)}"
                if bab_match.group(2) and bab_match.group(2).strip():
                    current_bab_title = bab_match.group(2).strip()
                else:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        candidate = lines[j].strip()
                        cl = candidate.lower()
                        if candidate and not re.match(r'^(bab|bagian|pasal)\s', cl):
                            current_bab_title = candidate
                            break
                    else:
                        current_bab_title = ""
                temp_bagian = ""
                temp_bagian_title = ""
            
            bagian_match = re.match(r'^\s*bagian\s+(ke\s*[a-z]+|[a-z]+|\d+)(?:[\.\:\s]+(.*))?$', lower)
            if bagian_match:
                norm_bagian = bagian_match.group(1).replace(' ', '')
                temp_bagian = f"bagian {norm_bagian}"
                if bagian_match.group(2) and bagian_match.group(2).strip():
                    temp_bagian_title = bagian_match.group(2).strip()
                else:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        candidate = lines[j].strip()
                        cl = candidate.lower()
                        if candidate and not re.match(r'^(bab|bagian|pasal)\s', cl):
                            temp_bagian_title = candidate
                            break
                    else:
                        temp_bagian_title = ""
        
        current_bagian = temp_bagian
        current_bagian_title = temp_bagian_title
        
        pasal_line = lines[pos].strip().lower()
        pasal_num_match = re.match(r'^pasal\s+(\d+)(?:[\.\:\s]+(.*))?$', pasal_line)
        if not pasal_num_match:
            continue
        pasal_num = pasal_num_match.group(1)
        pasal_label = f"pasal {pasal_num}"
        
        pasal_lines = []
        if pasal_num_match.group(2) and pasal_num_match.group(2).strip():
            pasal_lines.append(pasal_num_match.group(2).strip())
        
        next_pasal_pos = None
        for pp in pasal_positions:
            if pp > pos:
                next_pasal_pos = pp
                break
        
        end_pos = next_pasal_pos if next_pasal_pos else len(lines)
        
        content_start = pos + 1
        while content_start < end_pos and not lines[content_start].strip():
            content_start += 1
        
        pasal_lines.extend(lines[content_start:end_pos])
        
        actual_end = len(pasal_lines)
        while actual_end > 0:
            check_line = pasal_lines[actual_end - 1].strip().lower()
            if not check_line:
                actual_end -= 1
                continue
            if re.match(r'^bab\s+[ivxlcdm]+', check_line):
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
                if any(re.match(r'^(bab\s+[ivxlcdm]+|bagian\s+\w+)', prev) for prev in previous_nonempty):
                    actual_end -= 1
                    continue
            break
        
        pasal_lines = pasal_lines[:actual_end]
        pasal_content = '\n'.join(pasal_lines).strip()
        if not pasal_content:
            continue
        
        pasal_content = re.split(
            r'\n\s*(?:ditetapkan|diundangkan)\s+di[\s:]',
            pasal_content, flags=re.IGNORECASE
        )[0].strip()
        if not pasal_content:
            continue
        
        ayat_markers = re.findall(r'(?m)^\s*\(\d+\)', pasal_content)
        
        if len(ayat_markers) >= 2:
            ayat_chunks = _split_by_ayat(pasal_content)
            for ayat_content in ayat_chunks:
                ayat_match = re.search(r'\((\d+)\)', ayat_content.strip())
                ayat_num = ayat_match.group(1) if ayat_match else ""
                
                content = f"{pasal_label}\n\n{ayat_content}"
                results.append({
                    "type": "ayat", "bab": current_bab, "bab_title": current_bab_title,
                    "bagian": current_bagian, "bagian_title": current_bagian_title,
                    "pasal": pasal_label, "ayat": ayat_num, "butir_num": "",
                    "content": content
                })
        else:
            ayat_items = _split_butir(pasal_content)
            if len(ayat_items) > 1:
                for idx, ayat_raw in enumerate(ayat_items):
                    ayat_num_match = re.search(r'(\d+)[\.\)]\s*', ayat_raw.strip())
                    ayat_num = ayat_num_match.group(1) if ayat_num_match else str(idx + 1)
                    
                    content = f"{pasal_label}\n\n{ayat_raw}"
                    results.append({
                        "type": "ayat", "bab": current_bab, "bab_title": current_bab_title,
                        "bagian": current_bagian, "bagian_title": current_bagian_title,
                        "pasal": pasal_label, "ayat": ayat_num, "butir_num": "",
                        "content": content
                    })
            else:
                results.append({
                    "type": "ayat", "bab": current_bab, "bab_title": current_bab_title,
                    "bagian": current_bagian, "bagian_title": current_bagian_title,
                    "pasal": pasal_label, "ayat": "", "butir_num": "",
                    "content": f"{pasal_label}\n\n{ayat_items[0]}"
                })
    return results

def _split_by_ayat(pasal_content: str) -> list:
    parts = re.split(r'\n(?=\s*\(\d+\))', pasal_content)
    ayat_list = []
    intro = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^\s*\(\d+\)', part):
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
    has_numbered = re.search(r'(?m)^\s*\d+\.(?!\d)', pasal_content)
    if not has_numbered:
        return [_normalize_letter_numbering(pasal_content)]
    
    parts = re.split(r'(?m)(?<=\n)(?=\s*\d+\.(?!\d))', pasal_content)
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
            
            if section.get("bagian"):
                chunk_meta["bagian"] = section.get("bagian")
            if section.get("bagian_title"):
                chunk_meta["bagian_title"] = section.get("bagian_title")
                
            chunk_meta["section"] = section["pasal"]
            chunk_meta["ayat"] = section.get("ayat", "")
            if section.get("butir_num"):
                chunk_meta["butir_number"] = section["butir_num"]
            else:
                chunk_meta.pop("butir_number", None)
            chunk_meta["chunk_index"] = chunk_idx
            
            all_chunks.append(Document(page_content=enriched, metadata=chunk_meta))
    return all_chunks

def save_results_to_folder(file_path: str, extracted_docs: list, chunks: list):
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
            "pasal": meta.get("section", ""),
            "ayat": meta.get("ayat", ""),
            "metadata": meta,
            "character_count": len(chunk.page_content),
            "content": chunk.page_content
        }
        
        if meta.get("bagian"):
            item["bagian"] = meta.get("bagian")
        if meta.get("bagian_title"):
            item["bagian_title"] = meta.get("bagian_title")
            
        if meta.get("butir_number"):
            item["butir"] = meta["butir_number"]
        chunks_data.append(item)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)

def save_chunks_to_postgres(chunks: list) -> bool:
    conn = None
    cursor = None
    if not chunks:
        return False
    
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT", "")),
            database=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", "")
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chunks_perdes'
            );
        """)
        if not cursor.fetchone()[0]:
            return False

        insert_query = """
        INSERT INTO chunks_perdes (file_name, content, metadata)
        VALUES (%s, %s, %s)
        """
        
        for chunk in chunks:
            file_name = chunk.metadata.get("source", "Unknown")
            content = chunk.page_content
            metadata_json = json.dumps(chunk.metadata)
            cursor.execute(insert_query, (file_name, content, metadata_json))

        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error menyimpan ke PostgreSQL: {e}")
        return False
    finally:
        if cursor is not None: cursor.close()
        if conn is not None: conn.close()


def extract_and_chunk_pdf(file_path: str, save_to_db: bool = True):
    documents = extract_text_from_pdf(file_path)
    chunks = chunk_documents(documents)
    
    save_results_to_folder(file_path, documents, chunks)
    
    if save_to_db:
        postgres_saved = save_chunks_to_postgres(chunks)
        if postgres_saved:
            logger.info(f"Chunks untuk '{os.path.basename(file_path)}' berhasil disimpan ke PostgreSQL.")
        else:
            logger.warning(f"Chunks untuk '{os.path.basename(file_path)}' gagal tersimpan ke PostgreSQL.")
    else:
        logger.info(f"PREVIEW MODE: Ekstraksi '{os.path.basename(file_path)}' selesai. Melewati DB/Finetune.")

    return chunks