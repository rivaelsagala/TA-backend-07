"""
RAFT Dataset Generator v2 — Peraturan Desa
===========================================
Ditulis ulang dari nol dengan prinsip kualitas tinggi:

1. Oracle filtering — skip chunk yang tidak substantif
2. Pertanyaan spesifik — tidak bisa dijawab tanpa dokumen ini
3. Distractor berbasis kesamaan topik (TF-IDF keyword overlap)
4. Thought process reasoning nyata, bukan template
5. Gaya completion di-mapping ke tipe pertanyaan
6. Oracle-absent completion yang menjelaskan ketiadaan jawaban secara substantif
"""

import os, sys, json, time, random, re, math, shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
HF_BASE_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL  = "openai/gpt-4o-mini"
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
DATASET_DIR      = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Parameter generation
MAX_TOKENS        = 1024
REQUEST_DELAY     = 1.5
NUM_DISTRACTORS   = 4
ORACLE_PRESENT_RATIO = 0.80   # 80% oracle hadir, 20% absent

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI TIPE PERTANYAAN
# weight → probabilitas terpilih
# compatible_styles → gaya jawaban yang cocok untuk tipe ini
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "faktual": {
        "weight": 0.20,
        "prompt": (
            "Buat pertanyaan FAKTUAL yang menanyakan angka, tanggal, atau fakta spesifik "
            "yang HANYA ada di dokumen ini. Wajib sebut nama peraturan dan nomor pasal dalam pertanyaan. "
            "Contoh: 'Menurut Pasal 7 Peraturan Desa X No. 1 Tahun 2018, berapa hari tenggat waktu pelaporan?'"
        ),
        "compatible_styles": ["langsung", "formal_hukum"],
    },
    "definisional": {
        "weight": 0.15,
        "prompt": (
            "Buat pertanyaan DEFINISIONAL yang menanyakan definisi istilah teknis yang didefinisikan "
            "dalam dokumen ini. Sertakan nama peraturan dalam pertanyaan. "
            "Contoh: 'Apa yang dimaksud dengan [istilah] menurut Pasal 1 Peraturan Desa X No. 1 Tahun 2018?'"
        ),
        "compatible_styles": ["penjelasan", "formal_hukum", "terstruktur"],
    },
    "prosedural": {
        "weight": 0.15,
        "prompt": (
            "Buat pertanyaan PROSEDURAL tentang langkah-langkah atau tata cara yang diatur dalam dokumen. "
            "Pertanyaan harus menyebut konteks spesifik dari pasal yang dibahas. "
            "Contoh: 'Bagaimana tata cara pengisian anggota BPD antar waktu menurut Pasal 19?'"
        ),
        "compatible_styles": ["terstruktur", "penjelasan"],
    },
    "kondisional": {
        "weight": 0.15,
        "prompt": (
            "Buat pertanyaan KONDISIONAL tentang konsekuensi, syarat, atau kondisi tertentu "
            "yang diatur di dokumen. Harus spesifik ke pasal yang bersangkutan. "
            "Contoh: 'Apa yang terjadi jika anggota BPD tidak hadir 3 kali berturut-turut menurut Pasal 14?'"
        ),
        "compatible_styles": ["langsung", "penjelasan", "formal_hukum"],
    },
    "enumeratif": {
        "weight": 0.15,
        "prompt": (
            "Buat pertanyaan ENUMERATIF yang menanyakan daftar lengkap sesuatu yang tercantum di dokumen. "
            "Harus merujuk pasal yang spesifik. "
            "Contoh: 'Sebutkan semua kewenangan BPD yang diatur dalam Pasal 5 Peraturan Desa X Tahun 2018!'"
        ),
        "compatible_styles": ["terstruktur", "langsung"],
    },
    "komparatif": {
        "weight": 0.10,
        "prompt": (
            "Buat pertanyaan KOMPARATIF yang membandingkan dua ketentuan atau dua hal dalam dokumen. "
            "Kedua hal yang dibandingkan harus ada dalam dokumen yang diberikan. "
            "Contoh: 'Apa perbedaan tugas kepala desa dan BPD menurut peraturan ini?'"
        ),
        "compatible_styles": ["terstruktur", "penjelasan"],
    },
    "interpretatif": {
        "weight": 0.10,
        "prompt": (
            "Buat pertanyaan INTERPRETATIF tentang tujuan, alasan, atau makna di balik suatu ketentuan. "
            "Jawaban harus bisa disimpulkan dari teks dokumen, bukan opini. "
            "Contoh: 'Mengapa Pasal 12 menetapkan batas waktu 7 hari untuk penyampaian hasil pemilihan?'"
        ),
        "compatible_styles": ["percakapan", "penjelasan"],
    },
}

# Gaya jawaban — definisi penuh untuk prompt LLM
COMPLETION_STYLES = {
    "langsung":     "Jawab langsung dan padat dalam 2-3 kalimat. Fokus pada inti jawaban, sertakan angka/fakta spesifik.",
    "penjelasan":   "Berikan jawaban lengkap dengan konteks: apa ketentuannya, mengapa ada, apa implikasinya. Minimal 3-4 kalimat.",
    "terstruktur":  "Mulai dengan pernyataan utama satu kalimat, lalu rinci poin-poin penting secara berurutan menggunakan angka atau huruf.",
    "percakapan":   "Jelaskan seolah berbicara kepada warga desa awam. Gunakan bahasa mudah dipahami tapi tetap akurat secara hukum. Minimal 3 kalimat.",
    "formal_hukum": "Gunakan bahasa hukum formal. Kutip pasal dan ayat secara presisi. Struktur seperti penjelasan resmi pemerintah.",
    "ringkasan":    "Mulai dengan satu kalimat ringkasan, lalu uraikan 2-3 aspek penting dari ketentuan tersebut.",
}


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    retries: int = 3,
) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GENERATOR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
    }
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            wait = (attempt + 1) * (10 if r.status_code == 503 else 5)
            print(f"  HTTP {r.status_code}, tunggu {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & FILTER CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_chunks(processed_dir: Path) -> Dict[str, List[Dict]]:
    all_chunks = {}
    for fpath in sorted(processed_dir.glob("*_chunks.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
        all_chunks[doc_id] = chunks
        print(f"  {len(chunks):>4} chunks ← {fpath.name}")
    return all_chunks


def load_single_chunk_file(filepath: Path) -> Dict[str, List[Dict]]:
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    doc_id = chunks[0]["metadata"]["document_id"] if chunks else filepath.stem
    print(f"  {len(chunks):>4} chunks ← {filepath.name}")
    return {doc_id: chunks}


def clean_content(text: str) -> str:
    """Bersihkan metadata header dari konten chunk."""
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()


def is_substantive(content: str, min_chars: int = 250) -> bool:
    """
    Chunk layak jadi oracle hanya jika:
    - Panjang konten >= min_chars
    - Bukan sekadar header pasal tanpa isi
    - Mengandung minimal 3 kalimat / klausa
    """
    c = clean_content(content)
    if len(c) < min_chars:
        return False
    # Hitung kalimat sederhana (split by . ? !)
    sentences = [s.strip() for s in re.split(r'[.!?]', c) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PEMILIHAN DISTRACTOR BERBASIS KEYWORD OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini",
    "itu", "atau", "juga", "tidak", "dalam", "adalah", "oleh", "akan",
    "sebagai", "telah", "dapat", "tersebut", "lebih", "serta", "atas",
    "bahwa", "sesuai", "berdasarkan", "nomor", "pasal", "ayat", "tahun",
    "tentang", "desa", "peraturan", "ketentuan", "hal",
}

def keyword_overlap_score(text_a: str, text_b: str) -> float:
    """Hitung Jaccard similarity kata-kata non-stopword antara dua teks."""
    def tokenize(t):
        words = re.findall(r'\b[a-z]{3,}\b', t.lower())
        return set(w for w in words if w not in STOPWORDS_ID)
    a, b = tokenize(text_a), tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_distractors(
    oracle_chunk: Dict,
    all_doc_chunks: Dict[str, List[Dict]],
    n: int = NUM_DISTRACTORS,
) -> List[Dict]:
    """
    Pilih distractor yang topiknya MIRIP dengan oracle (keyword overlap tinggi)
    tapi dari dokumen BERBEDA. Ini membuat task lebih menantang dan mendidik.
    """
    oracle_doc_id = oracle_chunk["metadata"]["document_id"]
    oracle_text   = clean_content(oracle_chunk["content"])

    # Kumpulkan semua kandidat dari dokumen lain
    candidates = []
    for doc_id, chunks in all_doc_chunks.items():
        if doc_id == oracle_doc_id:
            continue
        for c in chunks:
            score = keyword_overlap_score(oracle_text, clean_content(c["content"]))
            candidates.append((score, c))

    # Kalau tidak ada dokumen lain (hanya 1 dokumen), fallback ke chunk lain
    if not candidates:
        print("  [WARN] Hanya 1 dokumen — distractor dari chunk lain dalam dokumen yang sama.")
        for c in all_doc_chunks[oracle_doc_id]:
            if c.get("chunk_index") != oracle_chunk.get("chunk_index"):
                score = keyword_overlap_score(oracle_text, clean_content(c["content"]))
                candidates.append((score, c))

    # Urutkan: prioritaskan overlap sedang (0.1–0.4) — cukup mirip tapi tidak sama
    # Terlalu mirip (>0.4) bisa tumpang tindih, terlalu jauh (<0.05) terlalu mudah
    def priority_score(item):
        s = item[0]
        if 0.10 <= s <= 0.40:
            return s + 1.0   # prioritas utama
        elif s > 0.40:
            return 2.0 - s   # terlalu mirip, turunkan sedikit
        else:
            return s         # terlalu jauh, prioritas rendah

    candidates.sort(key=priority_score, reverse=True)

    # Ambil top kandidat dengan sedikit randomness
    top_pool = candidates[:max(n * 3, 15)]
    random.shuffle(top_pool)
    selected = [c for _, c in top_pool[:n]]
    return selected


def get_doc_label(chunk: Dict) -> str:
    meta  = chunk.get("metadata", {})
    title = meta.get("title", "Dokumen tidak diketahui")
    pasal = chunk.get("pasal", "") or meta.get("section", "")
    return f"{title}, {pasal.title()}" if pasal else title


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PERTANYAAN
# ─────────────────────────────────────────────────────────────────────────────

def generate_question(
    oracle_content: str,
    doc_title: str,
    question_type: str,
) -> Optional[str]:
    """Generate satu pertanyaan spesifik yang hanya bisa dijawab dari dokumen ini."""
    qcfg = QUESTION_TYPES[question_type]

    system = (
        "Anda adalah pakar pembuat dataset fine-tuning model RAG untuk peraturan desa Indonesia.\n\n"
        f"TUGAS: {qcfg['prompt']}\n\n"
        "ATURAN KETAT:\n"
        "1. Pertanyaan WAJIB menyebut nama peraturan dan/atau nomor pasal agar tidak bisa dijawab tanpa dokumen.\n"
        "2. Pertanyaan harus spesifik — tidak bisa dijawab dari pengetahuan umum.\n"
        "3. Jawaban harus 100% ada dalam dokumen yang diberikan.\n"
        "4. Tulis dalam Bahasa Indonesia yang baik.\n"
        "5. Output HANYA pertanyaan saja, tanpa penjelasan apapun.\n\n"
        f"Dokumen: {doc_title}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Dokumen:\n{oracle_content}\n\nBuat satu pertanyaan bertipe {question_type}:"},
    ]
    result = call_llm(messages, temperature=0.85, max_tokens=200)
    if not result:
        return None
    q = result.strip().strip('"\'')
    q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*|Q[:\s]*)", "", q, flags=re.IGNORECASE).strip()
    return q if len(q) > 20 else None


def pick_best_question(
    oracle_content: str,
    doc_title: str,
    n_candidates: int = 3,
) -> Optional[str]:
    """
    Generate beberapa pertanyaan dari tipe berbeda, pilih yang paling spesifik.
    Kriteria: panjang (proxy spesifisitas) + menyebut nama peraturan/pasal.
    """
    types   = list(QUESTION_TYPES.keys())
    weights = [QUESTION_TYPES[t]["weight"] for t in types]
    chosen_types = random.choices(types, weights=weights, k=n_candidates)
    # Pastikan tidak ada tipe duplikat
    chosen_types = list(dict.fromkeys(chosen_types))

    candidates = []
    for qtype in chosen_types:
        q = generate_question(oracle_content, doc_title, qtype)
        if q:
            candidates.append((qtype, q))
        time.sleep(REQUEST_DELAY)

    if not candidates:
        return None, None

    # Beri skor: panjang + bonus jika menyebut "pasal" / "peraturan" / tahun
    def specificity_score(q: str) -> float:
        score = len(q)
        if re.search(r'pasal\s*\d+', q, re.IGNORECASE):
            score += 30
        if re.search(r'peraturan desa', q, re.IGNORECASE):
            score += 20
        if re.search(r'tahun\s*\d{4}', q, re.IGNORECASE):
            score += 15
        return score

    best_type, best_q = max(candidates, key=lambda x: specificity_score(x[1]))
    return best_type, best_q


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE THOUGHT PROCESS + COMPLETION
# ─────────────────────────────────────────────────────────────────────────────

def generate_thought_and_completion(
    question: str,
    documents: List[str],
    doc_labels: List[str],
    oracle_index: int,          # posisi oracle (0-based), -1 jika absent
    oracle_label: str,
    question_type: str,
    style: str,
    oracle_present: bool,
) -> Tuple[Optional[str], Optional[str]]:

    # Format dokumen
    docs_formatted = ""
    for i, (doc, label) in enumerate(zip(documents, doc_labels), 1):
        docs_formatted += f"\n--- Dokumen {i}: {label} ---\n{doc}\n"

    style_instruction = COMPLETION_STYLES[style]

    # Hint posisi oracle untuk LLM (tidak muncul di output)
    if oracle_present and oracle_index >= 0:
        oracle_hint = (
            f"\n[INTERNAL — jangan sebut di output] "
            f"Dokumen {oracle_index + 1} ({oracle_label}) mengandung jawaban. "
            f"Gunakan ini sebagai panduan analisis.\n"
        )
    else:
        oracle_hint = (
            "\n[INTERNAL — jangan sebut di output] "
            "TIDAK ADA dokumen yang mengandung jawaban lengkap untuk pertanyaan ini.\n"
        )

    # Instruksi thought process berbeda untuk oracle-present vs absent
    if oracle_present:
        thought_instruction = (
            "=== ATURAN THOUGHT_PROCESS ===\n"
            "Analisis SETIAP dokumen satu per satu dengan cara ini:\n"
            "• Untuk dokumen TIDAK relevan: jelaskan MENGAPA secara substantif "
            "(bukan sekadar '(Abaikan)'). Tunjukkan apa yang dibahas dokumen itu "
            "dan mengapa berbeda dari yang ditanyakan.\n"
            "• Untuk dokumen RELEVAN: kutip kalimat kunci dengan format "
            "<<kutipan>>...<<akhir kutipan>> lalu jelaskan mengapa kutipan itu "
            "menjawab pertanyaan.\n"
            "• VARIASIKAN urutan analisis — tidak harus selalu Dokumen 1, 2, 3.\n"
            "• VARIASIKAN panjang analisis tiap dokumen.\n"
        )
        completion_instruction = (
            "=== ATURAN COMPLETION ===\n"
            f"Gaya: {style_instruction}\n\n"
            "1. Jawab LANGSUNG pertanyaan — completion harus bisa berdiri sendiri.\n"
            "2. Sertakan fakta spesifik: angka, definisi, nama jabatan, syarat.\n"
            "3. JANGAN sebut 'Dokumen 1/2/3' — gunakan nama peraturan + nomor pasal.\n"
            "   BENAR: '...sesuai Pasal 7 Peraturan Desa Majasetra No. 1 Tahun 2018.'\n"
            "   SALAH: '...sebagaimana disebutkan di Dokumen 2.'\n"
            "4. Kutip teks asli jika relevan: <<kutipan>>...<<akhir kutipan>>\n"
            "5. VARIASIKAN kalimat pembuka — jangan selalu mulai dengan subjek yang sama.\n"
        )
    else:
        thought_instruction = (
            "=== ATURAN THOUGHT_PROCESS (ORACLE ABSENT) ===\n"
            "Analisis SETIAP dokumen dan tunjukkan bahwa tidak satupun menjawab pertanyaan:\n"
            "• Jelaskan apa topik masing-masing dokumen.\n"
            "• Jelaskan secara spesifik MENGAPA topik tersebut tidak menjawab pertanyaan.\n"
            "• Boleh menyebut jika ada dokumen yang 'mendekati' tapi tidak cukup — dan jelaskan kekurangannya.\n"
        )
        completion_instruction = (
            "=== ATURAN COMPLETION (ORACLE ABSENT) ===\n"
            "Nyatakan bahwa informasi tidak tersedia, tapi dengan SUBSTANSI:\n"
            "• Sebutkan topik yang dibahas dokumen-dokumen yang ada.\n"
            "• Jelaskan dengan tepat apa yang TIDAK ada di dokumen.\n"
            "• JANGAN mengarang jawaban dari dokumen distractor.\n"
            "Contoh yang BENAR:\n"
            "'Dokumen yang tersedia membahas prosedur pengisian anggota BPD dan ketentuan "
            "pemberhentian, namun tidak satupun memuat informasi mengenai [topik pertanyaan]. "
            "Pertanyaan ini tidak dapat dijawab berdasarkan konteks dokumen yang diberikan.'\n"
        )

    system = (
        "Anda adalah model AI yang sedang di-fine-tune dengan metode RAFT "
        "untuk domain peraturan desa Indonesia.\n\n"
        "Output HARUS berupa JSON valid:\n"
        "```json\n"
        "{\n"
        '  "thought_process": "...",\n'
        '  "completion": "..."\n'
        "}\n"
        "```\n\n"
        f"{thought_instruction}\n"
        f"{completion_instruction}"
    )

    user = (
        f"Pertanyaan: {question}\n\n"
        f"Dokumen-dokumen:\n{docs_formatted}"
        f"{oracle_hint}\n"
        f"Analisis semua dokumen dan jawab dengan gaya '{style}'. Output JSON:"
    )

    result = call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.75,
        max_tokens=1536,
    )
    if not result:
        return None, None

    # Parse JSON dengan fallback
    for attempt in [result, re.search(r'\{[\s\S]*\}', result)]:
        try:
            raw = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
            if not raw:
                continue
            parsed     = json.loads(raw)
            thought    = (parsed.get("thought_process") or "").strip()
            completion = (parsed.get("completion") or "").strip()

            # Validasi minimum
            if len(thought) < 100:
                print(f"  [WARN] thought_process terlalu pendek ({len(thought)} chars)")
                return None, None
            if len(completion) < 150:
                print(f"  [WARN] completion terlalu pendek ({len(completion)} chars)")
                return None, None

            # Cek completion tidak mengandung referensi "Dokumen N"
            if re.search(r'\bDokumen\s+\d+\b', completion):
                print(f"  [WARN] completion masih sebut 'Dokumen N' — retry")
                return None, None

            return thought, completion
        except (json.JSONDecodeError, AttributeError):
            continue

    print(f"  Gagal parse JSON.")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# VALIDASI & QUALITY GATE
# ─────────────────────────────────────────────────────────────────────────────

def is_bad_sample(sample: Dict) -> Tuple[bool, str]:
    """
    Quality gate — tolak sampel yang tidak memenuhi standar.
    Returns (is_bad, reason).
    """
    completion    = sample.get("completion", "")
    thought       = sample.get("thought_process", "")
    instruction   = sample.get("instruction", "")
    docs          = sample.get("documents", [])

    if len(completion.strip()) < 150:
        return True, f"completion terlalu pendek ({len(completion)} chars)"

    if len(thought.strip()) < 100:
        return True, f"thought terlalu pendek ({len(thought)} chars)"

    if len(instruction.strip()) < 20:
        return True, "instruction terlalu pendek"

    if not docs:
        return True, "tidak ada dokumen"

    # Cek referensi 'Dokumen N' di completion
    if re.search(r'\bDokumen\s+\d+\b', completion):
        return True, "completion menyebut 'Dokumen N'"

    # Deteksi completion yang hanya menyalin oracle mentah (Jaccard > 0.85)
    oracle_present = sample.get("metadata_extra", {}).get("oracle_present", True)
    if oracle_present and docs:
        comp_words = set(completion.lower().split())
        for doc in docs:
            doc_words = set(doc.lower().split())
            if not doc_words:
                continue
            j = len(comp_words & doc_words) / len(comp_words | doc_words)
            if j > 0.85:
                return True, f"completion terlalu mirip dengan dokumen (Jaccard={j:.2f})"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_raft_dataset(
    all_doc_chunks: Dict[str, List[Dict]],
    output_path: Path,
    chunks_per_doc: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:

    random.seed(seed)
    dataset, failed, skipped = [], 0, 0

    # Hitung total oracle (hanya chunk substantif)
    all_oracle_chunks = []
    for doc_id, chunks in all_doc_chunks.items():
        substantive = [c for c in chunks if is_substantive(c["content"])]
        if chunks_per_doc:
            substantive = random.sample(substantive, min(len(substantive), chunks_per_doc))
        all_oracle_chunks.extend([(doc_id, c) for c in substantive])

    non_substantive = sum(len(v) for v in all_doc_chunks.values()) - len(all_oracle_chunks)
    print(f"\nTotal chunks      : {sum(len(v) for v in all_doc_chunks.values())}")
    print(f"Oracle substantif : {len(all_oracle_chunks)}")
    print(f"Dilewati (pendek) : {non_substantive}")
    print(f"Output            : {output_path}\n")
    print("=" * 70)

    pbar = tqdm(all_oracle_chunks, desc="Generating RAFT")

    for doc_id, chunk in pbar:
        oracle_content = clean_content(chunk["content"])
        doc_title      = chunk["metadata"].get("title", "Dokumen tidak diketahui")
        oracle_label   = get_doc_label(chunk)

        # ── Step 1: Generate pertanyaan terbaik ──
        question_type, question = pick_best_question(oracle_content, doc_title, n_candidates=3)
        if not question:
            failed += 1
            pbar.set_postfix(ok=len(dataset), fail=failed, skip=skipped)
            pbar.update(1)
            continue

        # ── Step 2: Pilih distractor berbasis keyword overlap ──
        distractor_chunks   = select_distractors(chunk, all_doc_chunks)
        distractor_contents = [clean_content(c["content"]) for c in distractor_chunks]
        distractor_labels   = [get_doc_label(c) for c in distractor_chunks]

        # ── Step 3: Tentukan oracle hadir atau tidak ──
        oracle_present = random.random() < ORACLE_PRESENT_RATIO

        if oracle_present:
            oracle_pos = random.randint(0, len(distractor_contents))
            documents  = distractor_contents[:oracle_pos] + [oracle_content] + distractor_contents[oracle_pos:]
            doc_labels = distractor_labels[:oracle_pos] + [oracle_label] + distractor_labels[oracle_pos:]
        else:
            documents  = distractor_contents
            doc_labels = distractor_labels
            oracle_pos = -1

        # ── Step 4: Pilih gaya yang kompatibel dengan tipe pertanyaan ──
        compatible = QUESTION_TYPES[question_type]["compatible_styles"]
        style      = random.choice(compatible)

        # ── Step 5: Generate thought + completion ──
        thought, completion = generate_thought_and_completion(
            question       = question,
            documents      = documents,
            doc_labels     = doc_labels,
            oracle_index   = oracle_pos,
            oracle_label   = oracle_label,
            question_type  = question_type,
            style          = style,
            oracle_present = oracle_present,
        )

        if not thought or not completion:
            failed += 1
            pbar.set_postfix(ok=len(dataset), fail=failed, skip=skipped)
            pbar.update(1)
            time.sleep(REQUEST_DELAY)
            continue

        # ── Step 6: Quality gate ──
        sample = {
            "instruction"   : question,
            "documents"     : documents,
            "thought_process": thought,
            "completion"    : completion,
            "metadata_extra": {
                "style"          : style,
                "question_type"  : question_type,
                "oracle_present" : oracle_present,
                "oracle_doc_id"  : doc_id if oracle_present else None,
            },
        }

        bad, reason = is_bad_sample(sample)
        if bad:
            print(f"\n  [SKIP] {reason}")
            skipped += 1
            pbar.set_postfix(ok=len(dataset), fail=failed, skip=skipped)
            pbar.update(1)
            time.sleep(REQUEST_DELAY)
            continue

        # ── Step 7: Simpan ──
        dataset.append(sample)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        pbar.set_postfix(ok=len(dataset), fail=failed, skip=skipped)
        pbar.update(1)
        time.sleep(REQUEST_DELAY)

    pbar.close()
    print("=" * 70)
    print(f"Selesai!")
    print(f"  Berhasil : {len(dataset)}")
    print(f"  Gagal    : {failed}")
    print(f"  Dilewati : {skipped} (tidak lolos quality gate)")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# VALIDASI DATASET
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(filepath: Path):
    """Tampilkan ringkasan statistik dataset yang dihasilkan."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    total = len(samples)
    if total == 0:
        print("Dataset kosong.")
        return

    oracle_present  = sum(1 for s in samples if s.get("metadata_extra", {}).get("oracle_present", True))
    oracle_absent   = total - oracle_present

    # Distribusi question type
    qtypes = Counter(s.get("metadata_extra", {}).get("question_type", "?") for s in samples)

    # Distribusi style
    styles = Counter(s.get("metadata_extra", {}).get("style", "?") for s in samples)

    # Panjang rata-rata
    avg_inst   = sum(len(s["instruction"])    for s in samples) / total
    avg_thought = sum(len(s["thought_process"]) for s in samples) / total
    avg_comp   = sum(len(s["completion"])     for s in samples) / total
    avg_docs   = sum(len(s["documents"])      for s in samples) / total

    # Distinct-2 completion
    all_bigrams = []
    for s in samples:
        tokens = s["completion"].lower().split()
        all_bigrams.extend(zip(tokens, tokens[1:]))
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    print("\n" + "=" * 60)
    print("STATISTIK DATASET")
    print("=" * 60)
    print(f"Total sampel     : {total}")
    print(f"Oracle present   : {oracle_present} ({oracle_present/total*100:.1f}%)")
    print(f"Oracle absent    : {oracle_absent}  ({oracle_absent/total*100:.1f}%)")
    print()
    print("Distribusi tipe pertanyaan:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"  {qt:<15}: {count:>4} ({count/total*100:.0f}%)")
    print()
    print("Distribusi gaya jawaban:")
    for st, count in sorted(styles.items(), key=lambda x: -x[1]):
        print(f"  {st:<15}: {count:>4} ({count/total*100:.0f}%)")
    print()
    print(f"Rata-rata panjang instruction  : {avg_inst:.0f} chars")
    print(f"Rata-rata panjang thought      : {avg_thought:.0f} chars")
    print(f"Rata-rata panjang completion   : {avg_comp:.0f} chars")
    print(f"Rata-rata jumlah dokumen       : {avg_docs:.1f}")
    print(f"Distinct-2 completion          : {distinct_2:.3f}  (target > 0.7)")

    # Warning
    if oracle_absent / total < 0.15:
        print("\n⚠️  WARNING: oracle_absent < 15% — tambah sampel tanpa oracle!")
    if distinct_2 < 0.5:
        print("\n⚠️  WARNING: Distinct-2 rendah — completion terlalu homogen!")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Ubah ke nama file tertentu, atau None untuk semua file
    SINGLE_FILE = None

    if SINGLE_FILE:
        path = PROCESSED_DIR / SINGLE_FILE
        if not path.exists():
            print(f"ERROR: File tidak ditemukan: {path}")
            sys.exit(1)
        all_doc_chunks = load_single_chunk_file(path)
    else:
        print("Memuat semua chunks...")
        all_doc_chunks = load_all_chunks(PROCESSED_DIR)

    total_chunks = sum(len(v) for v in all_doc_chunks.values())
    print(f"Total: {len(all_doc_chunks)} dokumen, {total_chunks} chunks\n")

    # Test koneksi
    print("Testing API...")
    test = call_llm([{"role": "user", "content": "Katakan 'OK' saja."}], max_tokens=10)
    print(f"API response: {test}\n")

    # Output path + backup otomatis
    OUTPUT_FILENAME = "raft_dataset_v2.jsonl"
    output_path = DATASET_DIR / OUTPUT_FILENAME
    if output_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DATASET_DIR / f"raft_dataset_v2_backup_{ts}.jsonl"
        shutil.move(str(output_path), str(backup))
        print(f"Backup: {backup}")

    # Generate
    raft_dataset = generate_raft_dataset(
        all_doc_chunks=all_doc_chunks,
        output_path=output_path,
        chunks_per_doc=1,   # None = semua chunk substantif
        seed=42,
    )

    # Validasi
    if output_path.exists():
        validate_dataset(output_path)

        # Tampilkan 2 sampel
        print("\nCONTOH SAMPEL:")
        with open(output_path) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                s = json.loads(line)
                print(f"\n{'─'*60}")
                print(f"[{i+1}] {s['instruction']}")
                print(f"Type : {s['metadata_extra'].get('question_type')}")
                print(f"Style: {s['metadata_extra'].get('style')}")
                print(f"Oracle present: {s['metadata_extra'].get('oracle_present')}")
                print(f"Thought ({len(s['thought_process'])} chars): {s['thought_process'][:200]}...")
                print(f"Completion ({len(s['completion'])} chars): {s['completion'][:200]}...")