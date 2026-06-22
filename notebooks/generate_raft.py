"""
# RAFT Dataset Generator for Peraturan Desa

Skrip ini menghasilkan dataset Retrieval Augmented Fine-Tuning (RAFT) dalam format JSONL.

## Referensi Jurnal
1. Dong et al. (2024) - RAFT: Adapting Language Model to Domain Specific RAG
2. Liu et al. (2024) - RA-DIT: Retrieval-Augmented Dual Instruction Tuning
3. Gao et al. (2024) - Retrieval-Augmented Generation for Large Language Models: A Survey

## Model HuggingFace yang Digunakan
- Qwen/Qwen2.5-7B-Instruct (via HuggingFace Inference API)
- Alternatif: meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3
"""

import os
import sys
import json
import time
import random
import re
import shutil
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ============================================================
# 1. SETUP & KONFIGURASI
# ============================================================

# Tambahkan root project ke path
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
HF_BASE_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

# Model untuk generate dataset
GENERATOR_MODEL = "openai/gpt-4o-mini"

# Path data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Parameter
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
REQUEST_DELAY = 1.5  # detik antara request
NUM_DISTRACTORS = 4  # jumlah distractor per sampel (RAFT paper recommends 4)

# Question type taxonomy (FLAN, Evol-Instruct, Bloom's Taxonomy)
QUESTION_TYPES = {
    "faktual": {
        "instruction": "Buat pertanyaan FAKTUAL yang menanyakan data, angka, atau fakta spesifik (misal: 'Berapa lama...', 'Berapa jumlah...', 'Kapan...').",
        "weight": 0.20
    },
    "definisional": {
        "instruction": "Buat pertanyaan DEFINISIONAL yang menanyakan pengertian atau definisi (misal: 'Apa yang dimaksud dengan...', 'Jelaskan makna...').",
        "weight": 0.15
    },
    "prosedural": {
        "instruction": "Buat pertanyaan PROSEDURAL yang menanyakan langkah-langkah atau tata cara (misal: 'Bagaimana tata cara...', 'Jelaskan prosedur...').",
        "weight": 0.15
    },
    "komparatif": {
        "instruction": "Buat pertanyaan KOMPARATIF yang membandingkan dua hal atau lebih (misal: 'Apa perbedaan antara X dan Y...', 'Bandingkan ketentuan tentang...').",
        "weight": 0.10
    },
    "kondisional": {
        "instruction": "Buat pertanyaan KONDISIONAL yang menanyakan sebab-akibat atau kondisi (misal: 'Apa yang terjadi jika...', 'Bagaimana jika...').",
        "weight": 0.15
    },
    "enumeratif": {
        "instruction": "Buat pertanyaan ENUMERATIF yang menanyakan daftar atau rincian lengkap (misal: 'Sebutkan semua...', 'Apa saja yang termasuk...').",
        "weight": 0.15
    },
    "interpretatif": {
        "instruction": "Buat pertanyaan INTERPRETATIF yang menanyakan alasan, tujuan, atau makna di balik suatu ketentuan (misal: 'Mengapa ketentuan tentang X penting...', 'Apa tujuan dari...').",
        "weight": 0.10
    },
}

# Completion style variation (Attribute-Conditioned Generation, GEM Benchmark)
COMPLETION_STYLES = {
    "langsung": "Jawab langsung dan padat dalam 2-3 kalimat. Fokus pada inti jawaban tanpa basa-basi.",
    "penjelasan": "Berikan jawaban lengkap dengan konteks dan penjelasan mengapa ketentuan tersebut ada. Minimal 3-4 kalimat.",
    "terstruktur": "Gunakan format terstruktur: mulai dengan pernyataan utama, lalu rinci poin-poin penting secara berurutan.",
    "percakapan": "Jawab seolah menjelaskan kepada warga desa yang bertanya. Gunakan bahasa yang lebih mudah dipahami namun tetap akurat secara hukum. Minimal 3 kalimat.",
    "formal_hukum": "Gunakan bahasa hukum formal seperti penjelasan resmi pemerintah atau putusan. Kutip pasal dan ayat secara presisi.",
    "ringkasan": "Mulai dengan ringkasan singkat satu kalimat, lalu detailkan 2-3 aspek terpenting dari ketentuan tersebut.",
}

# Oracle-absent ratio (RAFT Paper, Zhang et al. 2024, Section 4.5)
ORACLE_PRESENT_RATIO = 0.8  # 80% of samples include oracle, 20% oracle-absent

print(f"HF_BASE_URL: {HF_BASE_URL}")
print(f"Generator Model: {GENERATOR_MODEL}")
print(f"Processed Dir: {PROCESSED_DIR}")
print(f"Dataset Dir: {DATASET_DIR}")


# ============================================================
# 2. LOAD & PREPROCESS CHUNKS
# ============================================================

def load_all_chunks(processed_dir: Path) -> Dict[str, List[Dict]]:
    """Load semua file *_chunks.json. Returns: dict {document_id: [chunks]}"""
    all_chunks = {}
    chunk_files = sorted(processed_dir.glob("*_chunks.json"))
    
    for fpath in chunk_files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
        all_chunks[doc_id] = chunks
        print(f"  Loaded {len(chunks):>4} chunks from: {fpath.name} (doc_id={doc_id})")
    
    return all_chunks

def load_single_chunk_file(filepath: Path) -> Dict[str, List[Dict]]:
    """Load satu file *_chunks.json. Returns: dict {document_id: [chunks]}"""
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    doc_id = chunks[0]["metadata"]["document_id"] if chunks else filepath.stem
    all_chunks = {doc_id: chunks}
    print(f"  Loaded {len(chunks):>4} chunks from: {filepath.name} (doc_id={doc_id})")
    return all_chunks

def clean_chunk_content(content: str) -> str:
    """Bersihkan konten chunk dari metadata header."""
    cleaned = re.sub(r"^\[dokumen:.*?\]\s*", "", content, flags=re.MULTILINE)
    cleaned = re.sub(r"\[desa:.*?\]\s*", "", cleaned)
    cleaned = re.sub(r"\[kabupaten:.*?\]\s*", "", cleaned)
    cleaned = re.sub(r"\[nomor:.*?\]\s*", "", cleaned)
    return cleaned.strip()

def get_document_title(chunk: Dict) -> str:
    """Ambil title dokumen dari metadata."""
    return chunk["metadata"].get("title", "Dokumen tidak diketahui")


# ============================================================
# 3. HUGGINGFACE API CLIENT
# ============================================================

def call_hf_chat(
    messages: List[Dict[str, str]],
    model: str = GENERATOR_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    retries: int = 3,
    frequency_penalty: float = 0.3,
    presence_penalty: float = 0.2,
) -> Optional[str]:
    """Panggil HuggingFace Inference API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    
    for attempt in range(retries):
        try:
            resp = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  Rate limited. Menunggu {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 503:
                wait = (attempt + 1) * 10
                print(f"  Model loading. Menunggu {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP Error: {e}")
                if attempt < retries - 1:
                    time.sleep(3)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


# ============================================================
# 4. STRATEGY: GENERATE RAFT SAMPLES
# ============================================================

def select_distractors(
    oracle_chunk: Dict,
    all_doc_chunks: Dict[str, List[Dict]],
    n: int = NUM_DISTRACTORS,
) -> List[Dict]:
    """Pilih n distractor documents dari dokumen BERBEDA. Returns full chunk objects with metadata."""
    oracle_doc_id = oracle_chunk["metadata"]["document_id"]
    
    candidates = []
    for doc_id, chunks in all_doc_chunks.items():
        if doc_id != oracle_doc_id:
            for c in chunks:
                candidates.append(c)
    
    if not candidates:
        for c in all_doc_chunks[oracle_doc_id]:
            if c["chunk_index"] != oracle_chunk["chunk_index"]:
                candidates.append(c)
    
    selected = random.sample(candidates, min(n, len(candidates)))
    return selected


def build_doc_label(chunk: Dict) -> str:
    """Build a human-readable document label from chunk metadata."""
    meta = chunk.get("metadata", {})
    title = meta.get("title", "Dokumen tidak diketahui")
    pasal = chunk.get("pasal", "") or meta.get("section", "")
    if pasal:
        return f"{title}, {pasal.title()}"
    return title


def generate_question_from_chunk(
    oracle_content: str, doc_title: str, question_type: Optional[str] = None
) -> Optional[str]:
    """Gunakan LLM untuk generate pertanyaan dari oracle_content dengan tipe pertanyaan spesifik."""
    
    # Select question type if not provided
    if question_type is None:
        question_type = random.choice(list(QUESTION_TYPES.keys()))
    
    type_config = QUESTION_TYPES[question_type]
    type_instruction = type_config["instruction"]
    
    # Randomly select perspective
    perspectives = [
        "Netral — tanyakan dari sudut pandang umum.",
        "Dari sudut pandang warga desa yang ingin memahami hak atau kewajibannya.",
        "Dari sudut pandang kepala desa atau perangkat desa yang perlu menjalankan ketentuan.",
        "Dari sudut pandang badan permusyawaratan desa (BPD) yang mengawasi pelaksanaan.",
    ]
    perspective = random.choice(perspectives)
    
    system_prompt = (
        "Anda adalah pakar pembuat dataset untuk fine-tuning model RAG "
        "(Retrieval Augmented Generation) khusus domain peraturan desa Indonesia.\n\n"
        f"Tugas: {type_instruction}\n\n"
        f"Perspektif: {perspective}\n\n"
        "Aturan:\n"
        "1. Pertanyaan HARUS bisa dijawab HANYA dari isi dokumen.\n"
        "2. Pertanyaan harus spesifik dan menantang, bukan pertanyaan umum.\n"
        "3. Pertanyaan harus menggunakan Bahasa Indonesia yang baik.\n"
        "4. Sertakan konteks dokumen dalam pertanyaan (nama peraturan, nomor, tahun) jika relevan.\n"
        "5. Output HANYA pertanyaan, tanpa penjelasan atau komentar tambahan.\n\n"
        f"Dokumen: {doc_title}\n\n"
        f"Isi:\n{oracle_content}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Buat satu pertanyaan bertipe {question_type} berdasarkan isi dokumen di atas."}
    ]
    
    result = call_hf_chat(messages, temperature=0.9, max_tokens=250)
    if result:
        q = result.strip().strip('"').strip("'").strip()
        q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*|Q[:\s]*)", "", q, flags=re.IGNORECASE).strip()
        return q
    return None


def generate_question_candidates(
    oracle_content: str, doc_title: str, n_candidates: int = 3
) -> Optional[str]:
    """Generate multiple question candidates and select the most diverse one."""
    available_types = list(QUESTION_TYPES.keys())
    selected_types = random.sample(available_types, min(n_candidates, len(available_types)))
    
    candidates = []
    for qtype in selected_types:
        q = generate_question_from_chunk(oracle_content, doc_title, question_type=qtype)
        if q:
            candidates.append(q)
        time.sleep(REQUEST_DELAY)
    
    if not candidates:
        return None
    
    # Simple diversity: return the longest unique question (proxy for specificity)
    # Remove near-duplicates via word-level Jaccard
    unique = [candidates[0]]
    for c in candidates[1:]:
        words_c = set(c.lower().split())
        is_dup = False
        for u in unique:
            words_u = set(u.lower().split())
            intersection = words_c & words_u
            union = words_c | words_u
            jaccard = len(intersection) / len(union) if union else 1.0
            if jaccard > 0.6:
                is_dup = True
                break
        if not is_dup:
            unique.append(c)
    
    # Select randomly from unique candidates
    return random.choice(unique)


def generate_thought_and_completion(
    instruction: str,
    documents: List[str],
    doc_labels: List[str],
    oracle_index: int,
    oracle_content: str,
    oracle_label: str,
    style: str = "langsung",
    oracle_present: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate thought_process dan completion menggunakan LLM dengan metadata dokumen."""
    
    # Format documents with their actual labels
    docs_formatted = ""
    for i, (doc, label) in enumerate(zip(documents, doc_labels), 1):
        docs_formatted += f"\n--- Dokumen {i}: {label} ---\n{doc}\n"
    
    style_instruction = COMPLETION_STYLES.get(style, COMPLETION_STYLES["langsung"])
    
    # Handle oracle-absent case
    oracle_absent_instruction = ""
    if not oracle_present:
        oracle_absent_instruction = (
            "\nPENTING: Pertanyaan ini mungkin TIDAK memiliki jawaban dalam dokumen-dokumen yang diberikan. "
            "Jika tidak ada dokumen yang menjawab pertanyaan dengan memadai, tulis completion:\n"
            "'Informasi mengenai [topik pertanyaan] tidak tersedia dalam dokumen-dokumen yang diberikan.'\n"
            "Dan pada thought_process, jelaskan bahwa tidak ada dokumen yang memuat jawaban.\n"
        )
    
    system_prompt = (
        "Anda adalah model AI yang sedang di-fine-tune dengan metode RAFT "
        "(Retrieval Augmented Fine-Tuning) untuk domain peraturan desa Indonesia.\n\n"
        "Anda akan menerima:\n"
        "1. Sebuah pertanyaan/instruksi.\n"
        "2. Beberapa dokumen dengan identitas lengkap (nama peraturan, nomor, pasal).\n\n"
        "Anda HARUS menghasilkan output JSON valid dengan 2 field:\n\n"
        '```json\n'
        '{\n'
        '  "thought_process": "Analisis mendalam setiap dokumen...",\n'
        '  "completion": "Jawaban LENGKAP, DESKRIPTIF, dan ALAMIAH..."\n'
        '}\n'
        '```\n\n'
        "=== ATURAN THOUGHT_PROCESS ===\n"
        "1. Analisis SETIAP dokumen secara individual.\n"
        "2. Untuk dokumen TIDAK relevan: jelaskan secara spesifik MENGAPA tidak relevan, bukan hanya label '(Abaikan)'.\n"
        "3. Untuk dokumen RELEVAN: KUTIP kalimat atau frasa spesifik dari dokumen tersebut sebagai bukti, "
        "gunakan format <<kutipan>>...<<akhir kutipan>>.\n"
        "4. Berikan JEMBATAN PENALARAN: jelaskan MENGAPA kutipan tersebut menjawab pertanyaan.\n"
        "5. TIDAK harus menganalisis dalam urutan 1, 2, 3. Boleh mulai dari dokumen yang paling menarik perhatian Anda.\n"
        "6. Variasikan panjang analisis: kadang singkat (2-3 kalimat), kadang detail (5-6 kalimat).\n\n"
        "CONTOH thought_process yang BAIK:\n"
        '"Dokumen 2 (Peraturan Desa X, Pasal 5) membahas tentang sanksi administratif, '
        'namun pertanyaan menanyakan sanksi pidana — keduanya berbeda ranah hukum, jadi dokumen ini '
        'kurang tepat (Abaikan). Dokumen 4 (Peraturan Desa Y, Pasal 13) secara eksplisit menyatakan '
        '<<kutipan>>pihak perusahaan yang tidak mematuhi ketentuan dikenakan sanksi berupa surat peringatan '
        'dan pencabutan izin<<akhir kutipan>>. Kutipan ini langsung menjawab pertanyaan karena menyebutkan '
        'dua jenis sanksi secara konkret (Sangat Relevan). Berdasarkan informasi ini, jawabannya jelas '
        'bahwa sanksi meliputi peringatan tertulis dan pencabutan izin."\n\n'
        "=== ATURAN COMPLETION (SANGAT PENTING) ===\n"
        f"Gaya jawaban: {style_instruction}\n\n"
        "1. JANGAN PERNAH menyebut 'Dokumen 1', 'Dokumen 2', dst. sebagai referensi!\n"
        "   Sebagai gantinya, gunakan NAMA PERATURAN yang ACTUAL beserta NOMOR dan PASAL-nya.\n"
        "   Contoh BENAR: '...berdasarkan Pasal 13 Peraturan Desa Biru No. 05 Tahun 2016.'\n"
        "   Contoh SALAH: '...sebagaimana tercantum dalam Dokumen 3.'\n\n"
        "2. Completion HARUS berisi JAWABAN LENGKAP yang bisa berdiri sendiri.\n"
        "3. Sertakan detail/fakta SPESIFIK: definisi lengkap, angka pasti, nama jabatan, syarat rinci.\n"
        "4. Gunakan kutipan langsung dari dokumen jika relevan, format: <<kutipan>>teks asli<<akhir kutipan>>\n"
        "5. Jawaban harus TERTSERA alami, seperti penjelasan dari seorang pakar — BUKAN template kaku.\n"
        "6. Variasikan cara membuka dan menutup jawaban. JANGAN selalu mulai dengan subjek yang sama.\n\n"
        "CONTOH completion yang BENAR:\n"
        '- "Sanksi bagi perusahaan yang tidak mematuhi peraturan desa meliputi dua tahap: '
        'pertama, melayangkan surat peringatan, dan kedua, mencabut surat rekomendasi perijinan '
        'dari pemerintahan setempat. Ketentuan ini diatur dalam Pasal 13 Peraturan Desa Biru '
        'No. 05 Tahun 2016, yang bertujuan menjamin kepatuhan pihak swasta terhadap regulasi desa."\n'
        '- "Menurut Pasal 1 Peraturan Desa Cipedes No. 01 Tahun 2018, profil desa didefinisikan '
        'sebagai <<kutipan>>gambaran menyeluruh tentang karakter desa yang meliputi data dasar keluarga, '
        'potensi sumber daya alam, sumber daya manusia, kelembagaan, prasarana dan sarana serta '
        'perkembangan kemajuan dan permasalahan yang dihadapi desa<<akhir kutipan>>. Definisi ini '
        'menunjukkan bahwa profil desa mencakup aspek yang sangat komprehensif."\n\n'
        "CONTOH completion yang SALAH (JANGAN lakukan):\n"
        '- "Sumber pendanaan berasal dari APBN, sebagaimana dijelaskan dalam Dokumen 3." ← referensi generik\n'
        '- "Dokumen 1" ← terlalu singkat\n'
        '- "Jawaban dapat ditemukan di Dokumen 2" ← tidak menjawab langsung\n'
        + oracle_absent_instruction
    )

    user_prompt = (
        f"Pertanyaan: {instruction}\n\n"
        f"Dokumen-dokumen:\n{docs_formatted}\n\n"
        f"Analisis semua dokumen secara mendalam dan berikan jawaban lengkap dengan gaya '{style}'. Output JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = call_hf_chat(messages, temperature=0.8, max_tokens=1536)
    if not result:
        return None, None

    # Parse JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            thought = parsed.get("thought_process")
            completion = parsed.get("completion")
            if completion and len(completion.strip()) < 60:
                print(f"  [WARN] completion terlalu singkat ({len(completion)} chars), retry...")
                return None, None
            return thought, completion
    except json.JSONDecodeError:
        pass

    try:
        parsed = json.loads(result)
        thought = parsed.get("thought_process")
        completion = parsed.get("completion")
        if completion and len(completion.strip()) < 60:
            print(f"  [WARN] completion terlalu singkat ({len(completion)} chars), retry...")
            return None, None
        return thought, completion
    except json.JSONDecodeError:
        print(f"  Gagal parse JSON. Raw: {result[:300]}...")
        return None, None


# ============================================================
# 5. GENERATE RAFT DATASET
# ============================================================

def generate_raft_dataset(
    all_doc_chunks: Dict[str, List[Dict]],
    output_path: Path,
    chunks_per_doc: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """Generate dataset RAFT dari semua dokumen."""
    random.seed(seed)
    dataset = []
    failed = 0
    
    total_oracles = 0
    for doc_id, chunks in all_doc_chunks.items():
        n = min(len(chunks), chunks_per_doc) if chunks_per_doc else len(chunks)
        total_oracles += n
    
    print(f"Total oracle chunks: {total_oracles}")
    print(f"Distractors per sample: {NUM_DISTRACTORS}")
    print(f"Output: {output_path}\n")
    print("=" * 70)
    
    pbar = tqdm(total=total_oracles, desc="Generating RAFT")
    
    for doc_id, chunks in all_doc_chunks.items():
        doc_title = get_document_title(chunks[0])
        
        if chunks_per_doc and len(chunks) > chunks_per_doc:
            selected_chunks = random.sample(chunks, chunks_per_doc)
        else:
            selected_chunks = chunks
        
        for chunk in selected_chunks:
            oracle_content = clean_chunk_content(chunk["content"])
            
            if len(oracle_content) < 50:
                pbar.update(1)
                continue
            
            # Step 1: Generate question with overgenerate & filter
            question = generate_question_candidates(oracle_content, doc_title, n_candidates=3)
            if not question:
                failed += 1
                pbar.update(1)
                continue
            
            time.sleep(REQUEST_DELAY)
            
            # Step 2: Select distractors (returns full chunk objects)
            distractor_chunks = select_distractors(chunk, all_doc_chunks, n=NUM_DISTRACTORS)
            distractor_contents = [clean_chunk_content(c["content"]) for c in distractor_chunks]
            distractor_labels = [build_doc_label(c) for c in distractor_chunks]
            
            # Step 3: Oracle presence/absence (RAFT Paper, Section 4.5)
            oracle_present = random.random() < ORACLE_PRESENT_RATIO

            oracle_content_clean = clean_chunk_content(chunk["content"])
            oracle_label = build_doc_label(chunk)

            if oracle_present:
                oracle_pos = random.randint(0, len(distractor_contents))
                documents = distractor_contents[:oracle_pos] + [oracle_content_clean] + distractor_contents[oracle_pos:]
                doc_labels = distractor_labels[:oracle_pos] + [oracle_label] + distractor_labels[oracle_pos:]
            else:
                documents = distractor_contents
                doc_labels = distractor_labels
                oracle_pos = -1  # oracle not present
            
            # Step 4: Select completion style
            style = random.choice(list(COMPLETION_STYLES.keys()))
            
            # Step 5: Generate thought_process & completion
            thought, completion = generate_thought_and_completion(
                instruction=question,
                documents=documents,
                doc_labels=doc_labels,
                oracle_index=oracle_pos,
                oracle_content=oracle_content_clean,
                oracle_label=oracle_label,
                style=style,
                oracle_present=oracle_present,
            )
            
            if not thought or not completion:
                failed += 1
                pbar.update(1)
                time.sleep(REQUEST_DELAY)
                continue
            
            # Step 6: Format & save
            sample = {
                "instruction": question,
                "documents": documents,
                "thought_process": thought,
                "completion": completion,
                "metadata_extra": {
                    "style": style,
                    "oracle_present": oracle_present,
                }
            }
            dataset.append(sample)
            
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            pbar.set_postfix(ok=len(dataset), fail=failed)
            pbar.update(1)
            time.sleep(REQUEST_DELAY)
    
    pbar.close()
    print("=" * 70)
    print(f"\nSelesai! {len(dataset)} sampel berhasil, {failed} gagal.")
    print(f"  Output: {output_path}")
    return dataset


# ============================================================
# 6. VALIDASI & INSPEKSI DATASET
# ============================================================

def _get_ngrams(tokens, n):
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def validate_raft_dataset(filepath: Path) -> Dict:
    """Validasi dataset RAFT."""
    stats = {
        "total_samples": 0, "valid": 0, "invalid": 0, "errors": [],
        "avg_instruction_len": 0, "avg_thought_len": 0,
        "avg_completion_len": 0, "avg_docs_per_sample": 0,
        # Diversity metrics (added)
        "distinct_1_completion": 0.0,
        "distinct_2_completion": 0.0,
        "distinct_2_thought": 0.0,
        "avg_jaccard_similarity": 0.0,
        "question_patterns": {},
        "question_pattern_warning": False,
        "near_duplicate_pairs": 0,
    }

    instruction_lens, thought_lens, completion_lens, docs_counts = [], [], [], []
    required_keys = {"instruction", "documents", "thought_process", "completion"}

    # Collect texts for diversity metrics
    all_completions_tokens = []
    all_thought_tokens = []
    all_instructions = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stats["total_samples"] += 1
            try:
                sample = json.loads(line.strip())
                missing = required_keys - set(sample.keys())
                if missing:
                    stats["errors"].append(f"Line {line_no}: missing keys {missing}")
                    stats["invalid"] += 1
                    continue
                if not isinstance(sample["documents"], list):
                    stats["errors"].append(f"Line {line_no}: documents not a list")
                    stats["invalid"] += 1
                    continue
                stats["valid"] += 1
                instruction_lens.append(len(sample["instruction"]))
                thought_lens.append(len(sample["thought_process"]))
                completion_lens.append(len(sample["completion"]))
                docs_counts.append(len(sample["documents"]))

                # Collect for diversity
                all_completions_tokens.append(sample["completion"].lower().split())
                all_thought_tokens.append(sample["thought_process"].lower().split())
                all_instructions.append(sample["instruction"])
            except json.JSONDecodeError as e:
                stats["errors"].append(f"Line {line_no}: JSON error - {e}")
                stats["invalid"] += 1

    if instruction_lens:
        stats["avg_instruction_len"] = sum(instruction_lens) / len(instruction_lens)
        stats["avg_thought_len"] = sum(thought_lens) / len(thought_lens)
        stats["avg_completion_len"] = sum(completion_lens) / len(completion_lens)
        stats["avg_docs_per_sample"] = sum(docs_counts) / len(docs_counts)

    # --- Diversity Metrics ---
    n_samples = len(all_completions_tokens)

    if n_samples >= 1:
        # Distinct-1 (completion): unique unigrams / total unigrams
        all_comp_unigrams = []
        for tokens in all_completions_tokens:
            all_comp_unigrams.extend(tokens)
        total_unigrams = len(all_comp_unigrams)
        unique_unigrams = len(set(all_comp_unigrams))
        stats["distinct_1_completion"] = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0

        # Distinct-2 (completion): unique bigrams / total bigrams
        all_comp_bigrams = []
        for tokens in all_completions_tokens:
            all_comp_bigrams.extend(_get_ngrams(tokens, 2))
        total_bigrams = len(all_comp_bigrams)
        unique_bigrams = len(set(all_comp_bigrams))
        stats["distinct_2_completion"] = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0

        # Distinct-2 (thought): unique bigrams / total bigrams over thought_process
        all_thought_bigrams = []
        for tokens in all_thought_tokens:
            all_thought_bigrams.extend(_get_ngrams(tokens, 2))
        total_thought_bigrams = len(all_thought_bigrams)
        unique_thought_bigrams = len(set(all_thought_bigrams))
        stats["distinct_2_thought"] = unique_thought_bigrams / total_thought_bigrams if total_thought_bigrams > 0 else 0.0

    if n_samples >= 2:
        # Average pairwise Jaccard similarity (word-level)
        word_sets = [set(tokens) for tokens in all_completions_tokens]

        if n_samples > 50:
            # Sample 50 random pairs for efficiency
            pairs = set()
            while len(pairs) < 50:
                a, b = random.sample(range(n_samples), 2)
                pairs.add((min(a, b), max(a, b)))
            pairs = list(pairs)
        else:
            pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]

        jaccard_scores = []
        near_dup_count = 0
        for i, j in pairs:
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            sim = len(intersection) / len(union) if union else 1.0
            jaccard_scores.append(sim)
            if sim > 0.6:
                near_dup_count += 1

        stats["avg_jaccard_similarity"] = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        stats["near_duplicate_pairs"] = near_dup_count

    # Question type pattern detection
    question_patterns_list = [
        "Berapa", "Apa", "Bagaimana", "Mengapa", "Sebutkan",
        "Jelaskan", "Siapa", "Berdasarkan", "Menurut",
    ]
    pattern_counts = {p: 0 for p in question_patterns_list}
    total_questions = len(all_instructions)
    if total_questions > 0:
        for instr in all_instructions:
            first_word = instr.strip().split()[0].rstrip("?:.,;!-\"").capitalize() if instr.strip() else ""
            for pattern in question_patterns_list:
                if first_word.lower() == pattern.lower():
                    pattern_counts[pattern] += 1
                    break
        # Remove patterns with 0 count for cleaner output? No, keep all for completeness
        stats["question_patterns"] = pattern_counts
        # Flag if any single pattern exceeds 35%
        for count in pattern_counts.values():
            if count / total_questions > 0.35:
                stats["question_pattern_warning"] = True
                break

    return stats


# ============================================================
# 9. PERBAIKAN COMPLETION PADA DATASET
# ============================================================

def is_bad_completion(completion: str) -> bool:
    """Deteksi completion yang buruk: terlalu singkat atau hanya referensi dokumen."""
    if not completion:
        return True
    c = completion.strip()
    if len(c) < 50:
        return True
    bad_patterns = [
        r'^dokumen\s*\d+\.?$',
        r'^jawaban?\s+(dapat\s+)?ditemukan\s+di\s+dokumen\s*\d+',
        r'^merujuk\s+pada\s+dokumen\s*\d+',
        r'^lihat\s+dokumen\s*\d+',
        r'^dokumen\s*\d+\s+menyebutkan\s+bahwa',
        r'sebagaimana\s+(dijelaskan|tercantum|tertera)\s+dalam\s+dokumen\s*\d+',
    ]
    for pat in bad_patterns:
        if re.match(pat, c, re.IGNORECASE):
            return True
    return False


def repair_completion(sample: dict, style: str = "langsung") -> Optional[str]:
    """Regenerate completion yang lebih baik berdasarkan instruction + dokumen relevan."""
    instruction = sample["instruction"]
    documents = sample["documents"]
    thought_process = sample.get("thought_process", "")

    # Try to find relevant doc from thought_process
    relevant_doc_num = None
    match = re.search(r'dokumen\s*(\d+)\s*\(?(?:sangat\s+relevan|relevan)', thought_process, re.IGNORECASE)
    if match:
        relevant_doc_num = int(match.group(1))

    relevant_content = ""
    if relevant_doc_num and 1 <= relevant_doc_num <= len(documents):
        relevant_content = documents[relevant_doc_num - 1]
    else:
        for i, doc in enumerate(documents, 1):
            if f"dokumen {i}" in thought_process.lower() and "(abaikan)" not in thought_process.lower().split(f"dokumen {i}")[1][:30]:
                relevant_content = doc
                relevant_doc_num = i
                break

    # Try to extract document title from thought_process or sample metadata
    doc_title_hint = ""
    title_match = re.search(r'(Peraturan Desa\s+[^\s,]+(?:\s+No\.\s*\d+)?(?:\s+Tahun\s+\d+)?)', thought_process, re.IGNORECASE)
    if title_match:
        doc_title_hint = title_match.group(1)

    style_instruction = COMPLETION_STYLES.get(style, COMPLETION_STYLES["langsung"])

    docs_text = ""
    for i, doc in enumerate(documents, 1):
        marker = " (RELEVAN)" if i == relevant_doc_num else ""
        docs_text += f"\nDokumen {i}{marker}:\n{doc[:500]}\n"

    system_prompt = (
        "Anda adalah pakar peraturan desa Indonesia. Tugas Anda: menjawab pertanyaan "
        "berdasarkan dokumen yang diberikan.\n\n"
        "ATURAN JAWABAN (SANGAT PENTING):\n"
        f"Gaya jawaban: {style_instruction}\n\n"
        "1. Jawab LANGSUNG pertanyaan dengan LENGKAP dan DESKRIPTIF (minimal 2-3 kalimat).\n"
        "2. Sertakan fakta spesifik dari dokumen: definisi, angka, syarat, nama jabatan, dsb.\n"
        "3. Jawaban harus bisa berdiri sendiri tanpa perlu membaca dokumen.\n"
        "4. JANGAN PERNAH menyebut 'Dokumen 1', 'Dokumen 2', dst. sebagai referensi!\n"
        "   Gunakan NAMA PERATURAN yang ACTUAL beserta NOMOR dan PASAL-nya.\n"
        "   Contoh BENAR: '...berdasarkan Pasal 5 Peraturan Desa Biru No. 07 Tahun 2015.'\n"
        "   Contoh SALAH: '...sebagaimana tercantum dalam Dokumen 3.'\n"
        "5. Gunakan kutipan langsung dari dokumen jika relevan, format: <<kutipan>>teks asli<<akhir kutipan>>\n"
        "6. Gunakan Bahasa Indonesia yang formal dan jelas.\n\n"
        "CONTOH JAWABAN BENAR:\n"
        '- "Kepala desa berwenang menetapkan peraturan desa setelah mendapat persetujuan '
        'dari badan permusyawaratan desa, berdasarkan Pasal 8 Peraturan Desa Biru '
        'No. 07 Tahun 2015."\n'
        '- "Masa bakti kepengurusan tim Kibbla adalah 3 (tiga) tahun, sesuai dengan '
        'Pasal 12 Peraturan Desa Biru No. 07 Tahun 2015 tentang Kesehatan Ibu, Bayi Baru Lahir, '
        'Bayi Dan Anak Balita."\n\n'
        "CONTOH JAWABAN SALAH (JANGAN lakukan):\n"
        '- "Dokumen 3" ← HANYA referensi, tidak ada jawaban\n'
        '- "Jawaban dapat ditemukan di Dokumen 1" ← tidak menjawab\n'
        '- "Dokumen 2 menyebutkan bahwa..." ← hanya memulai tanpa menyelesaikan\n\n'
        "Jawab pertanyaan berikut HANYA dengan jawaban langsung, tanpa analisis dokumen:"
    )

    user_prompt = (
        f"Pertanyaan: {instruction}\n\n"
        f"Dokumen-dokumen:{docs_text}\n\n"
        f"Berikan jawaban lengkap dan deskriptif dengan gaya '{style}':"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = call_hf_chat(messages, temperature=0.5, max_tokens=512)
    if result and len(result.strip()) >= 50:
        return result.strip()
    return None


def repair_dataset(filepath: Path, output_path: Optional[Path] = None) -> Dict:
    """Perbaiki completion yang buruk dalam dataset JSONL."""
    if output_path is None:
        output_path = filepath

    samples = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    total = len(samples)
    if total == 0:
         return {"total": total, "repaired": 0, "failed": 0}
         
    bad_indices = []
    for i, s in enumerate(samples):
        if is_bad_completion(s.get("completion", "")):
            bad_indices.append(i)

    print(f"Total sampel     : {total}")
    print(f"Completion buruk : {len(bad_indices)}")
    print(f"Completion baik  : {total - len(bad_indices)}")
    print("=" * 60)

    if not bad_indices:
        print("\n✅ Semua completion sudah baik! Tidak ada perbaikan needed.")
        return {"total": total, "repaired": 0, "failed": 0}

    repaired = 0
    failed = 0

    pbar = tqdm(bad_indices, desc="Repairing completions")
    for idx in pbar:
        sample = samples[idx]
        old_completion = sample["completion"]

        # Randomly assign a style for repair
        style = random.choice(list(COMPLETION_STYLES.keys()))
        new_completion = repair_completion(sample, style=style)
        if new_completion:
            sample["completion"] = new_completion
            repaired += 1
            pbar.set_postfix(ok=repaired, fail=failed)
            print(f"\n  [{idx+1}] BEFORE: {old_completion[:60]}")
            print(f"  [{idx+1}] AFTER : {new_completion[:80]}...")
        else:
            failed += 1
            pbar.set_postfix(ok=repaired, fail=failed)
            print(f"\n  [{idx+1}] FAILED to repair: {old_completion[:60]}")

        time.sleep(REQUEST_DELAY)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    pbar.close()
    print("\n" + "=" * 60)
    print(f"Selesai perbaikan!")
    print(f"  Diperbaiki : {repaired}")
    print(f"  Gagal      : {failed}")
    print(f"  Output     : {output_path}")

    return {"total": total, "repaired": repaired, "failed": failed}


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # --- KONFIGURASI FILE ---
    # Set ke nama file untuk proses 1 file saja, atau None untuk semua file di folder
    # Contoh: "PERATURAN_DESA_BIRU_NOMOR_07_TAHUN_2015_chunks.json"
    SINGLE_FILE = "KEPALA_DESA_KABUPATEN_BANDUNG_NOMOR_01_TAHUN_2018_T_E_N_T_A_N_G_RENCANA_PEMBANGUNAN_JANGKA_MENENGAH_DESA_RPJM_DESA_DESA_CIPEDES_KECAMATAN_PASEH_KABUPATEN_BANDUNG_chunks.json"  # Ubah ke nama file untuk proses satu file saja

    if SINGLE_FILE:
        single_path = PROCESSED_DIR / SINGLE_FILE
        if not single_path.exists():
            print(f"ERROR: File tidak ditemukan: {single_path}")
            sys.exit(1)
        print(f"Memuat file tunggal: {SINGLE_FILE}")
        all_doc_chunks = load_single_chunk_file(single_path)
    else:
        print("Memuat semua chunks...")
        all_doc_chunks = load_all_chunks(PROCESSED_DIR)
    total_chunks = sum(len(v) for v in all_doc_chunks.values())
    print(f"\nTotal: {len(all_doc_chunks)} dokumen, {total_chunks} chunks")

    # Test koneksi API
    print("\nTesting API connection...")
    test_resp = call_hf_chat(
        [{"role": "user", "content": "Katakan 'Koneksi berhasil!' dalam satu kalimat pendek."}],
        max_tokens=30, temperature=0.1
    )
    print(f"Response: {test_resp}")
    
    # Setup Output
    OUTPUT_FILENAME = "raft_perdes_dataset.jsonl"
    output_path = DATASET_DIR / OUTPUT_FILENAME

    # --- SISTEM BACKUP OTOMATIS ---
    if output_path.exists():
        # Buat timestamp berdasarkan waktu saat ini
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"raft_perdes_dataset_backup_{timestamp}.jsonl"
        backup_path = DATASET_DIR / backup_filename
        
        # Pindahkan file lama menjadi file backup
        shutil.move(str(output_path), str(backup_path))
        print(f"✅ Data sebelumnya aman! Telah dibackup ke:\n   {backup_path}")

    CHUNKS_PER_DOC = 1  # Ubah ke None untuk full dataset
    print(f"\nMode: {'TEST' if CHUNKS_PER_DOC else 'FULL'}")
    
    # 1. Generate Dataset
    raft_dataset = generate_raft_dataset(
        all_doc_chunks=all_doc_chunks,
        output_path=output_path,
        chunks_per_doc=CHUNKS_PER_DOC,
        seed=42,
    )

    # 2. Validasi Dataset
    print(f"\nMemvalidasi: {output_path}\n")
    if output_path.exists():
        val_stats = validate_raft_dataset(output_path)
        print(f"Total sampel  : {val_stats['total_samples']}")
        print(f"Valid         : {val_stats['valid']}")
        print(f"Invalid       : {val_stats['invalid']}")
        print(f"\nAvg instruction len : {val_stats['avg_instruction_len']:.0f} chars")
        print(f"Avg thought len     : {val_stats['avg_thought_len']:.0f} chars")
        print(f"Avg completion len  : {val_stats['avg_completion_len']:.0f} chars")
        print(f"Avg docs per sample : {val_stats['avg_docs_per_sample']:.1f}")

        # Diversity Metrics
        print(f"\n--- Diversity Metrics ---")
        print(f"Distinct-1 (completion) : {val_stats['distinct_1_completion']:.2f}")
        print(f"Distinct-2 (completion) : {val_stats['distinct_2_completion']:.2f}")
        print(f"Distinct-2 (thought)    : {val_stats['distinct_2_thought']:.2f}")
        print(f"Avg Jaccard similarity  : {val_stats['avg_jaccard_similarity']:.2f}")
        print(f"Near-duplicate pairs    : {val_stats['near_duplicate_pairs']}")

        total_q = val_stats['valid']
        if val_stats['question_patterns']:
            print(f"Question patterns:")
            for pattern, count in val_stats['question_patterns'].items():
                pct = (count / total_q * 100) if total_q > 0 else 0
                print(f"  - {pattern}: {count} ({pct:.0f}%)")

        if val_stats['question_pattern_warning']:
            print("⚠️  WARNING: Ada pattern pertanyaan yang mendominasi >35% dari total!")

        if val_stats["errors"]:
            print(f"\nErrors ({len(val_stats['errors'])}):")
            for err in val_stats["errors"][:10]:
                print(f"  - {err}")

        # 3. Print Sampel
        print("\n" + "=" * 80)
        print("CONTOH SAMPEL DATASET RAFT")
        print("=" * 80)
        with open(output_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                sample = json.loads(line.strip())
                
                print(f"\n{'~'*80}")
                print(f"SAMPEL #{i+1}")
                print(f"{'~'*80}")
                print(f"\nINSTRUCTION:\n   {sample['instruction']}")
                
                print(f"\nDOCUMENTS ({len(sample['documents'])} docs):")
                for j, doc in enumerate(sample['documents'], 1):
                    preview = doc[:150].replace('\n', ' ')
                    print(f"   Doc {j}: {preview}...")
                
                print(f"\nTHOUGHT PROCESS:\n   {sample['thought_process'][:300]}...")
                print(f"\nCOMPLETION:\n   {sample['completion']}\n")

        # 4. Repair Dataset
        print("\nMenjalankan proses repair completion...")
        repair_result = repair_dataset(output_path)
    else:
        print("Dataset tidak ditemukan. Proses dibatalkan.")
