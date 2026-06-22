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
NUM_DISTRACTORS = 2  # jumlah distractor per sampel

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
) -> List[str]:
    """Pilih n distractor documents dari dokumen BERBEDA."""
    oracle_doc_id = oracle_chunk["metadata"]["document_id"]
    
    candidates = []
    for doc_id, chunks in all_doc_chunks.items():
        if doc_id != oracle_doc_id:
            for c in chunks:
                candidates.append(c)
    
    if not candidates:
        # Fallback: chunk lain dari dokumen sama
        for c in all_doc_chunks[oracle_doc_id]:
            if c["chunk_index"] != oracle_chunk["chunk_index"]:
                candidates.append(c)
    
    selected = random.sample(candidates, min(n, len(candidates)))
    return [clean_chunk_content(c["content"]) for c in selected]

def generate_question_from_chunk(oracle_content: str, doc_title: str) -> Optional[str]:
    """Gunakan LLM untuk generate pertanyaan dari oracle_content."""
    system_prompt = (
        "Anda adalah pakar pembuat dataset untuk fine-tuning model RAG "
        "(Retrieval Augmented Generation) khusus domain peraturan desa Indonesia.\n\n"
        "Tugas: Buat SATU pertanyaan dalam Bahasa Indonesia berdasarkan potongan dokumen berikut.\n\n"
        "Aturan:\n"
        "1. Pertanyaan HARUS bisa dijawab HANYA dari isi dokumen.\n"
        "2. Pertanyaan harus spesifik, bukan pertanyaan umum.\n"
        "3. Variasi format: 'Berdasarkan dokumen...', 'Menurut peraturan...', "
        "'Apa syarat...', 'Siapa yang bertanggung jawab...', 'Berapa jumlah...', "
        "'Jelaskan ketentuan...'.\n"
        "4. Pertanyaan menantang, butuh pemahaman detail.\n"
        "5. Output HANYA pertanyaan.\n\n"
        f"Dokumen: {doc_title}\n\n"
        f"Isi:\n{oracle_content}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Buat satu pertanyaan berdasarkan isi dokumen di atas."}
    ]
    
    result = call_hf_chat(messages, temperature=0.8, max_tokens=200)
    if result:
        q = result.strip().strip('"').strip("'").strip()
        q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*|Q[:\s]*)", "", q, flags=re.IGNORECASE).strip()
        return q
    return None

def generate_thought_and_completion(
    instruction: str,
    documents: List[str],
    oracle_index: int,
    oracle_content: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate thought_process dan completion menggunakan LLM."""
    docs_formatted = ""
    for i, doc in enumerate(documents, 1):
        docs_formatted += f"\n--- Dokumen {i} ---\n{doc}\n"

    system_prompt = (
        "Anda adalah model AI yang sedang di-fine-tune dengan metode RAFT "
        "(Retrieval Augmented Fine-Tuning) untuk domain peraturan desa Indonesia.\n\n"
        "Anda akan menerima:\n"
        "1. Sebuah pertanyaan/instruksi.\n"
        "2. Beberapa dokumen (sebagian relevan, sebagian pengecoh).\n\n"
        "Anda HARUS menghasilkan output JSON valid dengan 2 field:\n\n"
        '```json\n'
        '{\n'
        '  "thought_process": "Langkah analisis: evaluasi setiap dokumen, '
        '"tentukan mana relevan dan mana distractor, '
        '"jelaskan alasan, tarik kesimpulan.",\n'
        '  "completion": "Jawaban LENGKAP dan DESKRIPTIF yang langsung menjawab pertanyaan, '
        '"bukan sekadar referensi dokumen."\n'
        '}\n'
        '```\n\n'
        "ATURAN thought_process:\n"
        "- Analisis SETIAP dokumen (Dokumen 1, 2, dst).\n"
        "- Tandai tidak relevan sebagai '(Abaikan)'.\n"
        "- Tandai dokumen kunci sebagai '(Sangat Relevan)'.\n\n"
        "ATURAN completion (SANGAT PENTING):\n"
        "- completion HARUS berisi JAWABAN LENGKAP dari pertanyaan, BUKAN hanya 'Dokumen X'.\n"
        "- Jawaban harus bisa berdiri sendiri tanpa perlu membaca thought_process.\n"
        "- Sertakan detail/fakta spesifik dari dokumen relevan (definisi, angka, syarat, dsb).\n"
        "- Sebutkan sumber dokumen di AKHIR kalimat (contoh: '...sebagaimana tercantum dalam Dokumen 2.').\n"
        "- Minimal 2-3 kalimat yang informatif.\n\n"
        "CONTOH BENAR completion:\n"
        '- "Bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari, '
        'sebagaimana dijelaskan dalam Dokumen 1."\n'
        '- "Kepala desa berwenang membuat keputusan sebagai tindak lanjut '
        'peraturan desa, merujuk pada Dokumen 3."\n\n'
        "CONTOH SALAH (JANGAN lakukan ini):\n"
        '- "Dokumen 1"  ← terlalu singkat, tidak ada jawaban\n'
        '- "Dokumen 3 menyebutkan bahwa..."  ← hanya referensi tanpa jawaban langsung\n'
    )

    user_prompt = (
        f"Pertanyaan: {instruction}\n\n"
        f"Dokumen-dokumen:\n{docs_formatted}\n\n"
        f"Analisis semua dokumen dan berikan jawaban lengkap. Output JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = call_hf_chat(messages, temperature=0.6, max_tokens=1024)
    if not result:
        return None, None

    # Parse JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            thought = parsed.get("thought_process")
            completion = parsed.get("completion")
            if completion and len(completion.strip()) < 50:
                print(f"  [WARN] completion terlalu singkat ({len(completion)} chars), retry...")
                return None, None
            return thought, completion
    except json.JSONDecodeError:
        pass

    try:
        parsed = json.loads(result)
        thought = parsed.get("thought_process")
        completion = parsed.get("completion")
        if completion and len(completion.strip()) < 50:
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
            
            # Step 1: Generate question
            question = generate_question_from_chunk(oracle_content, doc_title)
            if not question:
                failed += 1
                pbar.update(1)
                continue
            
            time.sleep(REQUEST_DELAY)
            
            # Step 2: Select distractors
            distractors = select_distractors(chunk, all_doc_chunks, n=NUM_DISTRACTORS)
            
            # Step 3: Arrange - oracle di posisi acak
            oracle_pos = random.randint(0, NUM_DISTRACTORS)
            documents = distractors[:oracle_pos] + [oracle_content] + distractors[oracle_pos:]
            
            # Step 4: Generate thought_process & completion
            thought, completion = generate_thought_and_completion(
                instruction=question,
                documents=documents,
                oracle_index=oracle_pos,
                oracle_content=oracle_content,
            )
            
            if not thought or not completion:
                failed += 1
                pbar.update(1)
                time.sleep(REQUEST_DELAY)
                continue
            
            # Step 5: Format & save
            sample = {
                "instruction": question,
                "documents": documents,
                "thought_process": thought,
                "completion": completion,
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

def validate_raft_dataset(filepath: Path) -> Dict:
    """Validasi dataset RAFT."""
    stats = {
        "total_samples": 0, "valid": 0, "invalid": 0, "errors": [],
        "avg_instruction_len": 0, "avg_thought_len": 0,
        "avg_completion_len": 0, "avg_docs_per_sample": 0,
    }

    instruction_lens, thought_lens, completion_lens, docs_counts = [], [], [], []
    required_keys = {"instruction", "documents", "thought_process", "completion"}

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
            except json.JSONDecodeError as e:
                stats["errors"].append(f"Line {line_no}: JSON error - {e}")
                stats["invalid"] += 1

    if instruction_lens:
        stats["avg_instruction_len"] = sum(instruction_lens) / len(instruction_lens)
        stats["avg_thought_len"] = sum(thought_lens) / len(thought_lens)
        stats["avg_completion_len"] = sum(completion_lens) / len(completion_lens)
        stats["avg_docs_per_sample"] = sum(docs_counts) / len(docs_counts)

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
    ]
    for pat in bad_patterns:
        if re.match(pat, c, re.IGNORECASE):
            return True
    return False

def repair_completion(sample: dict) -> Optional[str]:
    """Regenerate completion yang lebih baik berdasarkan instruction + dokumen relevan."""
    instruction = sample["instruction"]
    documents = sample["documents"]
    thought_process = sample.get("thought_process", "")

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

    docs_text = ""
    for i, doc in enumerate(documents, 1):
        marker = " (RELEVAN)" if i == relevant_doc_num else ""
        docs_text += f"\nDokumen {i}{marker}:\n{doc[:500]}\n"

    system_prompt = (
        "Anda adalah pakar peraturan desa Indonesia. Tugas Anda: menjawab pertanyaan "
        "berdasarkan dokumen yang diberikan.\n\n"
        "ATURAN JAWABAN (SANGAT PENTING):\n"
        "1. Jawab LANGSUNG pertanyaan dengan LENGKAP dan DESKRIPTIF (minimal 2-3 kalimat).\n"
        "2. Sertakan fakta spesifik dari dokumen: definisi, angka, syarat, nama jabatan, dsb.\n"
        "3. Jawaban harus bisa berdiri sendiri tanpa perlu membaca dokumen.\n"
        "4. Sebutkan sumber dokumen di AKHIR kalimat dengan format '(Dokumen X)'.\n"
        "5. Gunakan Bahasa Indonesia yang formal dan jelas.\n\n"
        "CONTOH JAWABAN BENAR:\n"
        '- "Kepala desa berwenang menetapkan peraturan desa setelah mendapat persetujuan '
        'dari badan permusyawaratan desa, sebagaimana tercantum dalam Dokumen 3."\n'
        '- "Masa bakti kepengurusan tim Kibbla adalah 3 (tiga) tahun, sesuai dengan '
        'ketentuan dalam Dokumen 3."\n\n'
        "CONTOH JAWABAN SALAH (JANGAN lakukan):\n"
        '- "Dokumen 3" ← HANYA referensi, tidak ada jawaban\n'
        '- "Jawaban dapat ditemukan di Dokumen 1" ← tidak menjawab\n'
        '- "Dokumen 2 menyebutkan bahwa..." ← hanya memulai tanpa menyelesaikan\n\n'
        "Jawab pertanyaan berikut HANYA dengan jawaban langsung, tanpa analisis dokumen:"
    )

    user_prompt = (
        f"Pertanyaan: {instruction}\n\n"
        f"Dokumen-dokumen:{docs_text}\n\n"
        f"Berikan jawaban lengkap dan deskriptif:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = call_hf_chat(messages, temperature=0.4, max_tokens=512)
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

        new_completion = repair_completion(sample)
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