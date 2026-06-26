"""
RAFT Dataset Generator v4 — Peraturan Desa (Production-Grade)
=============================================================
Pembaruan V4:
1. Completion Natural (Kutipan hanya ada di thought_process)
2. Reasoning-based Question (Menggantikan interpretatif)
3. Answerability Check sebelum men-generate distractors
4. Hard Negatives (Similarity 0.75 - 0.95)
5. Distribusi dominan Pasal & Ayat Lookup
"""

import os, sys, json, time, random, re, shutil, pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_openai import OpenAIEmbeddings

import requests
import numpy as np
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
CACHE_DIR        = PROJECT_ROOT / "data" / "cache"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Parameter Generation
MAX_TOKENS           = 1536
REQUEST_DELAY        = 1.5
NUM_DISTRACTORS      = 4
ORACLE_PRESENT_RATIO = 0.80   
MULTIHOP_RATIO       = 0.10   # Diubah menjadi 10% sesuai arahan 
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")

print(f"Memuat model embedding ({EMBEDDING_MODEL_NAME}) via OpenAI API...")
embedder = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1")
)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI GAYA & TIPE PERTANYAAN
# ─────────────────────────────────────────────────────────────────────────────

QUERY_STYLES = {
    "natural": {"weight": 0.60, "instruction": "Tulis pertanyaan secara natural layaknya warga masyarakat bertanya pada sistem. JANGAN menyebut kata 'Pasal', 'Ayat'."},
    "semi_specific": {"weight": 0.25, "instruction": "Sebutkan konteks peraturan secara umum tanpa menyebut nomor pasal spesifik."},
    "legal": {"weight": 0.15, "instruction": "Secara eksplisit sebutkan nama peraturan dan/atau nomor pasal/ayat."}
}

QUESTION_TYPES = {
    "faktual": {"weight": 15, "prompt": "Buat pertanyaan FAKTUAL tentang angka, batas waktu, syarat spesifik.", "compatible_styles": ["langsung"]},
    "definisional": {"weight": 10, "prompt": "Buat pertanyaan DEFINISIONAL tentang makna istilah.", "compatible_styles": ["penjelasan"]},
    "prosedural": {"weight": 15, "prompt": "Buat pertanyaan PROSEDURAL tentang langkah-langkah mekanisme.", "compatible_styles": ["terstruktur"]},
    "kondisional": {"weight": 10, "prompt": "Buat pertanyaan KONDISIONAL tentang apa yang terjadi jika syarat terpenuhi/dilanggar.", "compatible_styles": ["langsung"]},
    "enumeratif": {"weight": 10, "prompt": "Buat pertanyaan ENUMERATIF meminta daftar lengkap (kewenangan/larangan).", "compatible_styles": ["terstruktur"]},
    "pasal_lookup": {"weight": 20, "prompt": "Buat pertanyaan LOOKUP yang langsung meminta substansi suatu pasal spesifik. Wajib sebut pasal.", "compatible_styles": ["formal_hukum"]},
    "ayat_lookup": {"weight": 10, "prompt": "Buat pertanyaan LOOKUP spesifik tentang maksud dari ayat tertentu dalam pasal.", "compatible_styles": ["formal_hukum"]},
    "komparatif": {"weight": 5, "prompt": "Buat pertanyaan KOMPARATIF membandingkan dua hal (jika ada).", "compatible_styles": ["terstruktur"]},
    "reasoning_based": {"weight": 5, "prompt": "Buat studi kasus/skenario BAGAIMANA JIKA (what-if) dari kondisi di teks. Jangan tanya 'mengapa' kecuali tertulis.", "compatible_styles": ["percakapan", "penjelasan"]}
}

COMPLETION_STYLES = {
    "langsung": "Jawab langsung dalam 1-2 kalimat. Fokus pada inti jawaban, natural tanpa mengutip pasal secara kaku.",
    "penjelasan": "Berikan konteks dengan bahasa yang mudah dipahami warga. Natural, minimal 2 kalimat.",
    "terstruktur": "Gunakan poin-poin agar mudah dibaca. Tidak perlu menyertakan kutipan dokumen asli.",
    "percakapan": "Bahasa santai tapi tetap akurat. Jangan menyebut 'Menurut dokumen X'.",
    "formal_hukum": "Gunakan bahasa baku. Jawab intinya saja secara profesional tanpa melampirkan *raw quote*.",
}

# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT & UTILS
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages: List[Dict], temperature: float = 0.7, retries: int = 3) -> Optional[str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GENERATOR_MODEL, "messages": messages, "max_tokens": MAX_TOKENS,
        "temperature": temperature, "top_p": 0.9, "frequency_penalty": 0.3
    }
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            wait = (attempt + 1) * 5
            time.sleep(wait)
    return None

def clean_content(text: str) -> str:
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()

def is_substantive(content: str, min_chars: int = 200) -> bool:
    c = clean_content(content)
    if len(c) < min_chars: return False
    sentences = [s.strip() for s in re.split(r'[.!?]', c) if len(s.strip()) > 10]
    return len(sentences) >= 2

def load_and_embed_chunks(processed_dir: Path) -> Dict[str, List[Dict]]:
    cache_file = CACHE_DIR / "chunk_embeddings.pkl"
    if cache_file.exists():
        print(f"Memuat embedding dari cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    all_chunks = {}
    texts_to_embed = []
    chunk_refs = []
    
    for fpath in sorted(processed_dir.glob("*_chunks.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
        all_chunks[doc_id] = chunks
        
        for c in chunks:
            texts_to_embed.append(clean_content(c['content']))
            chunk_refs.append(c)
            
    BATCH_SIZE = 64
    print(f"Mengekstraksi embedding untuk {len(texts_to_embed)} chunk via OpenAI API...")
    all_embeddings = []
    batches = [texts_to_embed[i:i+BATCH_SIZE] for i in range(0, len(texts_to_embed), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Embedding batches"):
        for attempt in range(3):
            try:
                vecs = embedder.embed_documents(batch)
                all_embeddings.extend([np.array(v) for v in vecs])
                break
            except Exception as e:
                wait = (attempt + 1) * 5
                print(f"  Retry embedding batch, tunggu {wait}s... ({e})")
                time.sleep(wait)
        else:
            print(f"Gagal mendapatkan embedding untuk batch. Hentikan program.")
            sys.exit(1)
    
    for c, emb in zip(chunk_refs, all_embeddings):
        norm = np.linalg.norm(emb)
        c['embedding'] = emb / norm if norm > 0 else emb
        
    print(f"Menyimpan embedding ke cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(all_chunks, f)
        
    return all_chunks

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_doc_label(chunk: Dict) -> str:
    meta = chunk.get("metadata", {})
    pasal = chunk.get("pasal", "") or meta.get("section", "")
    return f"{meta.get('title', 'Dokumen')}, {pasal.title()}" if pasal else meta.get('title', 'Dokumen')

def select_semantic_distractors(
    oracle_chunks: List[Dict],
    all_doc_chunks: Dict[str, List[Dict]],
    n: int = NUM_DISTRACTORS,
    negative_type: str = "none"
) -> List[Dict]:
    oracle_emb = np.mean([c['embedding'] for c in oracle_chunks], axis=0)
    oracle_emb = oracle_emb / np.linalg.norm(oracle_emb)
    
    candidates = []
    for doc_id, chunks in all_doc_chunks.items():
        for c in chunks:
            if c in oracle_chunks: continue
            sim = np.dot(oracle_emb, c['embedding'])
            candidates.append((sim, c))
            
    # Penambahan Hard Negative
    if negative_type == "hard_negative":
        valid = [c for s, c in candidates if 0.75 <= s <= 0.95]
    elif negative_type == "near_miss":
        valid = [c for s, c in candidates if 0.60 <= s < 0.75]
    elif negative_type == "completely_absent":
        valid = [c for s, c in candidates if s < 0.40]
    else: 
        valid = [c for s, c in candidates if 0.40 <= s <= 0.70]
        
    if len(valid) < n:
        candidates.sort(key=lambda x: x[0], reverse=(negative_type != "completely_absent"))
        valid = [c for s, c in candidates]
        
    top_pool = valid[:max(n * 3, 10)]
    random.shuffle(top_pool)
    return top_pool[:n]

def generate_question(oracle_chunks: List[Dict], question_type: str, query_style: str) -> Optional[str]:
    oracle_content = "\n\n".join([clean_content(c["content"]) for c in oracle_chunks])
    doc_titles = list({c["metadata"].get("title", "") for c in oracle_chunks})
    
    qcfg = QUESTION_TYPES[question_type]
    style_inst = QUERY_STYLES["legal"]["instruction"] if question_type in ["pasal_lookup", "ayat_lookup"] else QUERY_STYLES[query_style]["instruction"]
    ref_rule = "3. JANGAN sebut nama peraturan." if query_style == "natural" and question_type not in ["pasal_lookup", "ayat_lookup"] else f"3. Referensi Nama Peraturan: {', '.join(doc_titles)}"
        
    system = (
        "Anda adalah asisten data RAG hukum desa.\n"
        f"TIPE: {qcfg['prompt']}\n"
        f"GAYA: {style_inst}\n\n"
        "ATURAN KETAT:\n"
        "1. Pertanyaan harus dapat dijawab 100% menggunakan konteks.\n"
        "2. Hanya output 1 kalimat tanya.\n"
        f"{ref_rule}"
    )
    
    q = call_llm([{"role": "system", "content": system}, {"role": "user", "content": f"Konteks:\n{oracle_content}\n\nPertanyaan:"}], temperature=0.85)
    return re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*|Q[:\s]*)", "", q, flags=re.IGNORECASE).strip(' "\'') if q else None

# Validasi Answerability (Masalah #3)
def check_answerability(question: str, oracle_chunks: List[Dict]) -> bool:
    oracle_content = "\n\n".join([clean_content(c["content"]) for c in oracle_chunks])
    system = "Evaluator RAG. Jawab 'YA' jika pertanyaan BISA DIJAWAB SEPENUHNYA dengan dokumen. Jawab 'TIDAK' jika tidak bisa."
    user = f"Pertanyaan: {question}\n\nDokumen:\n{oracle_content}"
    ans = call_llm([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
    return "YA" in (ans or "").upper()

def generate_thought_and_completion(
    question: str, documents: List[str], doc_labels: List[str], oracle_indices: List[int],
    question_type: str, style: str, oracle_present: bool, negative_type: str
) -> Tuple[Optional[str], Optional[str]]:

    docs_formatted = "".join([f"\n--- Dokumen {i+1}: {l} ---\n{d}\n" for i, (d, l) in enumerate(zip(documents, doc_labels))])
    
    if oracle_present:
        oracle_hint = f"\n[INTERNAL] Jawaban ada pada Dokumen {[i+1 for i in oracle_indices]}.\n"
        thought_inst = (
            "Di dalam thought_process:\n"
            "• Identifikasi dokumen relevan dan WAJIB letakkan kutipan persis dengan format <<kutipan teks asli>>.\n"
            "• Jelaskan logikanya."
        )
        comp_inst = (
            "Di dalam completion:\n"
            f"• Gaya: {COMPLETION_STYLES[style]}\n"
            "• Jawab langsung dan NATURAL.\n"
            "• DILARANG KERAS menyertakan kutipan (<<...>>), DILARANG menyebut 'Menurut Dokumen X' atau 'Pasal Y' secara kaku jika tidak diminta."
        )
    else:
        oracle_hint = "\n[INTERNAL] Jawaban TIDAK ADA di dokumen mana pun.\n"
        thought_inst = "Di thought_process: Buktikan ketiadaan jawaban. Analisis batas kemiripan jika ada dokumen jebakan."
        comp_inst = "Di completion: Nyatakan informasi tidak tersedia secara natural tanpa mengarang fakta."

    system = (
        "Output HARUS JSON Valid:\n"
        '{\n  "thought_process": "...",\n  "completion": "..."\n}\n\n'
        f"=== ATURAN THOUGHT ===\n{thought_inst}\n\n"
        f"=== ATURAN COMPLETION ===\n{comp_inst}"
    )

    result = call_llm([{"role": "system", "content": system}, {"role": "user", "content": f"Pertanyaan: {question}\n\nDokumen:\n{docs_formatted}{oracle_hint}"}], temperature=0.75)
    
    if not result: return None, None
    for attempt in [result, re.search(r'\{[\s\S]*\}', result)]:
        try:
            parsed = json.loads(attempt if isinstance(attempt, str) else (attempt.group() if attempt else None))
            thought = (parsed.get("thought_process") or "").strip()
            completion = (parsed.get("completion") or "").strip()
            if len(thought) > 20 and len(completion) > 10: return thought, completion
        except: continue
    return None, None

def is_bad_sample(sample: Dict) -> Tuple[bool, str]:
    comp, thought = sample["completion"], sample["thought_process"]
    if re.search(r'\bDokumen\s+\d+\b', comp): return True, "Bocor menyebut 'Dokumen N'"
    
    # Validasi kutipan kini HANYA melihat thought_process
    if sample["metadata_extra"]["oracle_present"]:
        if "<<" in comp or ">>" in comp: return True, "Completion tidak natural (terdapat raw quote)"
        quotes = re.findall(r"<<(.+?)>>", thought)
        if not quotes: return True, "Tidak ada bukti/kutipan (<<...>>) di thought_process"
        
    return False, ""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_raft_dataset(all_doc_chunks: Dict, output_path: Path):
    random.seed(42)
    dataset, failed, skipped, unanswerable = [], 0, 0, 0
    
    # DAFTAR DOKUMEN TIDAK BERLAKU / HANYA UNTUK DISTRAKTOR
    # Masukkan document_id dari PDF yang sudah tidak berlaku di sini.
    # Contoh: "perdes_biru_10_2016"
    INVALID_DOC_IDS = [
        "perdes_biru_10_2016", # Ganti dengan document_id PDF yang sebenarnya tidak berlaku
    ]
    
    # Kumpulkan chunk substantif HANYA dari dokumen yang valid
    all_oracle_candidates = []
    for doc_id, chunks in all_doc_chunks.items():
        if doc_id in INVALID_DOC_IDS:
            print(f"Mengabaikan {doc_id} sebagai sumber pertanyaan (Hanya dipakai sebagai Distraktor).")
            continue
            
        all_oracle_candidates.extend([(doc_id, c) for c in chunks if is_substantive(c["content"])])
        
    print(f"Total Chunk Substantif (Valid): {len(all_oracle_candidates)}\n")
    pbar = tqdm(all_oracle_candidates, desc="Generating RAFT")

    for doc_id, chunk in pbar:
        is_multihop = random.random() < MULTIHOP_RATIO
        oracle_chunks = [chunk]
        if is_multihop:
            valid_pairs = [c for d, c in all_oracle_candidates if d == doc_id and c != chunk and 0.40 <= np.dot(chunk['embedding'], c['embedding']) <= 0.80]
            if valid_pairs: oracle_chunks.append(random.choice(valid_pairs))
            else: is_multihop = False
        
        q_types = list(QUESTION_TYPES.keys())
        q_weights = [v["weight"] for v in QUESTION_TYPES.values()]
        q_type  = random.choices(q_types, q_weights)[0]
        q_style = random.choices(list(QUERY_STYLES.keys()), [v["weight"] for v in QUERY_STYLES.values()])[0]
        
        question = generate_question(oracle_chunks, q_type, q_style)
        if not question: failed += 1; time.sleep(REQUEST_DELAY); continue
        
        # Validasi Oracle Present (Masalah #3)
        if not check_answerability(question, oracle_chunks):
            unanswerable += 1; time.sleep(REQUEST_DELAY); continue
            
        oracle_present = random.random() < ORACLE_PRESENT_RATIO
        negative_type = "none"
        
        if oracle_present:
            pool = select_semantic_distractors(oracle_chunks, all_doc_chunks, n=NUM_DISTRACTORS) + oracle_chunks
            random.shuffle(pool)
            documents, doc_labels, oracle_indices = [clean_content(c["content"]) for c in pool], [get_doc_label(c) for c in pool], [pool.index(c) for c in oracle_chunks]
        else:
            negative_type = random.choice(["hard_negative", "near_miss", "completely_absent"])
            distractors = select_semantic_distractors(oracle_chunks, all_doc_chunks, n=NUM_DISTRACTORS+len(oracle_chunks), negative_type=negative_type)
            documents, doc_labels, oracle_indices = [clean_content(c["content"]) for c in distractors], [get_doc_label(c) for c in distractors], []

        comp_style = random.choice(QUESTION_TYPES[q_type]["compatible_styles"])
        thought, completion = generate_thought_and_completion(question, documents, doc_labels, oracle_indices, q_type, comp_style, oracle_present, negative_type)
        if not thought or not completion: failed += 1; time.sleep(REQUEST_DELAY); continue

        sample = {
            "instruction": question, "documents": documents, "thought_process": thought, "completion": completion,
            "metadata_extra": {"query_style": q_style, "question_type": q_type, "multi_hop": is_multihop, "oracle_present": oracle_present, "negative_type": negative_type, "style": comp_style, "evidence_docs": [i + 1 for i in oracle_indices] if oracle_present else []}
        }

        bad, reason = is_bad_sample(sample)
        if bad: skipped += 1; time.sleep(REQUEST_DELAY); continue

        dataset.append(sample)
        with open(output_path, "a", encoding="utf-8") as f: f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        pbar.set_postfix(ok=len(dataset), unans=unanswerable, skip=skipped)
        time.sleep(REQUEST_DELAY)

    pbar.close()
    print(f"Selesai! Berhasil: {len(dataset)} | Dibuang (Unanswerable): {unanswerable} | Gagal: {failed} | Skip (Quality): {skipped}")

if __name__ == "__main__":
    # 1. Pastikan direktori tersedia
    if not PROCESSED_DIR.exists():
        print(f"Error: Folder {PROCESSED_DIR} tidak ditemukan. Pastikan pipeline chunking sudah dijalankan.")
        sys.exit(1)
        
    # 2. Muat dan embed chunk dokumen
    all_chunks = load_and_embed_chunks(PROCESSED_DIR)
    
    # 3. Setup output file
    OUTPUT_FILENAME = "raft_dataset_v4_production.jsonl"
    output_path = DATASET_DIR / OUTPUT_FILENAME
    
    # 4. Backup jika file sebelumnya ada
    if output_path.exists():
        backup = DATASET_DIR / f"raft_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.move(str(output_path), str(backup))
        print(f"File versi sebelumnya dibackup ke {backup}")

    # 5. Jalankan Generator
    generate_raft_dataset(all_chunks, output_path)