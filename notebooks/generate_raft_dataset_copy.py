"""
RAFT Dataset Generator — Retrieval-Grounded (V2)
=================================================
Pendekatan baru yang menyelaraskan distribusi training dengan distribusi inference.

Pipeline (self-contained, TIDAK butuh Flask backend):
  Question → Supabase Vector DB (Top-20) → CrossEncoder Re-ranking (Top-5)
  Question + Top-5 → LLM (System Prompt RAFT) → JSON RAFT Entry

Setiap entry berisi:
  - instruction: pertanyaan asli
  - documents: 5 dokumen hasil retrieval (apa adanya)
  - thought_process: analisis per-dokumen + ringkasan sintesis
  - completion: jawaban akhir yang 100% grounded pada documents
"""

import os, sys, json, time, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SETUP & KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# LLM API Config (via Maia Router / OpenAI-compatible endpoint)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
LLM_API_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

# Model & Generation Settings
GENERATOR_MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.001  # detik delay antar request untuk menghindari rate-limit

# Retrieval Settings
INITIAL_K = 10   # Jumlah dokumen awal dari vector search
FINAL_K = 5      # Jumlah dokumen final setelah re-ranking
CONFIDENCE_THRESHOLD = -5.0  # Jika top_score di bawah ini, dokumen dianggap tidak relevan dan di-skip

# Output
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# INISIALISASI RETRIEVAL PIPELINE (Supabase + Embeddings + CrossEncoder)
# ─────────────────────────────────────────────────────────────────────────────

from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from sentence_transformers import CrossEncoder

# Supabase
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase_table = os.getenv("SUPABASE_TABLE_NAME", "documents")
supabase: Client = create_client(supabase_url, supabase_key)

# Embeddings (same model as production RAG)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    base_url=os.getenv("OPENAI_BASE_URL", "")
)

# Cross-Encoder untuk Re-ranking (same model as production RAG)
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
print(f"Memuat Cross-Encoder ({RERANKER_MODEL_NAME})...")
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
print("Cross-Encoder berhasil dimuat!")


# ─────────────────────────────────────────────────────────────────────────────
# DAFTAR PERTANYAAN
# ─────────────────────────────────────────────────────────────────────────────
# Tambahkan pertanyaan di sini. Setiap pertanyaan akan di-retrieve secara real
# melalui pipeline RAG (Vector DB → Top-20 → Reranking → Top-5).
# Pengaturan Batasan Jumlah Pertanyaan dihapus agar hanya menggunakan daftar di bawah ini

QUESTIONS: List[str] = [
    # ── Contoh pertanyaan — ganti/tambah sesuai kebutuhan ──
    "Apa sih yang dimaksud dengan pemerintahan desa di desa Mekarrahayu?",
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT RAFT
# ─────────────────────────────────────────────────────────────────────────────

RAFT_SYSTEM_PROMPT = """Anda adalah AI yang bertugas membuat dataset untuk Retrieval-Augmented Fine-Tuning (RAFT).

Tujuan Anda adalah menghasilkan dataset yang sepenuhnya grounded pada hasil retrieval sehingga distribusi data saat training sama dengan distribusi saat inference.

Question telah digunakan untuk melakukan retrieval pada Vector Database dengan pipeline:

Question
      │
      ▼
Vector Database
      │
      ▼
Top-20 Retrieval
      │
      ▼
Re-ranking
      │
      ▼
Top-5 Documents

Documents yang diberikan merupakan hasil Top-5 setelah proses re-ranking dan merupakan satu-satunya sumber informasi (Single Source of Truth).

Tugas

Buat satu data RAFT yang terdiri dari:

instruction
documents
thought_process
completion

Instruction

Gunakan pertanyaan yang diberikan.

Jangan mengubah isi pertanyaan.

Documents

Gunakan seluruh Top-5 Documents sebagaimana diberikan.

Jangan mengubah isi dokumen.

Jangan menghapus dokumen.

Jangan menambahkan dokumen.

Thought Process

Thought Process bertujuan menjelaskan bagaimana jawaban diperoleh dari seluruh hasil retrieval.

Lakukan analisis terhadap setiap dokumen.

Untuk setiap dokumen jelaskan secara singkat:

apakah relevan,
informasi penting yang ditemukan,
kontribusinya terhadap jawaban,
apakah melengkapi dokumen lain,
apakah tidak relevan.

Setelah itu buat ringkasan sintesis seluruh dokumen.

Jangan membuat jawaban akhir pada bagian ini.

Completion

Completion merupakan jawaban akhir.

Aturan:

Gunakan HANYA informasi yang terdapat pada Documents.
Jangan menggunakan pengetahuan internal model.
Jangan mengarang.
Jangan menambahkan fakta baru.
Jangan membuat asumsi.
Jangan menggunakan informasi di luar Documents.
Sintesis informasi relevan dari Top-5 Documents HANYA JIKA dokumen-dokumen tersebut membahas subjek yang sama.
PENTING - SANGAT KETAT (ANTI CONTEXT-MIXING): Jika instruksi/pertanyaan secara eksplisit menyebutkan NOMOR PERATURAN tertentu (misal: No. 10 Tahun 2016), maka dokumen yang memiliki Nomor Peraturan BERBEDA (misal: No. 05 Tahun 2016) ADALAH SAMPAH (TIDAK RELEVAN) dan WAJIB DIABAIKAN. Tolak dokumen tersebut secara eksplisit di thought_process dan JANGAN PERNAH memasukkan isinya ke dalam completion! Anda HANYA boleh menggunakan dokumen yang persis sesuai dengan spesifikasi (Nama Desa DAN Nomor Peraturan).
Apabila terdapat konflik, gunakan informasi yang paling konsisten dengan mayoritas dokumen (kecuali pertanyaan mensyaratkan dokumen spesifik).
Apabila tidak ada informasi yang menjawab pertanyaan, jawab:

Informasi tidak ditemukan pada dokumen yang diberikan.

Yang Harus Dihindari
Halusinasi
Pengetahuan internal model
Informasi di luar Documents
Asumsi
Opini pribadi
Fakta yang tidak dapat ditelusuri ke Documents

Format Output

Output HARUS berupa SATU JSON VALID.

Jangan menggunakan Markdown.

Jangan menggunakan ```json.

Jangan memberikan penjelasan.

Jangan memberikan teks selain JSON.

Format yang harus dihasilkan adalah:

{
  "instruction": "<question>",
  "documents": [
    "<document 1>",
    "<document 2>",
    "<document 3>",
    "<document 4>",
    "<document 5>"
  ],
  "thought_process": {
    "document_analysis": [
      {
        "document": 1,
        "analysis": "..."
      },
      {
        "document": 2,
        "analysis": "..."
      },
      {
        "document": 3,
        "analysis": "..."
      },
      {
        "document": 4,
        "analysis": "..."
      },
      {
        "document": 5,
        "analysis": "..."
      }
    ],
    "summary": "Ringkasan sintesis seluruh dokumen."
  },
  "completion": "<jawaban akhir>"
}

Pastikan JSON valid sehingga setiap output dapat langsung disimpan sebagai satu baris (.jsonl) tanpa proses konversi tambahan."""


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI RETRIEVAL (Self-contained — TIDAK butuh Flask backend)
# ─────────────────────────────────────────────────────────────────────────────

def rerank_documents(query: str, documents: list, top_k: int = FINAL_K) -> Tuple[list, float]:
    """
    Re-ranking menggunakan MS Marco Cross-Encoder.
    Identik dengan app/services/reranker_service.py.
    
    Returns:
        Tuple (reranked_docs, top_score)
    """
    if not documents:
        return [], 0.0
    
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    reranked = [doc for doc, _ in scored_docs[:top_k]]
    top_score = float(scored_docs[0][1]) if scored_docs else 0.0
    
    # Log top scores
    for rank, (doc, score) in enumerate(scored_docs[:top_k]):
        src = doc.metadata.get("source", "?")
        print(f"    Rank {rank+1} | Score: {score:.4f} | {src}")
    
    return reranked, top_score


def retrieve_top5_documents(question: str) -> Optional[List[str]]:
    """
    Pipeline retrieval self-contained:
      1. Vector Search pada Supabase (K=20)
      2. Re-ranking via CrossEncoder (K=5)
    
    Menggunakan komponen yang IDENTIK dengan production RAG pipeline
    (app/services/rag_service.py + app/services/reranker_service.py).
    
    Returns:
        List of 5 document content strings, atau None jika gagal.
    """
    try:
        # 1. Setup Vector Store
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name=supabase_table,
            query_name="match_documents"
        )
        
        # 2. Vector Search — Top-20
        initial_docs = vector_store.similarity_search(question, k=INITIAL_K)
        
        if not initial_docs:
            print(f"  [WARN] Vector search tidak menemukan dokumen untuk: \"{question[:60]}...\"")
            return None
        
        print(f"  [INFO] Vector search: {len(initial_docs)} kandidat")
        
        # 3. Re-ranking — Top-5
        reranked_docs, top_score = rerank_documents(question, initial_docs, top_k=FINAL_K)
        
        if not reranked_docs:
            print("  [WARN] Re-ranking menghasilkan 0 dokumen.")
            return None
        
        print(f"  [INFO] Top re-ranking score: {top_score:.4f}")
        
        # 3.5. Confidence Filter (Skip jika semua dokumen tidak relevan)
        if top_score < CONFIDENCE_THRESHOLD:
            print(f"  [WARN] Top score ({top_score:.4f}) di bawah threshold ({CONFIDENCE_THRESHOLD}).")
            print("  [WARN] Dokumen dianggap tidak relevan. Skipping...")
            return None
        
        # 4. Ekstrak konten mentah
        doc_contents = [doc.page_content for doc in reranked_docs]
        
        # Padding jika kurang dari 5 (edge case)
        while len(doc_contents) < 5:
            doc_contents.append("[Dokumen tidak tersedia]")
        
        return doc_contents[:5]
        
    except Exception as e:
        print(f"  [ERROR] Error saat retrieval: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI LLM
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    messages: List[Dict],
    temperature: float = 0.3,
    max_tokens: int = MAX_TOKENS,
    retries: int = 3,
    model: str = GENERATOR_MODEL
) -> Optional[str]:
    """Kirim request ke LLM API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False
    }
    for attempt in range(retries):
        try:
            r = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            wait = (attempt + 1) * 5
            print(f"  [WARN] HTTP Error (attempt {attempt+1}/{retries}): {e}. Retry in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [WARN] Error (attempt {attempt+1}/{retries}): {e}. Retry in 3s...")
            time.sleep(3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PARSING & VALIDASI
# ─────────────────────────────────────────────────────────────────────────────

def parse_raft_json(raw_response: str) -> Optional[Dict]:
    """
    Parse response LLM menjadi JSON RAFT yang valid.
    Menangani berbagai format output yang mungkin dihasilkan LLM
    (dengan/tanpa markdown fence, whitespace, dsb.)
    """
    if not raw_response:
        return None
    
    # Bersihkan markdown fence jika ada
    cleaned = raw_response.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Attempt 1: Parse langsung
    try:
        data = json.loads(cleaned)
        if validate_raft_structure(data):
            return data
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Cari JSON object terbesar dalam response
    try:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            data = json.loads(match.group())
            if validate_raft_structure(data):
                return data
    except json.JSONDecodeError:
        pass
    
    return None


def validate_raft_structure(data: Dict) -> bool:
    """Validasi bahwa JSON memiliki struktur RAFT yang benar."""
    required_keys = {"instruction", "documents", "thought_process", "completion"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        print(f"  [WARN] Missing keys: {missing}")
        return False
    
    # Validasi documents adalah list
    if not isinstance(data["documents"], list):
        print("  [WARN] 'documents' bukan list")
        return False
    
    # Validasi thought_process memiliki document_analysis dan summary
    tp = data["thought_process"]
    if isinstance(tp, dict):
        if "document_analysis" not in tp or "summary" not in tp:
            print("  [WARN] 'thought_process' tidak memiliki 'document_analysis' atau 'summary'")
            return False
        if not isinstance(tp["document_analysis"], list):
            print("  [WARN] 'document_analysis' bukan list")
            return False
    else:
        print("  [WARN] 'thought_process' bukan dict")
        return False
    
    # Validasi completion tidak kosong
    if not data.get("completion", "").strip():
        print("  [WARN] 'completion' kosong")
        return False
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_raft_entry(question: str, documents: List[str], max_attempts: int = 3) -> Optional[Dict]:
    """
    Generate satu entry RAFT dari pertanyaan + Top-5 documents.
    
    Args:
        question: Pertanyaan yang sudah di-retrieve
        documents: List of 5 document strings dari retrieval pipeline
        max_attempts: Jumlah retry jika LLM menghasilkan output yang tidak valid
    
    Returns:
        Dict RAFT entry yang valid, atau None jika gagal setelah semua attempts.
    """
    # Bangun user message: pertanyaan + dokumen
    docs_formatted = "\n\n".join(
        f"--- Dokumen {i+1} ---\n{doc}" for i, doc in enumerate(documents)
    )
    
    user_message = f"Pertanyaan:\n{question}\n\nTop-5 Documents:\n{docs_formatted}"
    
    messages = [
        {"role": "system", "content": RAFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    for attempt in range(1, max_attempts + 1):
        print(f"    Attempt {attempt}/{max_attempts}...", end=" ")
        
        raw = call_llm(messages, temperature=0.3, max_tokens=MAX_TOKENS)
        if not raw:
            print("LLM tidak merespons.")
            continue
        
        parsed = parse_raft_json(raw)
        if parsed:
            print("[VALID] JSON valid!")
            return parsed
        else:
            print("[INVALID] JSON tidak valid, retry...")
            time.sleep(REQUEST_DELAY)
    
    return None


def run_raft_pipeline(
    questions: List[str],
    output_path: Path,
    resume: bool = True
):
    """
    Pipeline utama RAFT Dataset Generation.
    
    Untuk setiap pertanyaan:
    1. Retrieve Top-5 documents langsung dari Supabase + CrossEncoder
    2. Kirim ke LLM dengan system prompt RAFT
    3. Parse & validasi JSON output
    4. Simpan ke file JSONL (append mode)
    
    Args:
        questions: List pertanyaan untuk generate dataset
        output_path: Path file output .jsonl
        resume: Jika True, skip pertanyaan yang sudah ada di output file
    """
    print("=" * 70)
    print("RAFT Dataset Generator — Retrieval-Grounded (V2)")
    print("=" * 70)
    print(f"Retrieval     : Self-contained (Supabase -> Top-{INITIAL_K} -> Rerank -> Top-{FINAL_K})")
    print(f"Reranker      : {RERANKER_MODEL_NAME}")
    print(f"LLM Model     : {GENERATOR_MODEL}")
    print(f"Output        : {output_path}")
    print(f"Jumlah Q      : {len(questions)}")
    print(f"Resume mode   : {resume}")
    print("=" * 70)
    
    # Load existing questions jika resume mode
    existing_questions = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    existing_questions.add(entry.get("instruction", "").strip())
                except json.JSONDecodeError:
                    continue
        if existing_questions:
            print(f"Resume: {len(existing_questions)} pertanyaan sudah ada, akan di-skip.\n")
    
    # Filter pertanyaan yang belum diproses
    pending_questions = [
        q for q in questions if q.strip() not in existing_questions
    ]
    
    if not pending_questions:
        print("Semua pertanyaan sudah diproses! Tidak ada yang perlu di-generate.")
        return
    
    print(f"Pertanyaan yang akan diproses: {len(pending_questions)}\n")
    
    # Mulai generate
    success_count = 0
    fail_count = 0
    
    for idx, question in enumerate(tqdm(pending_questions, desc="Generating RAFT Dataset"), start=1):
        print(f"\n[{idx}/{len(pending_questions)}] {question[:80]}...")
        
        # STEP 1: Retrieve Top-5 Documents langsung dari Supabase + CrossEncoder
        print("  Retrieving documents...")
        documents = retrieve_top5_documents(question)
        
        if not documents:
            print("  [ERROR] Retrieval gagal, skip pertanyaan ini.")
            fail_count += 1
            continue
        
        print(f"  [OK] {len(documents)} dokumen diperoleh.")
        time.sleep(REQUEST_DELAY)
        
        # STEP 2: Generate RAFT entry via LLM
        print("  Generating RAFT entry... (Mohon tunggu, ini bisa memakan waktu 40-60 detik)")
        raft_entry = generate_single_raft_entry(question, documents, max_attempts=3)
        
        if not raft_entry:
            print("  [ERROR] Gagal generate RAFT entry setelah semua attempts.")
            fail_count += 1
            continue
        
        # STEP 3: Simpan ke file JSONL (append)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(raft_entry, ensure_ascii=False) + "\n")
        
        success_count += 1
        print(f"  [SUCCESS] Tersimpan! (Total: {success_count})")
        
        time.sleep(REQUEST_DELAY)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"SELESAI!")
    print(f"  Berhasil : {success_count}")
    print(f"  Gagal    : {fail_count}")
    print(f"  📁 Output  : {output_path}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hanya memproses pertanyaan dari list QUESTIONS di atas
    questions_to_process = QUESTIONS
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = DATASET_DIR / f"raft_dataset_retrieval_grounded_{timestamp}.jsonl"
    
    run_raft_pipeline(
        questions=questions_to_process,
        output_path=output,
        resume=True
    )
