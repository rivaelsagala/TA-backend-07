import os, sys, json, time, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
LLM_API_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.001  


INITIAL_K = 10  
FINAL_K = 5     
CONFIDENCE_THRESHOLD = -5.0  

EXCLUDED_DOCUMENT_IDS = [
    "perdes_dis_mekarrahayu_05_2016",
    "perdes_dis_padamukti_2_2018",
    "perdes_disbiru_07_2015",
    "perdes_disbiru_10_2016",
    "perdes_disbiru_6_2015",
    "perdes_discigentur kecamatan paseh kabupaten bandung_8_2018",
    "perdes_discikadut_03_2017",
    "perdes_discipedes kecamatan paseh kabupaten bandung_03_2018",
    "perdes_discipedes_04_2018",
    "perdes_dismajasetra_1_2018"
]

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from sentence_transformers import CrossEncoder

supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase_table = os.getenv("SUPABASE_TABLE_NAME", "documents")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    default_headers={"User-Agent": "curl/7.68.0"}
)

RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
print(f"Memuat Cross-Encoder ({RERANKER_MODEL_NAME})...")
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
print("Cross-Encoder berhasil dimuat!")


MAX_QUESTIONS: int = None

QUESTIONS: List[str] = [
    "Apa sih yang dimaksud dengan pemerintahan desa di desa Mekarrahayu?",
]

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
Retrieved Documents

Documents yang diberikan merupakan hasil retrieval setelah proses re-ranking dan merupakan satu-satunya sumber informasi (Single Source of Truth).

Tugas

Buat satu data RAFT yang terdiri dari:

instruction
thought_process
completion

Instruction

Gunakan pertanyaan yang diberikan.

Jangan mengubah isi pertanyaan.

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
Sintesis informasi relevan dari Retrieved Documents HANYA JIKA dokumen-dokumen tersebut membahas subjek yang sama.
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
  "thought_process": {
    "document_analysis": [
      {
        "document": 1,
        "analysis": "<analisis dokumen 1>"
      },
      {
        "document": 2,
        "analysis": "<analisis dokumen 2>"
      }
    ],
    "summary": "<ringkasan sintesis seluruh dokumen>"
  },
  "completion": "<jawaban akhir>"
}

Pastikan JSON valid sehingga setiap output dapat langsung disimpan sebagai satu baris (.jsonl) tanpa proses konversi tambahan.
PENTING:
- Hasilkan obyek di dalam `document_analysis` sebanyak jumlah dokumen yang diberikan. Jika ada 4 dokumen, hasilkan 4 obyek. JANGAN pernah menuliskan `...` atau komentar di dalam JSON.
- Jangan ada trailing comma."""



def rerank_documents(query: str, documents: list, top_k: int = FINAL_K) -> Tuple[list, float]:
    if not documents:
        return [], 0.0
    
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    reranked = [doc for doc, _ in scored_docs[:top_k]]
    top_score = float(scored_docs[0][1]) if scored_docs else 0.0
    
    for rank, (doc, score) in enumerate(scored_docs[:top_k]):
        src = doc.metadata.get("source", "?")
        print(f"    Rank {rank+1} | Score: {score:.4f} | {src}")
    
    return reranked, top_score


def retrieve_top5_documents(question: str) -> Optional[List[str]]:
    try:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name=supabase_table,
            query_name="match_documents"
        )
        
        raw_initial_docs = vector_store.similarity_search(question, k=INITIAL_K + len(EXCLUDED_DOCUMENT_IDS))

        initial_docs = []
        for doc in raw_initial_docs:
            doc_id = doc.metadata.get("document_id", "")
            if doc_id not in EXCLUDED_DOCUMENT_IDS:
                initial_docs.append(doc)
            if len(initial_docs) == INITIAL_K:
                break
                
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
        
        return doc_contents[:5]
        
    except Exception as e:
        print(f"  [ERROR] Error saat retrieval: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        "Content-Type": "application/json",
        "User-Agent": "curl/7.68.0"
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


def parse_raft_json(raw_response: str) -> Optional[Dict]:
    """
    Parse response LLM menjadi JSON RAFT yang valid.
    Menangani berbagai format output yang mungkin dihasilkan LLM
    (dengan/tanpa markdown fence, whitespace, dsb.)
    """
    if not raw_response:
        return None
    
    cleaned = raw_response.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Hapus trailing comma agar JSONDecodeError tidak terjadi akibat koma berlebih
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    
    try:
        data = json.loads(cleaned)
        if validate_raft_structure(data):
            return data
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSONDecodeError di attempt pertama: {e}")
        pass

    try:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            data = json.loads(match.group())
            if validate_raft_structure(data):
                return data
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSONDecodeError di attempt kedua: {e}")
        # Simpan raw response ke file untuk diinspeksi
        with open("failed_response.txt", "w", encoding="utf-8") as f:
            f.write(cleaned)
        print("  [DEBUG] Raw LLM Response disimpan di failed_response.txt")
        pass
    
    return None


def validate_raft_structure(data: Dict) -> bool:
    """Validasi bahwa JSON memiliki struktur RAFT yang benar."""
    required_keys = {"instruction", "thought_process", "completion"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        print(f"  [WARN] Missing keys: {missing}")
        return False
    
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

    if not data.get("completion", "").strip():
        print("  [WARN] 'completion' kosong")
        return False
    
    return True


def generate_single_raft_entry(question: str, documents: List[str], max_attempts: int = 3) -> Optional[Dict]:
    docs_formatted = "\n\n".join(
        f"--- Dokumen {i+1} ---\n{doc}" for i, doc in enumerate(documents)
    )
    
    user_message = f"Pertanyaan:\n{question}\n\nRetrieved Documents:\n{docs_formatted}"
    
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
            final_parsed = {
                "instruction": parsed.get("instruction", question),
                "documents": documents,
                "thought_process": parsed.get("thought_process", {}),
                "completion": parsed.get("completion", "")
            }
            print("[VALID] JSON valid!")
            return final_parsed
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
    
    pending_questions = [
        q for q in questions if q.strip() not in existing_questions
    ]
    
    if not pending_questions:
        print("Semua pertanyaan sudah diproses! Tidak ada yang perlu di-generate.")
        return
    
    print(f"Pertanyaan yang akan diproses: {len(pending_questions)}\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, question in enumerate(tqdm(pending_questions, desc="Generating RAFT Dataset"), start=1):
        print(f"\n[{idx}/{len(pending_questions)}] {question[:80]}...")
        
        print("  Retrieving documents...")
        documents = retrieve_top5_documents(question)
        
        if not documents:
            print("  [ERROR] Retrieval gagal, skip pertanyaan ini.")
            fail_count += 1
            continue
        
        print(f"  [OK] {len(documents)} dokumen diperoleh.")
        time.sleep(REQUEST_DELAY)
        
        print("  Generating RAFT entry... (Mohon tunggu, ini bisa memakan waktu 40-60 detik)")
        raft_entry = generate_single_raft_entry(question, documents, max_attempts=3)
        
        if not raft_entry:
            print("  [ERROR] Gagal generate RAFT entry setelah semua attempts.")
            fail_count += 1
            continue
        
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


def load_questions_from_file(filepath: Path) -> List[str]:
    """
    Load pertanyaan dari file teks (satu pertanyaan per baris).
    Mengabaikan baris kosong dan baris yang dimulai dengan #.
    """
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions

def load_questions_from_jsonl(filepath: Path) -> List[str]:
    """
    Load pertanyaan (instruction) dari dataset JSONL yang sudah ada.
    """
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "instruction" in data:
                    questions.append(data["instruction"].strip())
            except json.JSONDecodeError:
                continue
    return questions

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_lama = DATASET_DIR / "raft_dataset_finalv1.jsonl"
    
    if dataset_lama.exists():
        questions_to_process = load_questions_from_jsonl(dataset_lama)
        print(f"Loaded {len(questions_to_process)} pertanyaan dari {dataset_lama.name}")
    else:
        print(f"File {dataset_lama} tidak ditemukan, menggunakan QUESTIONS default.")
        questions_to_process = QUESTIONS
        
    if MAX_QUESTIONS is not None:
        questions_to_process = questions_to_process[:MAX_QUESTIONS]
        print(f"Membatasi pemrosesan hanya untuk {MAX_QUESTIONS} pertanyaan pertama.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = DATASET_DIR / f"raft_dataset_retrieval_grounded_{timestamp}.jsonl"
    
    run_raft_pipeline(
        questions=questions_to_process,
        output_path=output,
        resume=True
    )
