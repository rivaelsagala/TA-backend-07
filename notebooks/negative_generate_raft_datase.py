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

# LLM API Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
LLM_API_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

# Model & Generation Settings
GENERATOR_MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.001

# Retrieval Settings
INITIAL_K = 40  
FINAL_K = 5
CONFIDENCE_THRESHOLD = -9999.0 

# Dokumen yang WAJIB menjadi sumber negative sampling
EXCLUDED_DOCUMENT_IDS = [
    # "perdes_dis_mekarrahayu_05_2016",
    # "perdes_dis_padamukti_2_2018",
    "perdes_disbiru_07_2015",
    "perdes_disbiru_10_2016",
    "perdes_disbiru_6_2015",
    "perdes_discigentur kecamatan paseh kabupaten bandung_8_2018",
    "perdes_discikadut_03_2017",
    "perdes_discipedes kecamatan paseh kabupaten bandung_03_2018",
    "perdes_discipedes_04_2018",
    "perdes_dismajasetra_1_2018"
]

NEGATIVE_DOCUMENT_ID_SET = set(EXCLUDED_DOCUMENT_IDS)

# Output
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# INISIALISASI RETRIEVAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

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

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
print(f"Memuat Cross-Encoder ({RERANKER_MODEL_NAME})...")
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
print("Cross-Encoder berhasil dimuat!")

# ─────────────────────────────────────────────────────────────────────────────
# DAFTAR PERTANYAAN
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUESTIONS: int = None

QUESTIONS: List[str] = [
    "Apa sih yang dimaksud dengan pemerintahan desa di desa Mekarrahayu?"
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT NEGATIVE RAFT
# ─────────────────────────────────────────────────────────────────────────────

# RAFT_NEGATIVE_SYSTEM_PROMPT = """
# Anda adalah AI yang bertugas membuat dataset NEGATIVE SAMPLING untuk Retrieval-Augmented Fine-Tuning (RAFT).

# Tujuan:
# Membuat contoh training di mana pertanyaan diberikan bersama hasil retrieval yang TIDAK RELEVAN / TIDAK SESUAI dengan pertanyaan.

# Kondisi data:
# - Question telah digunakan untuk retrieval ke vector database.
# - Documents yang diberikan adalah hasil retrieval NEGATIVE SAMPLE.
# - Dokumen-dokumen ini berasal dari dokumen distraktor / dokumen lain yang tidak sesuai dengan target pertanyaan.
# - Documents adalah satu-satunya sumber analisis.

# Tugas:
# Buat SATU JSON VALID dengan struktur:
# - instruction
# - thought_process
# - completion

# ATURAN KHUSUS NEGATIVE SAMPLE:
# 1. Analisis setiap dokumen dan jelaskan kenapa dokumen tersebut tidak relevan / tidak sesuai / tidak menjawab pertanyaan.
# 2. Jika ada sedikit kemiripan istilah, tetap periksa apakah:
#    - nama desa berbeda,
#    - nomor peraturan berbeda,
#    - tahun berbeda,
#    - pasal/topik berbeda,
#    - konteks peraturan berbeda.
# 3. Jika dokumen berasal dari peraturan yang berbeda dari yang diminta pertanyaan, maka dokumen itu harus dianggap TIDAK RELEVAN.
# 4. Completion TIDAK BOLEH memaksakan jawaban dari dokumen yang salah.
# 5. Completion harus menegaskan bahwa informasi yang ditemukan tidak sesuai dengan pertanyaan atau tidak dapat digunakan untuk menjawab pertanyaan secara benar.

# Gaya thought_process:
# Untuk setiap dokumen, jelaskan:
# - apakah relevan atau tidak,
# - bagian apa yang dibahas dokumen,
# - kenapa tidak cocok dengan pertanyaan,
# - apakah dokumen berasal dari desa/peraturan/tahun lain,
# - apakah dokumen tidak dapat dipakai untuk menjawab pertanyaan.

# Completion:
# - Jangan menjawab seolah-olah dokumen itu benar.
# - Tegaskan bahwa dokumen yang ditemukan tidak sesuai / tidak relevan / tidak cukup untuk menjawab pertanyaan.
# - Gunakan kalimat natural, misalnya:
#   - "Informasi pada dokumen yang ditemukan tidak sesuai dengan pertanyaan karena membahas peraturan/dokumen lain."
#   - "Dokumen retrieval yang tersedia tidak relevan untuk menjawab pertanyaan ini."
#   - "Jawaban yang diminta tidak dapat ditentukan dari dokumen yang diberikan karena dokumen berasal dari konteks peraturan yang berbeda."

# Format output WAJIB:
# {
#   "instruction": "<question>",
#   "thought_process": {
#     "document_analysis": [
#       {
#         "document": 1,
#         "analysis": "..."
#       }
#     ],
#     "summary": "..."
#   },
#   "completion": "..."
# }

# PENTING:
# - Output HARUS berupa SATU JSON VALID.
# - Jangan gunakan markdown.
# - Jangan gunakan ```json.
# - Jangan beri penjelasan tambahan di luar JSON.
# """

RAFT_NEGATIVE_SYSTEM_PROMPT = """ Anda adalah AI yang bertugas membuat dataset NEGATIVE SAMPLING untuk Retrieval-Augmented Fine-Tuning (RAFT). Tujuan: Membuat contoh training di mana pertanyaan diberikan bersama hasil retrieval NEGATIVE, yaitu dokumen yang tidak boleh dipakai sebagai dasar jawaban. Jenis negative sample: 1. irrelevant: dokumen tidak relevan, salah topik, salah pasal, salah konteks, atau tidak memuat informasi yang dibutuhkan. 2. corrupted: dokumen tampak relevan dengan pertanyaan, tetapi isinya tidak sesuai dengan sumber ground truth / merupakan distractor / versi yang telah diedit, sehingga tidak valid sebagai evidence jawaban. Kondisi data: - Question telah digunakan untuk retrieval ke vector database. - Documents yang diberikan adalah hasil retrieval NEGATIVE SAMPLE. - Setiap dokumen memiliki: - document_id - content - negative_type - Documents adalah satu-satunya sumber analisis. - Jangan menganggap dokumen pasti benar hanya karena topiknya mirip dengan pertanyaan. Tugas: Buat SATU JSON VALID dengan struktur: { "instruction": "<question>", "thought_process": { "document_analysis": [ { "document": 1, "analysis": "..." } ], "summary": "..." }, "completion": "..." } ATURAN: 1. Analisis SEMUA dokumen satu per satu sesuai urutannya. 2. Untuk setiap dokumen, jelaskan: - apa isi dokumen secara singkat, - apakah dokumen tampak relevan atau tidak, - negative_type dokumen tersebut (irrelevant atau corrupted), - kenapa dokumen tidak valid / tidak cukup / tidak sesuai untuk menjawab pertanyaan. 3. Jika negative_type = corrupted: - jelaskan bahwa dokumen bisa tampak sangat relevan, - tetapi isi dokumen tidak sesuai dengan sumber ground truth / telah berubah / merupakan distractor, - sehingga tidak boleh dijadikan dasar jawaban. 4. Jika negative_type = irrelevant: - jelaskan bahwa dokumen tidak menjawab pertanyaan karena membahas topik, pasal, desa, atau konteks lain. 5. Jangan menyusun jawaban faktual dari dokumen negative. 6. Completion harus menegaskan bahwa dokumen yang diberikan tidak dapat dijadikan dasar jawaban yang benar. Gaya output: - Gunakan bahasa Indonesia yang natural, singkat, dan tegas. - Untuk corrupted, jangan hanya menulis "tidak relevan"; jelaskan bahwa dokumen tampak relevan tetapi tidak valid sebagai evidence. - Jika ada perbedaan isi dengan ground truth, jelaskan inti perbedaannya secara ringkas tanpa mengarang informasi di luar dokumen yang diberikan. PENTING: - Output HARUS berupa SATU JSON VALID. - Jangan gunakan markdown. - Jangan gunakan ```json. - Jangan beri penjelasan tambahan di luar JSON. """

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def rerank_documents(query: str, documents: list, top_k: int = FINAL_K) -> Tuple[list, float]:
    """
    Re-ranking dokumen kandidat negative sampling.
    """
    if not documents:
        return [], 0.0

    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    reranked = [doc for doc, _ in scored_docs[:top_k]]
    top_score = float(scored_docs[0][1]) if scored_docs else 0.0

    for rank, (doc, score) in enumerate(scored_docs[:top_k]):
        doc_id = doc.metadata.get("document_id", "?")
        print(f"    Rank {rank+1} | Score: {score:.4f} | document_id={doc_id}")

    return reranked, top_score


def retrieve_top5_negative_documents(question: str) -> Optional[List[Dict[str, str]]]:
    """
    Ambil dokumen NEGATIVE SAMPLING:
    HANYA dokumen yang document_id-nya ada di EXCLUDED_DOCUMENT_IDS.

    Return:
        [
          {
            "document_id": "...",
            "content": "..."
          },
          ...
        ]
    """
    try:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name=supabase_table,
            query_name="match_documents"
        )

        raw_initial_docs = vector_store.similarity_search(
            question,
            k=INITIAL_K
        )

        if not raw_initial_docs:
            print(f'  [WARN] Vector search tidak menemukan kandidat untuk: "{question[:80]}"')
            return None

        # FILTER: HANYA ambil dokumen yang document_id-nya termasuk daftar negative
        negative_candidates = []
        for doc in raw_initial_docs:
            doc_id = str(doc.metadata.get("document_id", "")).strip()
            if doc_id in NEGATIVE_DOCUMENT_ID_SET:
                negative_candidates.append(doc)

        # kalau tidak ada satu pun dokumen negative -> skip pertanyaan
        if not negative_candidates:
            print("  [WARN] Tidak ada dokumen dari EXCLUDED_DOCUMENT_IDS untuk pertanyaan ini. Skip.")
            return None

        print(f"  [INFO] Kandidat negative docs ditemukan: {len(negative_candidates)}")

        # Rerank hanya di antara negative candidates
        reranked_docs, top_score = rerank_documents(
            question,
            negative_candidates,
            top_k=FINAL_K
        )

        if not reranked_docs:
            print("  [WARN] Re-ranking negative docs menghasilkan 0 dokumen.")
            return None

        print(f"  [INFO] Top rerank score negative docs: {top_score:.4f}")

        # Bentuk output documents beserta document_id
        # TIDAK ADA PADDING DOKUMEN KOSONG
        doc_payloads = []
        for doc in reranked_docs[:FINAL_K]:
            doc_id = str(doc.metadata.get("document_id", "")).strip()
            content = (doc.page_content or "").strip()

            if not doc_id or not content:
                continue

            doc_payloads.append({
                "document_id": doc_id,
                "content": content
            })

        if not doc_payloads:
            print("  [WARN] Semua dokumen kosong setelah formatting.")
            return None

        return doc_payloads

    except Exception as e:
        print(f"  [ERROR] Error saat retrieval negative sampling: {e}")
        import traceback
        traceback.print_exc()
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI LLM
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = MAX_TOKENS,
    retries: int = 3,
    model: str = GENERATOR_MODEL
) -> Optional[str]:
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


# ─────────────────────────────────────────────────────────────────────────────
# PARSING & VALIDASI
# ─────────────────────────────────────────────────────────────────────────────

def validate_raft_structure(data: Dict) -> bool:
    required_keys = {"instruction", "thought_process", "completion"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        print(f"  [WARN] Missing keys: {missing}")
        return False

    tp = data["thought_process"]
    if not isinstance(tp, dict):
        print("  [WARN] 'thought_process' bukan dict")
        return False

    if "document_analysis" not in tp or "summary" not in tp:
        print("  [WARN] 'thought_process' tidak memiliki 'document_analysis' atau 'summary'")
        return False

    if not isinstance(tp["document_analysis"], list):
        print("  [WARN] 'document_analysis' bukan list")
        return False

    if not data.get("completion", "").strip():
        print("  [WARN] 'completion' kosong")
        return False

    return True


def parse_raft_json(raw_response: str) -> Optional[Dict]:
    if not raw_response:
        return None

    cleaned = raw_response.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if validate_raft_structure(data):
            return data
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            data = json.loads(match.group())
            if validate_raft_structure(data):
                return data
    except json.JSONDecodeError:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SATU ENTRY NEGATIVE RAFT
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_negative_raft_entry(
    question: str,
    documents: List[Dict[str, str]],
    max_attempts: int = 3
) -> Optional[Dict]:
    """
    documents format:
    [
      {"document_id": "...", "content": "..."},
      ...
    ]
    """

    docs_formatted = "\n\n".join(
        f"--- Dokumen {i+1} ---\n"
        f"document_id: {doc['document_id']}\n"
        f"content:\n{doc['content']}"
        for i, doc in enumerate(documents)
    )

    user_message = f"""Pertanyaan:
{question}

Top-5 Negative Documents:
{docs_formatted}
"""

    messages = [
        {"role": "system", "content": RAFT_NEGATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for attempt in range(1, max_attempts + 1):
        print(f"    Attempt {attempt}/{max_attempts}...", end=" ")

        raw = call_llm(messages, temperature=0.2, max_tokens=MAX_TOKENS)
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


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def run_negative_raft_pipeline(
    questions: List[str],
    output_path: Path,
    resume: bool = True
):
    print("=" * 70)
    print("RAFT NEGATIVE Dataset Generator")
    print("=" * 70)
    print(f"Retrieval     : Supabase -> filter EXCLUDED_DOCUMENT_IDS -> rerank -> Top-{FINAL_K}")
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

    pending_questions = [q for q in questions if q.strip() not in existing_questions]

    if not pending_questions:
        print("Semua pertanyaan sudah diproses.")
        return

    success_count = 0
    fail_count = 0

    for idx, question in enumerate(tqdm(pending_questions, desc="Generating Negative RAFT Dataset"), start=1):
        print(f"\n[{idx}/{len(pending_questions)}] {question[:100]}...")

        # STEP 1: retrieval negative docs
        print("  Retrieving NEGATIVE documents...")
        documents = retrieve_top5_negative_documents(question)

        if not documents:
            print("  [SKIP] Tidak ada negative documents yang cocok untuk pertanyaan ini.")
            fail_count += 1
            continue

        print(f"  [OK] {len(documents)} negative documents diperoleh.")

        # STEP 2: generate entry
        print("  Generating NEGATIVE RAFT entry...")
        raft_entry = generate_single_negative_raft_entry(
            question=question,
            documents=documents,
            max_attempts=3
        )

        if not raft_entry:
            print("  [ERROR] Gagal generate NEGATIVE RAFT entry.")
            fail_count += 1
            continue

        # STEP 3: save
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(raft_entry, ensure_ascii=False) + "\n")

        success_count += 1
        print(f"  [SUCCESS] Tersimpan! (Total sukses: {success_count})")

        time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 70)
    print("SELESAI!")
    print(f"  Berhasil : {success_count}")
    print(f"  Gagal/Skip : {fail_count}")
    print(f"  Output  : {output_path}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_questions_from_jsonl(filepath: Path) -> List[str]:
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
    output = DATASET_DIR / f"raft_negative_dataset_{timestamp}.jsonl"

    run_negative_raft_pipeline(
        questions=questions_to_process,
        output_path=output,
        resume=True
    )