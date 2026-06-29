"""
RAFT Dataset Generator — Supabase Vector Search Edition
=======================================================
Menggunakan Embedding (Cosine Similarity) via Supabase pgvector 
untuk mendapatkan "Hard Negatives" yang 100% akurat secara semantik.
Ini secara sempurna mensimulasikan lingkungan RAG di Production!
"""

import os, sys, json, time, random, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# SETUP & KONFIGURASI
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

MAX_TOKENS           = 1024
REQUEST_DELAY        = 0.2
NUM_DISTRACTORS      = 4
ORACLE_PRESENT_RATIO = 0.80

# === SUPABASE VECTOR DB SETUP ===
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase_table = os.getenv("SUPABASE_TABLE_NAME", "documents")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    base_url=_base if "maiarouter" in _base else None
)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name=supabase_table,
    query_name="match_documents"
)

# ─────────────────────────────────────────────────────────────────────────────
# TIPE PERTANYAAN & GAYA
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "faktual": {
        "weight": 0.20,
        "prompt": "Buat pertanyaan FAKTUAL spesifik tentang angka, tanggal, atau fakta yang HANYA ada di dokumen ini. Wajib sebut nama peraturan dan nomor pasal."
    },
    "definisional": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan DEFINISIONAL tentang istilah teknis dalam dokumen. Sertakan nama peraturan."
    },
    "prosedural": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan PROSEDURAL tentang langkah/tata cara operasional. Boleh sebut peraturannya, boleh tidak."
    },
    "kondisional": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan KONDISIONAL tentang syarat atau konsekuensi. Sebutkan nama peraturannya."
    },
    "komparatif": {
        "weight": 0.10,
        "prompt": "Buat pertanyaan KOMPARATIF yang membandingkan kewajiban/hak dalam dokumen tersebut."
    },
    "natural_awam": {
        "weight": 0.25,
        "prompt": "Buat pertanyaan NATURAL bergaya bahasa sehari-hari (awam) seolah-olah ditanyakan oleh warga desa biasa yang tidak tahu hukum. JANGAN menyebutkan nama peraturan, JANGAN sebut nomor pasal, dan gunakan bahasa santai/kolokial. Contoh: 'Gimana sih syaratnya kalau mau jadi anggota BPD?' atau 'Kalau kades ngelanggar aturan, sanksinya apa ya?'"
    }
}

COMPLETION_STYLES = [
    "Jawab langsung dan padat, sebutkan fakta dan angka relevan.",
    "Jawab dengan formalitas bahasa hukum, kutip pasalnya dengan rapi.",
    "Berikan jawaban terstruktur dengan bullet points jika ada banyak syarat.",
    "Beri penjelasan lengkap yang mengedukasi warga awam dengan bahasa yang lebih santai namun akurat."
]

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages: List[Dict], temperature: float = 0.7, max_tokens: int = MAX_TOKENS, retries: int = 3) -> Optional[str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GENERATOR_MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.9, "stream": False
    }
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            wait = (attempt + 1) * 5
            time.sleep(wait)
        except Exception:
            time.sleep(3)
    return None

def clean_content(text: str) -> str:
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()

def is_substantive(content: str, min_chars: int = 150) -> bool:
    c = clean_content(content)
    if len(c) < min_chars: return False
    words = c.split()
    return len(words) >= 5

def get_doc_label(chunk_metadata: Dict, pasal: str = "") -> str:
    title = chunk_metadata.get("title", "Dokumen tidak diketahui")
    p = pasal or chunk_metadata.get("section", "")
    return f"{title}, {p.title()}" if p else title

def load_chunks_from_dir(directory: Path) -> List[Tuple[str, Dict]]:
    loaded = []
    if not directory.exists(): return loaded
    for fpath in directory.glob("*_chunks.json"):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
            for c in chunks:
                loaded.append((doc_id, c))
    return loaded

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC PEMILIHAN DISTRACTOR VIA SUPABASE VECTOR SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def select_distractors_supabase(oracle_text: str, oracle_doc_id: str, n: int) -> List[Dict]:
    """
    Mengambil distractor menggunakan Vector Similarity (Supabase pgvector).
    Ini menghasilkan Hard Negatives murni secara semantik yang persis akan dialami RAG di Production.
    """
    try:
        # Cari top 15 dokumen mirip berdasarkan semantic embeddings
        results = vector_store.similarity_search(oracle_text, k=15)
        
        candidates = []
        for doc in results:
            doc_id = doc.metadata.get("document_id", "")
            
            # Buang jika berasal dari dokumen yang sama (kita butuh pengecoh dari dokumen BEDA)
            if doc_id == oracle_doc_id:
                continue
                
            chunk_dict = {
                "metadata": doc.metadata,
                "content": doc.page_content,
                "pasal": doc.metadata.get("section", "")
            }
            candidates.append(chunk_dict)
            
        # Ambil top pool dari candidate yang tersisa, lalu acak posisinya sedikit
        top_pool = candidates[:max(n * 2, 8)]
        random.shuffle(top_pool)
        
        return top_pool[:n]
        
    except Exception as e:
        print(f"\n[!] Error querying Supabase: {e}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def generate_question(oracle_content: str, doc_title: str) -> Optional[Tuple[str, str]]:
    q_type = random.choices(list(QUESTION_TYPES.keys()), weights=[v["weight"] for v in QUESTION_TYPES.values()])[0]
    prompt = QUESTION_TYPES[q_type]["prompt"]
    
    system = (
        "Anda adalah pakar pembuat dataset RAFT RAG bidang Hukum Desa.\n"
        f"TUGAS: {prompt}\n"
        "ATURAN: Pertanyaan spesifik bisa dijawab 100% dari dokumen, output HANYA pertanyaan tanpa teks pembuka/penutup."
    )
    res = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Dokumen: {doc_title}\n{oracle_content}\nBuat pertanyaan:"}
    ])
    if not res: return None
    q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*)", "", res, flags=re.IGNORECASE).strip().strip('"\'')
    return (q_type, q) if len(q) > 20 else None

def generate_thought_and_completion(
    question: str, docs: List[str], doc_labels: List[str], oracle_idx: int, style: str
) -> Tuple[Optional[str], Optional[str]]:
    
    docs_fmt = "".join(f"\n--- Dokumen {i+1}: {l} ---\n{d}\n" for i, (d, l) in enumerate(zip(docs, doc_labels)))
    oracle_present = oracle_idx >= 0
    
    hint = (
        f"\n[INTERNAL] Dokumen {oracle_idx+1} mengandung jawaban." if oracle_present else
        "\n[INTERNAL] TIDAK ADA dokumen yang menjawab pertanyaan ini."
    )
    
    t_instr = (
        "Analisis tiap dokumen. Jika relevan, kutip verbatim <<teks asli>>. "
        "Jika tidak, jelaskan mengapa topiknya berbeda."
    ) if oracle_present else (
        "Analisis tiap dokumen. Tunjukkan bahwa topiknya tidak ada yang menjawab pertanyaan."
    )
    
    c_instr = (
        f"Gaya: {style}\nJawab langsung, kutip dokumen tanpa menyebut 'Dokumen 1'."
    ) if oracle_present else (
        "Tolak menjawab secara halus. Sebutkan bahwa dokumen yang ada membahas hal lain, bukan hal yang ditanyakan."
    )

    sys_msg = (
        "Anda AI pembuat data RAFT.\nOutput JSON valid:\n"
        '{"thought_process": "...", "completion": "..."}\n'
        f"ATURAN THOUGHT:\n{t_instr}\nATURAN COMPLETION:\n{c_instr}"
    )
    
    res = call_llm(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Q: {question}\n{docs_fmt}\n{hint}"}],
        temperature=0.75, max_tokens=1024
    )
    
    if not res: return None, None
    for attempt in [res, re.search(r'\{[\s\S]*\}', res)]:
        try:
            raw = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
            if not raw: continue
            data = json.loads(raw)
            return data.get("thought_process", "").strip(), data.get("completion", "").strip()
        except: continue
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_raft_pipeline(output_path: Path):
    print("Memuat dokumen utama (Oracle Pool)...")
    oracle_pool = load_chunks_from_dir(PROCESSED_DIR)
    
    valid_oracles = [(d, c) for d, c in oracle_pool if is_substantive(c["content"])]
    
    print(f"Total Chunks Tersedia: {len(oracle_pool)}")
    print(f"Valid Oracles yang siap diproses: {len(valid_oracles)}\n")
    print(f"Distractor akan diambil langsung dari Supabase Database!\n")

    dataset, failed = [], 0
    with open(output_path, "w", encoding="utf-8") as f:
        pass # Reset file

    for doc_id, chunk in tqdm(valid_oracles, desc="Generating RAFT Dataset"):
        oracle_text = clean_content(chunk["content"])
        oracle_lbl = get_doc_label(chunk.get("metadata", {}), chunk.get("pasal", ""))
        
        # 1. Generate Q
        q_res = generate_question(oracle_text, oracle_lbl)
        if not q_res: 
            failed += 1; continue
        q_type, question = q_res
        time.sleep(REQUEST_DELAY)
        
        # 2. Select Distractors via SUPABASE
        d_chunks = select_distractors_supabase(oracle_text, doc_id, NUM_DISTRACTORS)
        
        # Jika Supabase mengembalikan kurang dari 4 (karena error jaringan dll), lewati baris ini
        if len(d_chunks) < NUM_DISTRACTORS:
            print(f"\n[!] Gagal mendapatkan cukup distractor untuk doc_id: {doc_id}")
            failed += 1; continue

        d_texts = [clean_content(c["content"]) for c in d_chunks]
        d_lbls = [get_doc_label(c["metadata"], c.get("pasal", "")) for c in d_chunks]
        
        # 3. Setup Oracle Present vs Absent
        is_present = random.random() < ORACLE_PRESENT_RATIO
        if is_present:
            pos = random.randint(0, len(d_texts))
            docs = d_texts[:pos] + [oracle_text] + d_texts[pos:]
            lbls = d_lbls[:pos] + [oracle_lbl] + d_lbls[pos:]
        else:
            docs, lbls, pos = d_texts, d_lbls, -1
            
        style = random.choice(COMPLETION_STYLES)
        
        # 4. Generate Thought + Completion
        thought, completion = generate_thought_and_completion(question, docs, lbls, pos, style)
        time.sleep(REQUEST_DELAY)
        
        if not thought or not completion:
            failed += 1; continue
            
        sample = {
            "instruction": question,
            "documents": docs,
            "thought_process": thought,
            "completion": completion,
            "metadata_extra": {
                "question_type": q_type,
                "oracle_present": is_present,
                "oracle_doc_id": doc_id if is_present else None,
                "distractor_source": "supabase_vector_search"
            }
        }
        
        dataset.append(sample)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"\nSelesai! Berhasil: {len(dataset)}, Gagal: {failed}")
    print(f"Tersimpan di: {output_path}")

if __name__ == "__main__":
    output = DATASET_DIR / "raft_dataset_supabase.jsonl"
    run_raft_pipeline(output)
