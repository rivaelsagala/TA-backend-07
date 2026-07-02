"""
generate_multi_oracle_raft.py
==============================
Tambahan untuk pipeline RAFT generator existing (generate_raft_dataset.py).

TUJUAN
------
Pipeline single-oracle Anda mengasosiasikan setiap pertanyaan dengan TEPAT SATU
dokumen oracle (oracle_index = int tunggal). Model yang dilatih dengan skema ini
belajar pola "cari 1 dokumen paling cocok, jawab dari situ, abaikan sisanya" —
sehingga saat ada >1 dokumen relevan yang saling melengkapi, model hanya
mengambil satu dan mengabaikan yang lain.

Script ini menambahkan kelas sample baru: MULTI-ORACLE, di mana 2-3 chunk
yang berkaitan (paling sering: ayat-ayat dalam satu pasal, kadang pasal-pasal
berbeda dalam satu dokumen yang membahas tema sama) dijadikan oracle SEKALIGUS.

PERUBAHAN UTAMA vs pipeline lama
---------------------------------
1. oracle_index sekarang berupa LIST of int, bukan int tunggal.
   -> oracle_present=True & oracle_index=[2, 4, 5] artinya dokumen ke 2,4,5
      (1-based, sesuai posisi di list "documents") adalah oracle.
2. thought_process WAJIB mengevaluasi SETIAP dokumen satu per satu, format:
       "Dokumen 1: relevan/tidak relevan, <alasan singkat>."
   satu baris per dokumen, secara berurutan. Ini memaksa model belajar pola
   "scan semua dokumen dulu" bukan "tembak satu dokumen lalu berhenti".
3. completion WAJIB menyintesis informasi dari SEMUA oracle (bukan cuma kutip
   1 dokumen), kalau oracle > 1.
4. Strategi pengelompokan oracle (proporsional, default):
   - 70% group_by_pasal: ambil 2-3 chunk (ayat berbeda) dalam (document_id, pasal)
     yang sama. Paling aman karena ayat2 dalam 1 pasal pasti koheren temanya.
   - 30% group_by_bab: ambil 2-3 chunk dari pasal BERBEDA tapi (document_id, bab)
     sama -> menangkap kasus topik yang tersebar lintas pasal dalam 1 bab.

CARA PAKAI
----------
Jalankan SETELAH pipeline single-oracle existing Anda (generate_raft_dataset.py).
Script ini akan APPEND ke file output yang sama (atau file baru, lalu Anda
gabung manual / pakai concat_datasets()).

    python generate_multi_oracle_raft.py

Lalu gabungkan dengan dataset single-oracle Anda:
    cat raft_dataset_final.jsonl raft_dataset_multi_oracle.jsonl > raft_dataset_combined.jsonl

PROPORSI YANG DISARANKAN dalam dataset gabungan akhir:
    ~55-60% single-oracle (skema lama Anda)
    ~20-25% multi-oracle (skema baru di script ini)
    ~20% oracle absent / no-answer (skema lama Anda, refusal training)
Ini supaya model tidak bias ke salah satu pola.
"""

import os, sys, json, time, random, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util
    print("Memuat model embedding (all-MiniLM-L6-v2)...")
    EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except ImportError:
    EMBEDDING_MODEL = None
    print("Warning: 'sentence-transformers' tidak ditemukan. Menggunakan fallback lexical (Jaccard).")

EMBED_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP & KONFIGURASI (samakan dengan pipeline lama Anda)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
HF_BASE_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL  = "openai/gpt-4o-mini"
VALIDATOR_MODEL  = "openai/gpt-4o-mini"
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
DISTRAKTOR_DIR   = PROCESSED_DIR / "distraktor"
DATASET_DIR      = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS            = 1024
REQUEST_DELAY         = 0.001
NUM_DISTRACTORS       = 3        # sedikit lebih sedikit drpd single-oracle krn oracle > 1
MIN_ORACLES_PER_GROUP = 2
MAX_ORACLES_PER_GROUP = 3
GROUP_BY_PASAL_RATIO  = 0.70     # 70% group_by_pasal, 30% group_by_bab
MIN_GROUPS_PER_DOC    = 1        # minimal grup yg dicoba per dokumen (biar tiap dok kebagian)

COMPLETION_STYLES = [
    "Formal (Gunakan bahasa hukum, kutip pasalnya dengan rapi dan padat).",
    "Natural (Gunakan bahasa yang lebih natural untuk warga awam, namun tetap akurat dan bersumber langsung dari dokumen).",
]

QUESTION_TYPES_MULTI = {
    "synthesis": {
        "weight": 0.40,
        "prompt": (
            "Buat SATU pertanyaan yang jawabannya HANYA BISA LENGKAP jika menggabungkan "
            "informasi dari SEMUA potongan dokumen oracle yang diberikan (jangan buat "
            "pertanyaan yang bisa terjawab penuh dari satu potongan saja). "
            "Contoh pola: 'siapa saja unsur X yang harus hadir dalam Y', 'apa saja syarat "
            "dan tahapan Z', 'jelaskan ketentuan lengkap tentang W'."
        ),
    },
    "natural_awam_synthesis": {
        "weight": 0.35,
        "prompt": (
            "Buat pertanyaan NATURAL bahasa sehari-hari (gaya warga desa awam, santai, "
            "tanpa sebut nomor pasal/peraturan) yang jawabannya perlu menggabungkan "
            "SEMUA potongan dokumen oracle yang diberikan, bukan cuma satu."
        ),
    },
    "comparative_multi": {
        "weight": 0.25,
        "prompt": (
            "Buat pertanyaan KOMPARATIF atau yang meminta daftar/rincian lengkap, "
            "di mana setiap potongan dokumen oracle menyumbang BAGIAN BERBEDA dari jawaban "
            "(misal tiap ayat menyebut syarat/unsur yang berbeda)."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAS (sama seperti pipeline lama)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages, temperature=0.7, max_tokens=MAX_TOKENS, retries=3, model=GENERATOR_MODEL) -> Optional[str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": 0.9, "stream": False}
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError:
            time.sleep((attempt + 1) * 5)
        except Exception:
            time.sleep(3)
    return None


def clean_content(text: str) -> str:
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()


def is_substantive(content: str, min_chars: int = 100) -> bool:
    c = clean_content(content)
    if len(c) < min_chars:
        return False
    return len(c.split()) >= 5


def get_doc_label(chunk: Dict) -> str:
    title = chunk.get("metadata", {}).get("title", "Dokumen tidak diketahui")
    pasal = chunk.get("pasal", "") or chunk.get("metadata", {}).get("section", "")
    return f"{title}, {pasal.title()}" if pasal else title


def load_chunks_from_dir(directory: Path, is_distractor: bool = False) -> List[Tuple[str, Dict, bool]]:
    loaded = []
    if not directory.exists():
        return loaded
    for fpath in directory.glob("*_chunks.json"):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
            for c in chunks:
                loaded.append((doc_id, c, is_distractor))
    return loaded


def semantic_similarity_score(text_a: str, text_b: str) -> float:
    if EMBEDDING_MODEL:
        if text_a not in EMBED_CACHE:
            EMBED_CACHE[text_a] = EMBEDDING_MODEL.encode(text_a, convert_to_tensor=True)
        if text_b not in EMBED_CACHE:
            EMBED_CACHE[text_b] = EMBEDDING_MODEL.encode(text_b, convert_to_tensor=True)
        return util.pytorch_cos_sim(EMBED_CACHE[text_a], EMBED_CACHE[text_b]).item()
    stopwords = {"yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini", "itu",
                 "atau", "tidak", "dalam", "adalah", "oleh", "pasal", "nomor", "ayat", "tahun"}
    def tokenize(t):
        return set(w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in stopwords)
    a, b = tokenize(text_a), tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_distractors(oracle_texts: List[str], oracle_doc_id: str,
                        distractor_pool: List[Tuple[str, Dict, bool]], n: int) -> List[Dict]:
    """Sama seperti pipeline lama, tapi cek overlap rata-rata terhadap SEMUA oracle text."""
    candidates = []
    for d_id, chunk, is_pure_distractor in distractor_pool:
        if d_id == oracle_doc_id:
            continue
        c_text = clean_content(chunk["content"])
        scores = [semantic_similarity_score(ot, c_text) for ot in oracle_texts]
        score = max(scores) if scores else 0.0

        if is_pure_distractor:
            score += 0.2 if EMBEDDING_MODEL else 0.5

        lower_bound = 0.35 if EMBEDDING_MODEL else 0.1
        upper_bound = 0.75 if EMBEDDING_MODEL else 0.9
        if lower_bound <= score <= upper_bound:
            candidates.append((score, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected_chunks, seen_labels = [], set()
    for score, chunk in candidates:
        lbl = get_doc_label(chunk)
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            selected_chunks.append(chunk)
        if len(selected_chunks) >= max(n * 2, 6):
            break

    random.shuffle(selected_chunks)
    return selected_chunks[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PENGELOMPOKAN ORACLE GANDA
# ─────────────────────────────────────────────────────────────────────────────

def build_oracle_groups(valid_chunks: List[Tuple[str, Dict]]) -> List[Dict]:
    """
    Mengelompokkan chunk menjadi grup-grup oracle multi-dokumen.

    Strategi:
      - group_by_pasal: chunk2 dengan (document_id, pasal) sama (ayat berbeda).
      - group_by_bab:    chunk2 dengan (document_id, bab) sama, pasal berbeda
                         (menangkap topik yang nyebar lintas pasal).

    Return: list of dict {doc_id, chunks: [...], strategy: "pasal"/"bab"}
    """
    by_pasal = defaultdict(list)
    by_bab = defaultdict(list)

    for doc_id, chunk in valid_chunks:
        pasal_key = (doc_id, chunk.get("pasal", ""))
        bab_key = (doc_id, chunk.get("bab", ""))
        by_pasal[pasal_key].append(chunk)
        by_bab[bab_key].append(chunk)

    groups = []

    # group_by_pasal: ambil grup dengan >= MIN_ORACLES_PER_GROUP chunk
    for (doc_id, pasal), chunks in by_pasal.items():
        if len(chunks) >= MIN_ORACLES_PER_GROUP and pasal:
            groups.append({"doc_id": doc_id, "chunks": chunks, "strategy": "pasal"})

    # group_by_bab: ambil grup dengan chunk dari >=2 pasal BERBEDA dalam bab yang sama
    for (doc_id, bab), chunks in by_bab.items():
        distinct_pasal = set(c.get("pasal", "") for c in chunks)
        if len(chunks) >= MIN_ORACLES_PER_GROUP and len(distinct_pasal) >= 2 and bab:
            groups.append({"doc_id": doc_id, "chunks": chunks, "strategy": "bab"})

    return groups


def sample_oracle_subset(group: Dict) -> List[Dict]:
    """Ambil subset 2-3 chunk acak dari sebuah grup oracle."""
    n = random.randint(MIN_ORACLES_PER_GROUP, min(MAX_ORACLES_PER_GROUP, len(group["chunks"])))
    return random.sample(group["chunks"], n)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def generate_multi_question(oracle_chunks: List[Dict], doc_title: str) -> Optional[Tuple[str, str]]:
    q_type = random.choices(
        list(QUESTION_TYPES_MULTI.keys()),
        weights=[v["weight"] for v in QUESTION_TYPES_MULTI.values()],
    )[0]
    prompt = QUESTION_TYPES_MULTI[q_type]["prompt"]

    oracle_combined = "\n---\n".join(clean_content(c["content"]) for c in oracle_chunks)

    system = (
        "Anda adalah pakar pembuat dataset RAFT RAG bidang Hukum Desa, KHUSUS untuk "
        "kasus pertanyaan yang jawabannya tersebar di BEBERAPA potongan dokumen.\n"
        f"TUGAS: {prompt}\n"
        "ATURAN: Pertanyaan harus bisa dijawab LENGKAP HANYA jika menggabungkan semua "
        "potongan yang diberikan. Output HANYA pertanyaan, tanpa teks pembuka/penutup."
    )
    res = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Dokumen: {doc_title}\nPotongan-potongan terkait:\n{oracle_combined}\nBuat pertanyaan:"},
    ])
    if not res:
        return None
    q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*)", "", res, flags=re.IGNORECASE).strip().strip('"\'')
    return (q_type, q) if len(q) > 20 else None


def generate_multi_thought_and_completion(
    question: str, docs: List[str], doc_labels: List[str],
    oracle_indices_0based: List[int], style: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Beda dengan versi single-oracle: thought_process WAJIB membahas SETIAP
    dokumen satu per satu (Dokumen 1: ..., Dokumen 2: ..., dst), dan completion
    WAJIB menyintesis semua oracle, bukan kutip satu saja.
    """
    docs_fmt = "".join(
        f"\n--- Dokumen {i+1}: {l} ---\n{d}\n" for i, (d, l) in enumerate(zip(docs, doc_labels))
    )
    oracle_nums = [i + 1 for i in oracle_indices_0based]

    hint = (
        f"\n[INTERNAL] Dokumen {', '.join(map(str, oracle_nums))} SEMUA mengandung "
        "bagian dari jawaban (jawaban harus menggabungkan SEMUA dokumen ini)."
    )

    t_instr = (
        "Buat 'thought_process' analitis dengan urutan berikut:\n"
        "1. Identifikasi inti pertanyaan secara singkat.\n"
        "2. Evaluasi isi SETIAP dokumen (Dokumen 1 sampai terakhir) secara mendalam, sebutkan secara spesifik mengapa ia relevan atau tidak relevan untuk menjawab pertanyaan.\n"
        "3. Simpulkan bahwa Anda harus menggabungkan/mensintesis informasi dari dokumen-dokumen yang relevan.\n"
        f"Catatan: Anda HARUS mengevaluasi Dokumen {', '.join(map(str, oracle_nums))} sebagai relevan, dan sisanya tidak relevan."
    )

    c_instr = (
        f"Gaya: {style}\n"
        f"Jawab dengan MENYINTESIS informasi dari SEMUA dokumen relevan "
        f"(Dokumen {', '.join(map(str, oracle_nums))}) menjadi SATU jawaban utuh dan koheren. "
        "JANGAN hanya mengutip satu dokumen saja — gabungkan semua poin dari semua dokumen relevan. "
        "DILARANG KERAS menambahkan opini atau fakta yang tidak ada di dokumen-dokumen tersebut."
    )

    sys_msg = (
        "Anda AI pembuat data RAFT, mode MULTI-ORACLE (lebih dari satu dokumen relevan).\n"
        "Output JSON valid:\n"
        '{"thought_process": "...", "completion": "..."}\n'
        f"ATURAN THOUGHT:\n{t_instr}\nATURAN COMPLETION:\n{c_instr}"
    )

    res = call_llm(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Q: {question}\n{docs_fmt}\n{hint}"}],
        temperature=0.3, max_tokens=1024,
    )

    if not res:
        return None, None
    for attempt in [res, re.search(r'\{[\s\S]*\}', res)]:
        try:
            raw = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
            if not raw:
                continue
            data = json.loads(raw)
            return data.get("thought_process", "").strip(), data.get("completion", "").strip()
        except Exception:
            continue
    return None, None


def validate_multi_sample(sample: Dict) -> Optional[Dict]:
    """Validator khusus multi-oracle: cek semua oracle terpakai, tidak cuma satu."""
    oracle_nums = [i + 1 for i in sample["metadata_extra"]["oracle_index"]]
    sys_msg = (
        "Anda adalah evaluator ketat dataset RAFT MULTI-ORACLE (>1 dokumen relevan). "
        "Validasi sampel berikut:\n"
        "1. instruction_answered: true HANYA JIKA completion menjawab instruction secara lengkap.\n"
        "2. grounded: true HANYA JIKA seluruh isi completion berasal murni dari dokumen "
        "(tidak ada opini/fakta luar/halusinasi).\n"
        "3. uses_all_oracles: true HANYA JIKA completion benar-benar menggabungkan informasi "
        f"dari SEMUA dokumen oracle ({', '.join(map(str, oracle_nums))}), bukan cuma salah satu saja. "
        "Jika completion hanya berasal dari 1 dokumen oracle padahal ada lebih dari 1, set false.\n"
        "4. thought_correct: true JIKA thought_process mengidentifikasi inti pertanyaan, mengevaluasi SEMUA dokumen yang diberikan secara eksplisit, "
        "dan secara akurat memutuskan dokumen oracle mana yang relevan untuk digabungkan dan mana yang tidak relevan.\n"
        "Output JSON:\n"
        '{"pass": boolean, "instruction_answered": boolean, "grounded": boolean, '
        '"uses_all_oracles": boolean, "thought_correct": boolean, "score": float, "reason": "..."}'
    )
    user_msg = (
        f"Instruction: {sample['instruction']}\n"
        f"Documents: {json.dumps(sample['documents'], ensure_ascii=False)}\n"
        f"Thought Process: {sample['thought_process']}\n"
        f"Completion: {sample['completion']}\n"
        f"Oracle indices (1-based): {oracle_nums}"
    )
    res = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                    temperature=0.0, max_tokens=256, model=VALIDATOR_MODEL)
    if not res:
        return None
    try:
        match = re.search(r'\{[\s\S]*\}', res)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_oracle_pipeline(output_path: Path, max_samples_per_doc: int = 3):
    print("Memuat dokumen utama...")
    oracle_pool_raw = load_chunks_from_dir(PROCESSED_DIR, is_distractor=False)
    print(f"Memuat pure distractors (dari {DISTRAKTOR_DIR.name})...")
    pure_distractors = load_chunks_from_dir(DISTRAKTOR_DIR, is_distractor=True)
    distractor_pool = oracle_pool_raw + pure_distractors

    valid_chunks = [
        (doc_id, c) for doc_id, c, is_dis in oracle_pool_raw
        if not is_dis and is_substantive(c["content"])
    ]
    print(f"Valid chunks: {len(valid_chunks)}")

    groups = build_oracle_groups(valid_chunks)
    # mix proporsional: 70% pasal, 30% bab
    pasal_groups = [g for g in groups if g["strategy"] == "pasal"]
    bab_groups = [g for g in groups if g["strategy"] == "bab"]
    print(f"Grup oracle ditemukan: {len(pasal_groups)} by-pasal, {len(bab_groups)} by-bab")

    n_pasal = int(len(pasal_groups) * 1.0)  # pakai semua, ratio diatur lewat sampling weight di bawah
    target_groups = []
    target_groups += [(g, "pasal") for g in pasal_groups]
    target_groups += [(g, "bab") for g in bab_groups]
    random.shuffle(target_groups)

    # batasi jumlah grup per dokumen biar tidak over-represented satu perdes saja
    per_doc_count = defaultdict(int)
    filtered_groups = []
    for g, strat in target_groups:
        if per_doc_count[g["doc_id"]] < max_samples_per_doc:
            filtered_groups.append((g, strat))
            per_doc_count[g["doc_id"]] += 1

    print(f"Total grup yang akan diproses: {len(filtered_groups)}")

    dataset, failed = [], 0
    with open(output_path, "w", encoding="utf-8") as f:
        pass  # reset file

    for group, strategy in tqdm(filtered_groups, desc="Generating Multi-Oracle RAFT"):
        oracle_chunks = sample_oracle_subset(group)
        oracle_texts = [clean_content(c["content"]) for c in oracle_chunks]
        oracle_lbls = [get_doc_label(c) for c in oracle_chunks]
        doc_title = oracle_chunks[0].get("metadata", {}).get("title", "Dokumen")
        doc_id = group["doc_id"]

        success = False
        for attempt in range(4):
            q_res = generate_multi_question(oracle_chunks, doc_title)
            if not q_res:
                continue
            q_type, question = q_res
            time.sleep(REQUEST_DELAY)

            d_chunks = select_distractors(oracle_texts, doc_id, distractor_pool, NUM_DISTRACTORS)
            while len(d_chunks) < NUM_DISTRACTORS and len(distractor_pool) > 0:
                fb = random.choice(distractor_pool)[1]
                if fb not in d_chunks:
                    d_chunks.append(fb)

            d_texts = [clean_content(c["content"]) for c in d_chunks]
            d_lbls = [get_doc_label(c) for c in d_chunks]

            # acak posisi oracle di antara distraktor
            all_items = [(t, l, True) for t, l in zip(oracle_texts, oracle_lbls)] + \
                        [(t, l, False) for t, l in zip(d_texts, d_lbls)]
            random.shuffle(all_items)

            docs = [it[0] for it in all_items]
            lbls = [it[1] for it in all_items]
            oracle_indices_0based = [i for i, it in enumerate(all_items) if it[2]]

            style = random.choice(COMPLETION_STYLES)

            thought, completion = generate_multi_thought_and_completion(
                question, docs, lbls, oracle_indices_0based, style
            )
            time.sleep(REQUEST_DELAY)

            if not thought or not completion:
                continue

            sample = {
                "instruction": question,
                "documents": docs,
                "thought_process": thought,
                "completion": completion,
                "metadata_extra": {
                    "question_type": q_type,
                    "answer_type": "abstractive",  # sintesis multi-dok selalu abstractive
                    "oracle_present": True,
                    "oracle_doc_id": doc_id,
                    "oracle_index": oracle_indices_0based,  # LIST, bukan int
                    "multi_oracle": True,
                    "grouping_strategy": strategy,
                },
            }

            validation_res = validate_multi_sample(sample)
            time.sleep(REQUEST_DELAY)

            if (
                validation_res
                and validation_res.get("pass", False)
                and validation_res.get("instruction_answered", False)
                and validation_res.get("grounded", False)
                and validation_res.get("uses_all_oracles", False)
                and validation_res.get("thought_correct", False)
            ):
                sample["validation"] = validation_res
                dataset.append(sample)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                success = True
                break

        if not success:
            failed += 1

    print(f"\nSelesai! Berhasil: {len(dataset)}, Gagal: {failed}")
    print(f"Tersimpan di: {output_path}")


if __name__ == "__main__":
    output = DATASET_DIR / "raft_dataset_multi_oracle.jsonl"
    # Tingkatkan max_samples_per_doc (defaultnya hanya 3) agar lebih banyak dataset yang di-generate
    run_multi_oracle_pipeline(output, max_samples_per_doc=15)