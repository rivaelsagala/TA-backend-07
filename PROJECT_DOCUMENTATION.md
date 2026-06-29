# Project Documentation

## RAG Peraturan Desa — Backend System

> **Tugas Akhir (Skripsi) | Sistem Tanya Jawab Dokumen Hukum Berbasis Retrieval-Augmented Generation**

---

# Project Overview

Peraturan Desa (Perdes) merupakan regulasi hukum yang mengatur kehidupan masyarakat desa, namun aksesibilitasnya sangat terbatas. Dokumen-dokumen ini tersimpan dalam format PDF dengan ratusan halaman, sehingga warga, aparat desa, maupun peneliti kesulitan menemukan informasi spesifik secara cepat dan akurat. Proyek ini hadir sebagai solusi berbasis kecerdasan buatan untuk menjawab permasalahan tersebut.

**RAG Peraturan Desa** adalah sistem backend untuk aplikasi tanya jawab cerdas (_intelligent Q&A_) terhadap dokumen peraturan desa menggunakan teknik **Retrieval-Augmented Generation (RAG)**. Sistem ini mampu menerima pertanyaan dalam bahasa alami (natural language) dari pengguna, mencari pasal atau bagian yang paling relevan dari ratusan dokumen PDF peraturan desa yang telah diindeks, lalu menghasilkan jawaban yang akurat, terstruktur, dan dapat dilacak ke sumber aslinya. Selain model standar, sistem ini juga mengintegrasikan **model RAFT** (_Retrieval-Augmented Fine-Tuning_) yang di-fine-tune khusus pada domain hukum peraturan desa untuk meningkatkan kualitas jawaban.

---

> [!WARNING]
> **CATATAN KETIDAKSESUAIAN DENGAN LAPORAN TA:** 
> Di dalam Abstrak Laporan TA, disebutkan sistem menggunakan **FAISS** dan **sentence-transformer** dengan jumlah **16 dokumen** (atau 126 di abstrak Inggris). Namun, implementasi kode backend & dokumentasi ini menggunakan **Supabase (pgvector)** dan **OpenAI 	ext-embedding-3-large**. Anda HARUS merevisi Laporan TA Anda agar sesuai dengan implementasi kode sesungguhnya.

# Technology Stack

| Kategori                    | Teknologi                                    | Fungsi                                                                       |
| --------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| **Web Framework**           | Flask 3.1 + Flask-CORS                       | REST API server dan routing HTTP                                             |
| **Production Server**       | Gunicorn                                     | WSGI server untuk deployment produksi                                        |
| **Vector Database**         | Supabase (pgvector)                          | Menyimpan embedding dokumen dan melakukan similarity search                  |
| **Relational Database**     | PostgreSQL (via Supabase)                    | Menyimpan users, sesi chat, riwayat percakapan, dan metrik evaluasi          |
| **Embedding Model**         | OpenAI `text-embedding-3-large`              | Mengubah teks dokumen/query menjadi vektor numerik (1536 dimensi)            |
| **LLM — Base Models**       | Llama 3.1 8B, Qwen2.5 7B, DeepSeek-R1 7B     | Model bahasa besar untuk generasi jawaban (via HuggingFace Router)           |
| **LLM — OpenAI-Compatible** | GPT-4o-mini, GPT-3.5-turbo, Gemini 2.0 Flash | Model cloud via Maia Router API                                              |
| **LLM — RAFT Model**        | `model_merged_raft_perdes`                   | Model fine-tuned khusus domain Perdes (di-host di B200 Server)               |
| **Reranker**                | BAAI/bge-reranker-v2-m3 (HuggingFace)        | Cross-encoder untuk re-ranking dokumen hasil retrieval                       |
| **PDF Extraction**          | PyMuPDF (fitz)                               | Ekstraksi teks dari file PDF berbasis teks                                   |
| **OCR**                     | pytesseract + Pillow + OpenCV                | Ekstraksi teks dari PDF hasil scan (gambar)                                  |
| **Orchestration**           | LangChain + LangChain-OpenAI                 | Orkestrasi pipeline RAG dan integrasi vector store                           |
| **Evaluation Framework**    | RAGAS                                        | Evaluasi kualitas sistem RAG secara otomatis (faithfulness, relevancy, dll.) |
| **Reranking Library**       | rank-bm25, Cohere                            | Hybrid search dan reranking dokumen                                          |
| **Database Driver**         | psycopg2-binary                              | Koneksi Python ke PostgreSQL                                                 |
| **Supabase Client**         | supabase-py                                  | Client resmi Python untuk Supabase (vector store + auth)                     |
| **Configuration**           | python-dotenv                                | Manajemen environment variables                                              |
| **Logging**                 | Loguru                                       | Structured logging dengan format yang informatif                             |
| **Containerization**        | Docker + Docker Compose                      | Deployment yang konsisten di berbagai environment                            |
| **Text Tokenizer**          | tiktoken                                     | Tokenisasi teks untuk kalkulasi panjang konteks                              |

---

# System Architecture

Sistem dibangun dengan arsitektur **Layered Architecture** yang terdiri dari empat lapisan utama:

```
┌─────────────────────────────────────────────────────────┐
│                   CLIENT / FRONTEND                     │
│               (React / Web Application)                 │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / REST API
┌───────────────────────▼─────────────────────────────────┐
│                  PRESENTATION LAYER                     │
│           Flask Routes + Request Handlers               │
│   (routes.py, chat_handler, pdf_ingest_handler, ...)    │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                   │
│                Use Cases + Services                     │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  chat_use_case  │  │      rag_service              │  │
│  │  (orchestrator) │  │  Query Rewriting → Retrieval  │  │
│  │                 │  │  → Reranking → Expansion →    │  │
│  │  + off-topic    │  │  Generation                   │  │
│  │    filter       │  └──────────────────────────────┘  │
│  └─────────────────┘  ┌──────────────────────────────┐  │
│                       │   preprocessing_service       │  │
│  ┌─────────────────┐  │  PDF Extract → Clean →       │  │
│  │  ragas_service  │  │  Metadata Extract → Chunk    │  │
│  │  (evaluator)    │  └──────────────────────────────┘  │
│  └─────────────────┘                                    │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    DATA LAYER                           │
│                                                         │
│  ┌──────────────────────┐  ┌───────────────────────┐    │
│  │ Supabase (pgvector)  │  │  PostgreSQL (psycopg2) │    │
│  │ Vector Store         │  │  users, chat_sessions, │    │
│  │ (documents table)    │  │  chat_history          │    │
│  └──────────────────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│               EXTERNAL SERVICES LAYER                   │
│                                                         │
│  HuggingFace Router │ Maia Router (OpenAI-compat)       │
│  B200 RAFT Server   │ HF Reranker API                   │
└─────────────────────────────────────────────────────────┘
```

Setiap lapisan memiliki tanggung jawab yang terdefinisi dengan jelas. Handler menerima dan memvalidasi request HTTP, Use Case mengorkestrasi alur bisnis, Service mengimplementasikan logika inti (RAG, evaluasi, preprocessing), dan Data Layer berinteraksi dengan database.

---

# System Workflow

## Overview: End-to-End RAG & RAFT Pipeline

Berikut adalah gambaran besar (_big picture_) alur kerja keseluruhan sistem, mulai dari pemrosesan dokumen mentah, pembuatan dataset dan fine-tuning RAFT, hingga proses tanya jawab dan evaluasi.

```text
(PDF Peraturan Desa)
                         │
                         ▼
                 Data Preprocessing
      (OCR, Cleaning, Metadata, Chunking)
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Embedding + Vector DB      Build RAFT Dataset
              │            (Question, Context, Answer)
              │                         │
              │                         ▼
              │                  LoRA Fine-Tuning
              │                         │
              │                         ▼
              │                  Fine-Tuned Model
              │                         │
      ┌───────┴───────┐                 │
      ▼               ▼                 │
User Query        User Query            │
      │               │                 │
      ▼               ▼                 │
Query Embedding   Query Embedding       │
      │               │                 │
      ▼               ▼                 │
Similarity Search Similarity Search     │
(Vector DB)       (Vector DB)           │
      │               │                 │
      ▼               ▼                 │
Retrieved Context Retrieved Context     │
      │               │                 │
      ▼               ▼                 │
 ┌─────────┐     ┌─────────┐            │
 │ Baseline│     │  RAFT   │◄───────────┘
 │   LLM   │     │  LLM    │ (Model disuntikkan ke sini)
 │ (Prompt)│     │ (Prompt)│
 └────┬────┘     └────┬────┘
      │               │
      ▼               ▼
 Baseline           RAFT
  Answer           Answer
      └───────┬───────┘
              ▼
    Evaluation Comparison
(Faithfulness, Answer Relevancy,
 Context Precision, Context Recall,
 Noise Sensitivity, Latency)
              │
              ▼
     Comparative Analysis
```

## A. Alur Ingestion Dokumen (Satu Kali Setup)

### Flowchart: End-to-End Ingestion Pipeline (PDF to Database)

Berikut adalah detail logika alur kerja (workflow) dari API ingestion yang melibatkan `embedding_use_case` dan `preprocessing_service`:

```text
[POST /api/generate-embedding]
          │
          ▼
ingest_pdf_to_vector_db(file_path, save_to_db)
          │
          ▼
extract_and_chunk_pdf(file_path, save_to_db) ─────────────────────┐
          │                                                       │
          ├─► extract_text_from_pdf()                             │
          │   (PyMuPDF / Fallback OCR)                            │
          │                                                       │
          ├─► chunk_documents(documents)                          │
          │   (Cleaning, Metadata, Structural Chunking)           │
          │                                                       │
          ├─► save_results_to_folder()                            │
          │   (Simpan file lokal .txt & .json untuk review)       │
          │                                                       │
          ├─► if save_to_db == True:                              │
          │       save_chunks_to_postgres()                       │
          │       (Simpan teks asli ke tabel chunks_perdes)       │
          │                                                       │
          └───────────────────────────────────────────────────────┘
          │
          ▼
   Metadata Injection
(Inject source_file, document_id,
 village_name, perdes_number, dll)
          │
          ▼
Is save_to_db == True?
          │
    ┌─────┴─────┐
   YES          NO
    │           │
    ▼           ▼
Check Existing  Return "PREVIEW MODE"
Document in DB  (Proses Selesai, skip DB)
    │
    ▼
If exists > 0:
delete_document_chunks(document_id)
(Hapus vektor/dokumen lama di DB)
    │
    ▼
store_chunks_to_supabase(chunks)
(Kirim ke OpenAI untuk Embedding,
lalu simpan ke tabel `documents`
di Supabase pgvector)
    │
    ▼
Return "SUCCESS"
```

### Detail Tahapan Ekstraksi & Chunking (Data Preprocessing)

```
1. [Input]        File PDF Peraturan Desa disediakan
       │
2. [Extract]      PyMuPDF membaca teks per halaman
       │          ↳ Jika halaman < 50 karakter → Fallback ke OCR (pytesseract)
       │
3. [Clean]        preprocessing_service.clean_legal_text()
       │
       │  TAHAP 1: Normalisasi karakter unicode & kontrol
       │          ↳ Konversi \xa0, \u2018/\u2019 (curly quotes), en/em dash, bullet
       │
       │  TAHAP 1.5: Perbaikan OCR spacing & typo
       │          ↳ _fix_ocr_spacing(): gabungkan huruf tunggal yang terpisah
       │            (kecuali preposisi valid: a, i, di, ke, si, …)
       │          ↳ Marker list yang terputus di newline disambungkan kembali
       │          ↳ Perbaikan typo spesifik preamble ("menginat" → "mengingat")
       │
       │  TAHAP 2: Konversi ke lowercase
       │
       │  TAHAP 3: Perbaikan OCR spaced-keywords
       │          ↳ "P A S A L" → "pasal", "B A B" → "bab", dst.
       │          ↳ Normalisasi separator ":" pada kata kunci preamble
       │
       │  TAHAP 4: Penghapusan karakter noise
       │          ↳ Tanda baca aneh, karakter kontrol, simbol copyright/IP
       │
       │  TAHAP 5: Normalisasi whitespace & penggabungan baris terputus
       │          ↳ _merge_broken_lines(): gabung baris tanpa punctuation akhir
       │            kecuali jika baris berikutnya diawali structural starter
       │
       │  TAHAP 6-7: Pembersihan artefak PDF & header/footer berulang
       │          ↳ Nomor halaman tunggal, "salinan", "lembaran desa", dst.
       │
       │  TAHAP 8: Penghapusan signature block
       │          ↳ Blok ttd, ditetapkan/diundangkan di, kepala desa, dst.
       │
       │  TAHAP 9: Penghapusan konjungsi tunggal di awal baris
       │          ↳ "dan", "atau", "serta", dll. yang berdiri sendiri
       │
       │  [Potong Lampiran] Teks setelah kalimat penutup resmi
       │          ↳ Regex fleksibel: "agar setiap orang ... mengetahuinya"
       │            lalu "menempatkannya dalam lembaran desa."
       │
4. [Metadata]     extract_perdes_metadata()
       │          ↳ Normalisasi spaced-keywords pada teks mentah
       │          ↳ Bangun header_block (25 baris pertama s.d. MENIMBANG/MENGINGAT)
       │          ↳ Ekstraksi nama desa: regex PERATURAN DESA ... NOMOR/NO
       │            Fallback: KEPALA/PEMERINTAH DESA
       │          ↳ Ekstraksi nomor & tahun: toleran titik/koma/spasi berlebih
       │            pada pola NOMOR/NO ... TAHUN
       │          ↳ Ekstraksi judul: regex TENTANG ... s.d. kata kunci berikutnya
       │            Fallback baris demi baris jika regex gagal
       │          ↳ Ekstraksi kabupaten/kota: frekuensi Counter() pada seluruh teks
       │          ↳ Pembentukan document_id unik (e.g. perdes_biru_07_2015)
       │
5. [Chunk]        chunk_documents() → _parse_perdes_sections()
       │          ↳ Deteksi awal konten pada "menetapkan" / "bab i" / "pasal 1"
       │          ↳ Regex pasal toleran: pasal \d+ dengan opsional titik/titik dua
       │          ↳ BAB: toleran angka romawi & arab, judul inline atau baris berikut
       │          ↳ Bagian: toleran spasi ("ke sepuluh" → "kesepuluh"), inline title
       │          ↳ Pasal dengan ≥2 ayat (\(\d+\)) → _split_by_ayat()
       │          ↳ Pasal dengan butir bernomor → _split_butir()
       │            (intro tanpa nomor diprependkan ke butir pertama)
       │          ↳ _normalize_letter_numbering(): huruf a–n → angka 1–14
       │            untuk dot (a.) dan paren (a)) dalam butir
       │          ↳ Setiap chunk diberi metadata: bab, pasal, ayat, chunk_index
       │            (bagian & bagian_title disertakan bila ada)
       │
6. [Embed]        OpenAI text-embedding-3-large (1536 dim)
       │          ↳ Setiap chunk dikonversi menjadi vektor numerik
       │
7. [Store]        Supabase pgvector (tabel: documents)
                  ↳ Disimpan: content, metadata (JSONB), embedding (vector)

Mode Opsional:
  save_to_db=False  → PREVIEW MODE: simpan .txt & .json lokal, lewati DB
  save_to_db=True   → Mode normal: simpan ke PostgreSQL (chunks_perdes)
```

## B. Alur Chat / Tanya Jawab (Real-time)

```
1. [Request]      User mengirim pertanyaan via POST /api/chat
       │          Payload: {session_id, user_id, question, model_id, evaluate}
       │
2. [Filter]       chat_use_case.py melakukan pre-filter:
       │          ↳ is_chitchat() → jawab salam langsung (rule-based)
       │          ↳ is_off_topic() → tolak pertanyaan di luar domain hukum
       │
3. [History]      Ambil riwayat percakapan terakhir (maks. 10 messages)
       │          dari PostgreSQL untuk konteks follow-up
       │
4. [Rewrite]      rewrite_query_with_history()
       │          ↳ Jika ada history, gunakan GPT-3.5-turbo untuk
       │            mengubah follow-up ambigu menjadi standalone query
       │          ↳ Contoh: "apa isi pasal itu?" → "apa isi pasal 14
       │            Perdes Desa Biru No.7 Tahun 2015 tentang BUMDes?"
       │
5. [Retrieve]     Supabase similarity_search(rewritten_query, k=20)
       │          ↳ Vector similarity search menggunakan cosine distance
       │
6. [Rerank]       reranker_service.rerank_documents(top_k=5)
       │          ↳ Cross-encoder BAAI/bge-reranker-v2-m3 menilai ulang
       │            relevansi 20 kandidat → pilih top 5
       │          ↳ Confidence threshold check: jika skor < -5.0, jawab
       │            "informasi tidak ditemukan" (tanpa hallucination)
       │
7. [Expand]       fetch_adjacent_chunks() (skip untuk RAFT model)
       │          ↳ Ambil chunk tetangga (i-1, i+1) untuk konteks lebih kaya
       │          ↳ Batch query ke Supabase berdasarkan (document_id, chunk_index)
       │
8. [Generate]     HuggingFaceService.chat_with_context()
       │          ↳ Bangun prompt: system prompt + context + history + question
       │          ↳ Kirim ke model yang dipilih (LLM / RAFT)
       │          ↳ RAFT: kirim raw chunks langsung ke B200 server
       │
9. [Evaluate]     (Opsional, jika evaluate=True pada sesi)
       │          ↳ RAGAS menghitung: faithfulness, answer_relevancy,
       │            context_precision, context_recall, noise_sensitivity
       │          ↳ SemanticAnswerSimilarity (SAS) menghitung cosine similarity
       │            antara answer dan ground_truth menggunakan embedding model
       │          ↳ Semua metrik dikembalikan dalam satu dict evaluation_result
       │
10. [Save]        Simpan ke PostgreSQL: pertanyaan, jawaban, sources,
       │          metadata, skor RAGAS + semantic_similarity
       │
11. [Response]    Kembalikan JSON ke frontend:
                  {answer, sources, evaluation, model_used}
                  evaluation: {faithfulness, answer_relevancy,
                               context_precision, context_recall,
                               noise_sensitivity, semantic_similarity}
```

## C. Alur Chat dengan Model Fine-Tune RAFT

```
1–6. [Sama dengan Alur B]
       ├── Request → Filter → History → Rewrite → Retrieve → Rerank
       │
       │   Deteksi Tipe Model:
       │   model_info = AVAILABLE_MODELS.get(model_id=8)
       │   → model_type == "raft"  → masuk jalur RAFT
       │
7. [SKIP Expand]   Adjacent Chunk Expansion DILEWATI
       │          ↳ RAFT melakukan reasoning internal terhadap dokumen,
       │            tidak memerlukan konteks tetangga dari luar
       │          ↳ adjacent_map = {} (kosong, tidak di-fetch)
       │
8. [Prepare]      Siapkan raw_doc_chunks dari hasil reranking (top-5)
       │          ↳ raw_doc_chunks = [doc.page_content for doc in reranked_docs]
       │          ↳ Ini adalah list string individual (BELUM di-join),
       │            setiap elemen = 1 chunk pasal/ayat
       │
9. [Generate]     HuggingFaceService.chat_with_context()
       │          ↳ Karena is_raft = True:
       │            • Tidak dibangun system_prompt
       │            • Tidak dibangun joined context
       │            • messages = [{"role": "user", "content": user_question}]
       │
       │  ┌────────────────────────────────────────────────────────┐
       │  │         POST ke B200 RAFT Server                       │
       │  │  URL: {FINETUNED_API_URL}/api/chat-rag                 │
       │  │  Payload:                                              │
       │  │  {                                                     │
       │  │    "pertanyaan": "<user_question>",                    │
       │  │    "dokumen": ["<chunk1>", "<chunk2>", ..., "<chunk5>"]│
       │  │  }                                                     │
       │  └────────────────────────────────────────────────────────┘
       │
       │  ┌────────────────────────────────────────────────────────┐
       │  │         Response dari B200 RAFT Server                 │
       │  │  {                                                     │
       │  │    "jawaban": "<final_answer>",                        │
       │  │    "analisis": "<thought_process / chain-of-thought>", │
       │  │    "raw_response": "<full_model_output>",              │
       │  │    "num_documents": 5,                                 │
       │  │    "model_type": "raft",                               │
       │  │    "status": "success"                                 │
       │  │  }                                                     │
       │  └────────────────────────────────────────────────────────┘
       │
       │  ↳ Response di-standardisasi ke format OpenAI-compatible:
       │    {"choices": [{"message": {"content": jawaban}}],
       │     "raft_metadata": {"analisis": ..., "num_documents": ...}}
       │
10. [Evaluate]    (Opsional, jika evaluate=True pada sesi)
       │          ↳ RAGAS + SAS menghitung metrik sama seperti Alur B
       │          ↳ contexts diambil dari raw_doc_chunks (bukan expanded)
       │
11. [Save]        Simpan ke PostgreSQL dengan field tambahan:
       │          ↳ metadata["analysis"] = raft_metadata["analisis"]
       │            (thought process RAFT disimpan di kolom metadata JSONB)
       │
12. [Response]    Kembalikan JSON ke frontend dengan field tambahan:
                  {
                    answer, sources, evaluation, model_used,
                    "analysis": "<thought_process dari RAFT>"  ← KHUSUS RAFT
                    evaluation: {faithfulness, answer_relevancy,
                                 context_precision, context_recall,
                                 noise_sensitivity, semantic_similarity}
                  }
```

### Perbandingan Alur B (Model Standar) vs Alur C (RAFT)

| Tahapan                    | Model Standar (Alur B)                     | RAFT Model (Alur C)                        |
| -------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Adjacent Expansion**     | ✅ Dijalankan (ambil chunk tetangga)       | ❌ Dilewati                                |
| **System Prompt**          | ✅ Dibangun (instruksi + konteks gabungan) | ❌ Tidak ada                               |
| **Format Input ke LLM**    | Joined context string dalam system prompt  | List raw chunks terpisah (`"dokumen"`)     |
| **Endpoint LLM**           | HuggingFace Router / Maia Router           | B200 RAFT Server (`/api/chat-rag`)         |
| **Field Respons Tambahan** | —                                          | `"analysis"` (chain-of-thought thinking)   |
| **Chat History Konteks**   | ✅ Dikirim ke LLM                          | ❌ Tidak dikirim (RAFT stateless per-call) |

---

## E. Alur Sistem Evaluasi (Evaluation System Workflow)

> Evaluasi dijalankan **setelah generation selesai** dan hanya aktif jika flag `evaluate=True` pada sesi. Dua komponen berjalan berurutan: **RAGAS** (metrik berbasis LLM) dan **Semantic Answer Similarity / SAS** (metrik berbasis embedding).

```
[Trigger]   evaluate=True pada sesi + ground_truth dari request payload
     │
     │  Input yang tersedia setelah Tahap Generate (Alur B/C):
     │  • question      → pertanyaan user asli
     │  • answer        → jawaban dari LLM / RAFT
     │  • contexts      → list string chunk dari top-5 reranked docs
     │  • ground_truth  → jawaban pakar (dari payload; fallback ke answer jika None)
     │
─────┴──────────────────────────────────────────────────────────────────
 KOMPONEN 1 — RAGAS (ragas_service.evaluate_single_response)
─────────────────────────────────────────────────────────────────────────
     │
1. [Dataset]   Siapkan HuggingFace Dataset dari satu baris data:
     │          {question, contexts, answer, ground_truth}
     │
2. [LLM Judge] Kirim ke RAGAS evaluate() via LangchainLLMWrapper:
     │          Model: openai/gpt-3.5-turbo-16k (temperature=0.0)
     │          Role : juri absolut, tidak berubah antar evaluasi
     │
     │  ┌─────────────────────────────────────────────────────────────┐
     │  │ METRIK YANG DIHITUNG:                                       │
     │  │                                                             │
     │  │ ① faithfulness (0–1)                                       │
     │  │   Input : answer + contexts                                 │
     │  │   Ukur  : apakah setiap klaim di answer didukung konteks?  │
     │  │   Tinggi = jawaban grounded, tidak mengarang               │
     │  │                                                             │
     │  │ ② answer_relevancy (0–1)                                   │
     │  │   Input : question + answer                                 │
     │  │   Ukur  : apakah answer relevan dan menjawab question?     │
     │  │   Tinggi = jawaban tepat sasaran                           │
     │  │                                                             │
     │  │ ③ context_precision (0–1)                                  │
     │  │   Input : question + contexts + ground_truth               │
     │  │   Ukur  : seberapa presisi konteks? (chunk relevan di atas) │
     │  │   Tinggi = konteks tidak banyak noise                      │
     │  │                                                             │
     │  │ ④ context_recall (0–1)                                     │
     │  │   Input : contexts + ground_truth                          │
     │  │   Ukur  : seberapa banyak info ground_truth ada di konteks?│
     │  │   Tinggi = konteks lengkap mencakup referensi pakar        │
     │  │   ⚠️ Akurat hanya jika ground_truth adalah jawaban pakar   │
     │  │                                                             │
     │  │ ⑤ noise_sensitivity (0–1)                                  │
     │  │   Input : question + answer + contexts + ground_truth      │
     │  │   Ukur  : apakah model terpengaruh konteks tidak relevan?  │
     │  │   Rendah = model lebih robust terhadap noise               │
     │  └─────────────────────────────────────────────────────────────┘
     │
3. [Result]    df_results = evaluation_result.to_pandas()
     │          formatted_result = {faithfulness, answer_relevancy,
     │                              context_precision, context_recall,
     │                              noise_sensitivity}
     │
─────┴──────────────────────────────────────────────────────────────────
 KOMPONEN 2 — SAS: Semantic Answer Similarity
             (SemanticAnswerSimilarity.compute_sas)
─────────────────────────────────────────────────────────────────────────
     │
4. [Embed]     Vectorisasi answer + ground_truth dalam SATU API call:
     │          embed_documents([answer, ground_truth])
     │          Model: openai/text-embedding-3-large (1536 dim)
     │
5. [Cosine]    Hitung cosine similarity antara dua vektor:
     │          score = dot(vec_a, vec_b) / (norm_a × norm_b)
     │          Clip ke [0.0, 1.0] agar konsisten
     │
6. [Merge]     formatted_result["semantic_similarity"] = sas_score
     │          → Satu dict evaluasi lengkap dengan 6 metrik
     │
─────┴──────────────────────────────────────────────────────────────────
 PERSISTENSI — Simpan ke Database
─────────────────────────────────────────────────────────────────────────
     │
7. [Save]      chat_service.save_chat_message(evaluation=evaluation_result)
     │
     │  Mapping key Python → kolom PostgreSQL:
     │  ┌──────────────────────┬────────────────────────────────────┐
     │  │ evaluation dict key  │ Kolom chat_history (FLOAT)         │
     │  ├──────────────────────┼────────────────────────────────────┤
     │  │ "faithfulness"       │ faithfulness                       │
     │  │ "answer_relevancy"   │ answer_relevance                   │
     │  │ "context_precision"  │ context_precision                  │
     │  │ "context_recall"     │ context_recall                     │
     │  │ "noise_sensitivity"  │ noise_sensitivity                  │
     │  │ "semantic_similarity"│ semantic_similarity                │
     │  └──────────────────────┴────────────────────────────────────┘
     │
     │  Jika evaluate=False → semua kolom evaluasi disimpan sebagai NULL
     │
8. [Response]  evaluation_result dikembalikan ke frontend dalam field
               "evaluation" pada response JSON
```

### Tabel Ringkasan Metrik Evaluasi

| Metrik                  | Rentang | Input Wajib                              | Memerlukan Ground Truth Pakar | Keterangan Singkat                                           |
| ----------------------- | ------- | ---------------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| **faithfulness**        | 0–1     | answer, contexts                         | ❌                            | Klaim di jawaban harus didukung konteks                      |
| **answer_relevancy**    | 0–1     | question, answer                         | ❌                            | Jawaban harus relevan dan menjawab pertanyaan                |
| **context_precision**   | 0–1     | question, contexts, ground_truth         | ⚠️ Sebaiknya ada              | Chunk relevan harus di ranking atas                          |
| **context_recall**      | 0–1     | contexts, ground_truth                   | ✅ Wajib                      | Konteks harus mencakup info dari jawaban pakar               |
| **noise_sensitivity**   | 0–1     | question, answer, contexts, ground_truth | ✅ Wajib                      | Rendah lebih baik: model tidak terpengaruh noise             |
| **semantic_similarity** | 0–1     | answer, ground_truth                     | ✅ Wajib                      | Cosine similarity embedding: kesamaan makna jawaban vs pakar |

### Penanganan Kasus ground_truth Tidak Diberikan

```
Jika ground_truth = None pada request payload:
  → ragas_service memberi WARNING di log
  → ground_truth di-fallback ke answer (jawaban LLM itu sendiri)
  → Konsekuensi:
      • context_recall   → TIDAK VALID (mengukur konteks vs jawaban LLM, bukan pakar)
      • noise_sensitivity → TIDAK VALID (menggunakan jawaban LLM sebagai referensi)
      • semantic_similarity → SELALU 1.0 (answer == ground_truth)
  → Metrik faithfulness & answer_relevancy TETAP VALID
```

### Konfigurasi LLM Judge & Embedding (ragas_service.py)

| Komponen               | Model                                                | Parameter Kunci             | Tujuan                                |
| ---------------------- | ---------------------------------------------------- | --------------------------- | ------------------------------------- |
| **LLM Judge (RAGAS)**  | `openai/gpt-3.5-turbo-16k`                           | `temperature=0.0`, max 2000 | Juri stabil & deterministik           |
| **Embeddings (RAGAS)** | `openai/text-embedding-3-large`                      | 1536 dimensi                | Representasi semantik untuk relevancy |
| **SAS Embedder**       | `openai/text-embedding-3-large`                      | Sama dengan RAGAS embedder  | Cosine similarity answer vs truth     |
| **Wrapper**            | `LangchainLLMWrapper` + `LangchainEmbeddingsWrapper` | —                           | Standardisasi format JSON RAGAS       |

---

## D. Alur Pembuatan Dataset RAFT v4 (`notebooks/generate_raft.py`)

> **V4 — Pembaruan Utama:** Completion Natural (kutipan hanya di thought_process), Answerability Check, Hard Negative baru (similarity 0.75–0.95), distribusi dominan Pasal & Ayat Lookup, MULTIHOP_RATIO diturunkan ke 10%.

```
[Input]  File *_chunks.json dari folder data/processed/
    │
1. [Cache Check]   Cek data/cache/chunk_embeddings.pkl
    │              ↳ Jika ada → langsung load (hemat waktu & biaya API)
    │              ↳ Jika belum → embed & simpan cache
    │
2. [Load & Filter] Baca *_chunks.json, filter is_substantive()
    │              ↳ Chunk harus ≥ 200 karakter & minimal 2 kalimat valid
    │
3. [Embed]         Vectorisasi via OpenAI API (text-embedding-3-large)
    │              ↳ Batch 64 teks | Retry 3x | Normalisasi L2
    │              ↳ Simpan ke data/cache/chunk_embeddings.pkl
    │
4. [Multi-hop?]    MULTIHOP_RATIO = 10% (diturunkan dari 30%)
    │              ↳ Pasangan chunk: Cosine Similarity 0.40–0.80
    │              ↳ Jika tidak ada pasangan valid → is_multihop = False
    │
5. [Generate Q]    generate_question() via GPT-4o-mini
    │
    │   TIPE PERTANYAAN (diacak berbobot — dominan pasal/ayat lookup):
    │   ┌─────────────────┬───────┬──────────────────────────────────────┐
    │   │ Tipe            │ Bobot │ Gaya Jawaban                         │
    │   ├─────────────────┼───────┼──────────────────────────────────────┤
    │   │ faktual         │  15   │ langsung                             │
    │   │ definisional    │  10   │ penjelasan                           │
    │   │ prosedural      │  15   │ terstruktur                          │
    │   │ kondisional     │  10   │ langsung                             │
    │   │ enumeratif      │  10   │ terstruktur                          │
    │   │ pasal_lookup    │  20   │ formal_hukum  ← bobot terbesar       │
    │   │ ayat_lookup     │  10   │ formal_hukum                         │
    │   │ komparatif      │   5   │ terstruktur                          │
    │   │ reasoning_based │   5   │ percakapan / penjelasan              │
    │   └─────────────────┴───────┴──────────────────────────────────────┘
    │   ↳ pasal_lookup & ayat_lookup selalu pakai gaya "legal"
    │   ↳ natural query DILARANG sebut nama/tahun peraturan
    │
6. [Answerability] check_answerability() — BARU di V4
    │              ↳ LLM mengevaluasi apakah pertanyaan BISA dijawab
    │                sepenuhnya menggunakan oracle chunk
    │              ↳ Jika TIDAK → buang (hitung sebagai unanswerable)
    │              ↳ Mencegah pertanyaan ambigu masuk dataset
    │
7. [Distractor]    select_semantic_distractors() — 3 tipe negative
    │              ↳ normal         → similarity 0.40–0.70
    │              ↳ hard_negative  → similarity 0.75–0.95 (BARU)
    │                                 jebakan paling kuat, topik hampir sama
    │              ↳ near_miss      → similarity 0.60–0.75
    │              ↳ completely_absent → similarity < 0.40
    │
8. [Oracle Logic]  80/20 split (sesuai RAFT Paper)
    │              ↳ 80% oracle_present=True: oracle disisipkan acak
    │              ↳ 20% oracle_present=False: pilih dari
    │                  hard_negative / near_miss / completely_absent
    │
9. [Generate A]    generate_thought_and_completion() via GPT-4o-mini
    │
    │   { "thought_process": "...", "completion": "..." }
    │
    │   Oracle-PRESENT (PERUBAHAN V4):
    │   • thought_process: analisis dokumen + WAJIB kutipan <<teks asli>>
    │   • completion: jawaban NATURAL, DILARANG ada <<kutipan>> di sini
    │
    │   Oracle-ABSENT:
    │   • thought_process: buktikan ketiadaan jawaban
    │   • completion: nyatakan tidak tersedia secara natural
    │
10. [Quality Gate] is_bad_sample() — diperbarui V4
    │              ↳ TOLAK jika completion menyebut "Dokumen N"
    │              ↳ TOLAK jika completion mengandung <<...>> (raw quote)
    │              ↳ TOLAK jika oracle_present=True tapi thought
    │                tidak memiliki kutipan <<...>>
    │              → Sampel tidak lolos: DISKIP
    │
11. [Save]         Simpan ke data/dataset/raft_dataset_v4_production.jsonl
                   ↳ Format JSONL | Append per sampel (mode append, tidak overwrite)
                   ↳ Auto-backup file lama: raft_backup_YYYYMMDD_HHMMSS.jsonl
                   ↳ Counter: ok | unanswerable | fail | skip
```

### Format Sampel Dataset RAFT v4

```json
{
  "instruction": "Apa fungsi BPD menurut peraturan desa ini?",
  "documents": [
    "pasal 7\nkewenangan lokal...",
    "pasal 1\nbpd adalah lembaga..."
  ],
  "thought_process": "Pasal 1 relevan karena mendefinisikan BPD: <<badan permusyawaratan desa adalah lembaga yang melaksanakan fungsi pemerintahan>>. Pasal 7 membahas kewenangan, tidak relevan.",
  "completion": "BPD berfungsi sebagai lembaga pemerintahan desa yang anggotanya merupakan wakil penduduk berdasarkan keterwakilan wilayah.",
  "metadata_extra": {
    "query_style": "natural",
    "question_type": "pasal_lookup",
    "multi_hop": false,
    "oracle_present": true,
    "negative_type": "none",
    "style": "formal_hukum",
    "evidence_docs": [2]
  }
}
```

> Perhatikan: pada V4 `completion` tidak mengandung `<<kutipan>>` — jawaban harus natural. Kutipan **hanya** ada di `thought_process`.

### Keputusan Desain Penting (V4)

| Keputusan                                          | Alasan                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Kutipan hanya di `thought_process`**             | Completion yang natural lebih sesuai untuk fine-tuning model yang digunakan dalam percakapan nyata      |
| **`check_answerability()` sebelum distractor**     | Mencegah pertanyaan yang tidak bisa dijawab masuk dataset; kualitas > kuantitas                         |
| **`hard_negative` (0.75–0.95)**                    | Distractor yang sangat mirip secara semantik melatih model untuk membedakan dokumen yang hampir identik |
| **`reasoning_based` menggantikan `interpretatif`** | Studi kasus what-if lebih menantang dan mencerminkan kebutuhan nyata dibanding pertanyaan "mengapa"     |
| **`pasal_lookup` bobot terbesar (20)**             | Dataset didominasi pertanyaan lookup agar model terlatih menjawab pertanyaan pasal spesifik             |
| **MULTIHOP_RATIO diturunkan ke 10%**               | Multi-hop terlalu sering menghasilkan konteks yang tidak koheren; 10% lebih realistis                   |
| **Embedding Cache (`.pkl`)**                       | Embedding 500+ chunk memakan biaya API; cache mencegah re-embed setiap run                              |
| **Oracle Hint `[INTERNAL]`**                       | LLM diberitahu posisi oracle via prompt internal yang tidak masuk ke output final                       |

# Key Features

### 1. 🔍 Multi-Stage RAG Pipeline

Pipeline RAG yang berlapis dan sistematis: **Retrieval (k=20) → Re-ranking (k=5) → Adjacent Chunk Expansion → Generation**.

- **Tahap 1 (Retrieval):** Mengambil 20 kandidat dokumen (chunk) teratas yang secara semantik mirip dengan pertanyaan.
- **Tahap 2 (Re-ranking):** AI Cross-Encoder menilai ulang 20 kandidat tersebut secara ketat, lalu membuang 15 terbawah dan hanya menyisakan **5 pemenang (Top 5)**.
- **Tahap 3 (Expansion):** Khusus untuk 5 pemenang ini, sistem menarik paragraf tetangganya (sebelum & sesudah) untuk melengkapi teks yang terpotong.
  Setiap tahap berfokus pada efisiensi dan akurasi, memastikan LLM hanya menerima konteks yang paling relevan namun utuh.

### 2. 📝 Legal Document-Aware Chunking (Semantic & Recursive)

Chunking tidak dilakukan dengan ukuran karakter biasa (fixed-size), melainkan dengan **memahami struktur hukum dokumen** (BAB → Pasal → Ayat → Butir). Di dalam implementasi (meskipun di laporan TA disebutkan RecursiveCharacterTextSplitter), sistem menggunakan pendekatan hibrida *Semantic Legal-Aware Chunking* (_parse_perdes_sections). Setiap chunk merupakan satu unit logis peraturan yang atomic, sehingga retrieval lebih presisi dan jawaban dapat menyebut pasal secara spesifik.

### 3. 🔄 Query Rewriting untuk Follow-up Questions

Sistem mendeteksi dan menyelesaikan pertanyaan lanjutan yang ambigu secara otomatis menggunakan LLM ringan (GPT-3.5-turbo). Query "apa isi pasal itu?" akan diubah menjadi query yang mandiri dan spesifik sebelum masuk ke pipeline retrieval.

### 4. 🤖 Multi-Model Support

Sistem mendukung **8 model LLM yang dapat dipilih**, mencakup model open-source (Llama, Qwen, DeepSeek), model cloud (GPT-4o-mini, Gemini 2.0 Flash), dan model RAFT fine-tuned khusus domain peraturan desa. Frontend dapat memilih model per sesi.

### 5. 🧠 RAFT Model Integration

Terintegrasi dengan model fine-tuned berbasis **RAFT (Retrieval-Augmented Fine-Tuning)** yang di-host pada server GPU dedicated (B200). Model ini dilatih khusus untuk menjawab pertanyaan hukum peraturan desa, mampu melakukan reasoning internal terhadap dokumen, dan menghasilkan analisis berpikir (_thought process_) yang dapat ditampilkan ke pengguna.

### 6. 📊 Automated RAGAS + Semantic Similarity Evaluation

Evaluasi kualitas RAG dilakukan secara otomatis menggunakan dua komponen yang berjalan berurutan:

- **RAGAS Framework** — menghitung 5 metrik berbasis LLM Judge (GPT-3.5-turbo): Faithfulness, Answer Relevancy, Context Precision, Context Recall, dan Noise Sensitivity.
- **Semantic Answer Similarity (SAS)** — menghitung cosine similarity antara embedding `answer` dan `ground_truth` menggunakan `text-embedding-3-large`, menghasilkan skor kesamaan makna (0–1) yang tidak bergantung pada format atau tanda baca.

Evaluasi berjalan per sesi (dikontrol flag `evaluate` di `chat_sessions`), dan seluruh 6 metrik tersimpan sebagai kolom FLOAT di tabel `chat_history` untuk keperluan analisis dan skripsi.

### 7. 🔒 Topic Guard & Confidence Threshold

Sistem memiliki dua lapisan filter: **Off-Topic Filter** (berbasis keyword) menolak pertanyaan di luar domain hukum desa, dan **Confidence Threshold** (skor reranker < -5.0) mencegah model mengarang jawaban (_hallucination_) saat tidak ada dokumen yang relevan.

### 8. 📂 Smart Adjacent Chunk Expansion

Sistem mengatasi masalah teks terpotong (akibat proses _chunking_ di awal) dengan fitur **Ekspansi Chunk Bertetangga**. Setelah proses _Re-ranking_ menghasilkan **5 chunk pemenang (Top 5)**, sistem secara otomatis kembali ke database untuk mengambil 1 chunk tepat **sebelum** dan 1 chunk tepat **sesudah** dari masing-masing pemenang tersebut.

Hasilnya, LLM tidak hanya membaca 1 potongan kecil teks, melainkan membaca **5 blok teks yang utuh**. Karena masing-masing dari 5 pemenang ini diekspansi menjadi 3 bagian, total teks yang disuapkan ke LLM setara dengan 15 chunk dokumen (5 blok × 3 chunk).

**Visualisasi Output Tahapan ke LLM:**

- **Blok 1 (Top 1 Re-ranking):** Berisi `[Konteks Sebelumnya]` + `[Bagian Utama]` + `[Konteks Berikutnya]`
- **Blok 2 (Top 2 Re-ranking):** Berisi `[Konteks Sebelumnya]` + `[Bagian Utama]` + `[Konteks Berikutnya]`
- **Blok 3 (Top 3 Re-ranking):** Berisi `[Konteks Sebelumnya]` + `[Bagian Utama]` + `[Konteks Berikutnya]`
- **Blok 4 (Top 4 Re-ranking):** Berisi `[Konteks Sebelumnya]` + `[Bagian Utama]` + `[Konteks Berikutnya]`
- **Blok 5 (Top 5 Re-ranking):** Berisi `[Konteks Sebelumnya]` + `[Bagian Utama]` + `[Konteks Berikutnya]`

**Contoh Bentuk Fisik 1 Blok Teks:**

```text
[Sumber: Peraturan Desa Majasetra No. 1 Tahun 2018] [Desa: majasetra]

[Konteks Sebelumnya — pasal/butir sebelumnya]
BAB I KETENTUAN UMUM
Pasal 1
Dalam Peraturan Desa ini yang dimaksud dengan :

[Bagian Utama]
5. Desa adalah kesatuan masyarakat hukum yang memiliki batas wilayah yang berwenang untuk mengatur dan mengurus urusan pemerintahan...

[Konteks Berikutnya — pasal/butir selanjutnya]
6. Pemerintahan Desa adalah penyelenggaraan urusan pemerintahan dan kepentingan masyarakat setempat...
```

Dengan struktur seperti ini, pemahaman hierarki LLM menjadi sempurna dan mampu mengutip sumber dengan sangat spesifik.

### 9. 💬 Persistent Chat Session & History

Sistem mendukung **manajemen sesi percakapan** yang tersimpan di PostgreSQL. Setiap sesi menyimpan riwayat percakapan lengkap, flag evaluasi RAGAS, dan metadata model yang digunakan. Riwayat ini juga digunakan sebagai konteks dinamis oleh LLM.

### 10. 📄 PDF Processing dengan OCR Fallback

Ekstraksi teks dari PDF menggunakan PyMuPDF sebagai metode utama (cepat & akurat). Jika halaman terdeteksi sebagai hasil scan (< 50 karakter teks), sistem **otomatis fallback ke OCR** menggunakan pytesseract dengan bahasa Indonesia.

---

# Database Design

Sistem menggunakan dua database yang terpisah berdasarkan fungsinya:

## PostgreSQL — Relasional (Data Transaksional)

| Tabel               | Kolom Utama                                                                                                                                                                                                            | Peran                                                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`users`**         | `id`, `username`, `password`, `created_at`                                                                                                                                                                             | Menyimpan data autentikasi pengguna sistem                                                                                                                                                      |
| **`chat_sessions`** | `id`, `user_id`, `session_name`, `evaluate`, `created_at`                                                                                                                                                              | Merepresentasikan satu sesi percakapan. Field `evaluate` (BOOLEAN) menentukan apakah evaluasi RAGAS aktif untuk sesi tersebut                                                                   |
| **`chat_history`**  | `id`, `session_id`, `user_id`, `user_query`, `llm_response`, `metadata` (JSONB), `is_evaluated`, `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`, `noise_sensitivity`, `semantic_similarity` | Menyimpan setiap pasang tanya-jawab beserta seluruh metrik evaluasi RAGAS + SAS dalam kolom bertipe FLOAT yang terstruktur. Kolom evaluasi bernilai NULL jika sesi tidak mengaktifkan evaluasi. |
| **`chunks_perdes`** | `id`, `file_name`, `content`, `metadata` (JSONB), `created_at`                                                                                                                                                         | Tabel opsional untuk menyimpan hasil chunking dokumen di sisi PostgreSQL                                                                                                                        |

## Supabase (PostgreSQL + pgvector) — Vector Store

| Tabel           | Kolom Utama                                                                                | Peran                                                                                                                                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`documents`** | `id` (UUID), `content` (TEXT), `metadata` (JSONB), `embedding` (vector 1536), `created_at` | Tabel utama vector store. Menyimpan setiap chunk dokumen beserta embedding-nya untuk similarity search. Metadata JSONB menyimpan informasi struktural (document_id, village_name, chunk_index, pasal, bab, dll.) |

**SQL Function kustom `match_documents`** dibuat di Supabase untuk menjalankan similarity search berbasis cosine distance (`<=>`) yang digunakan oleh LangChain `SupabaseVectorStore`.

---

# API / Integration

## REST API Endpoints (Flask)

| Method   | Endpoint                          | Fungsi                                                                                        |
| -------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `POST`   | `/api/chat`                       | Endpoint utama: menerima pertanyaan user, menjalankan pipeline RAG, dan mengembalikan jawaban |
| `POST`   | `/api/chat-sessions`              | Membuat sesi percakapan baru dengan opsi nama sesi dan flag evaluasi                          |
| `GET`    | `/api/chat-sessions`              | Mengambil daftar semua sesi percakapan milik user                                             |
| `GET`    | `/api/chat-history/<session_id>`  | Mengambil riwayat percakapan lengkap dari sesi tertentu                                       |
| `PUT`    | `/api/chat-sessions/<session_id>` | Memperbarui nama sesi atau status flag evaluasi RAGAS                                         |
| `DELETE` | `/api/chat-sessions/<session_id>` | Menghapus sesi dan seluruh riwayat terkait (CASCADE)                                          |
| `POST`   | `/api/generate-embedding`         | Memproses dan mengingesti file PDF ke vector store Supabase                                   |
| `GET`    | `/api/models`                     | Mengambil informasi daftar model yang tersedia                                                |
| `GET`    | `/api/user/<id>`                  | Mengambil informasi data pengguna                                                             |

## External Service Integrations

| Layanan                    | Protokol                      | Fungsi                                                         |
| -------------------------- | ----------------------------- | -------------------------------------------------------------- |
| **Supabase**               | `supabase-py` SDK             | Vector store (pgvector) dan query similarity search            |
| **HuggingFace Router API** | HTTP POST (OpenAI-compatible) | Inference model LLM open-source (Llama, Qwen, DeepSeek)        |
| **Maia Router API**        | HTTP POST (OpenAI-compatible) | Inference model cloud (GPT-4o-mini, GPT-3.5-turbo, Gemini)     |
| **B200 RAFT Server**       | HTTP POST (`/api/chat-rag`)   | Inference model RAFT fine-tuned lokal di server GPU dedicated  |
| **HF Reranker API**        | HTTP POST                     | Cross-encoder BAAI/bge-reranker-v2-m3 untuk re-ranking dokumen |

---

# Challenges & Solutions

### 1. Variasi Format PDF Peraturan Desa yang Tidak Konsisten

**Tantangan:** Ratusan dokumen PDF peraturan desa berasal dari berbagai desa dengan format yang sangat berbeda — ada yang berbasis teks, ada yang hasil scan, layout kolom berbeda, dan banyak typo/OCR error (misalnya, "PASAL" menjadi "P A S A L" atau "pasl", serta marker list yang terputus di newline).

**Solusi:** Membangun `preprocessing_service.py` yang komprehensif dengan multi-stage cleaning pipeline (9 tahap): normalisasi karakter unicode, perbaikan _OCR spacing error_ via `_fix_ocr_spacing()`, deteksi dan perbaikan _spaced keywords_ menggunakan regex, fallback OCR otomatis berbasis PyMuPDF + pytesseract, serta penggabungan baris terputus via `_merge_broken_lines()`. Metadata diekstrak menggunakan `header_block` approach (25 baris pertama digabungkan menjadi satu string) agar regex tidak bergantung pada pemisahan baris yang tidak konsisten akibat layout PDF.

---

### 2. Chunking Struktur Hukum yang Akurat

**Tantangan:** Chunking berbasis fixed-size (karakter/token) memotong dokumen secara sembarang, menghasilkan chunk yang terpotong di tengah pasal atau menggabungkan dua pasal berbeda dalam satu chunk. Variasi format juga menyebabkan regex kaku gagal mendeteksi BAB/Pasal yang memiliki titik, tanda titik dua, atau judul inline.

**Solusi:** Implementasi `_parse_perdes_sections()` yang memahami hierarki dokumen hukum (BAB → Pasal → Ayat → Butir) dengan regex yang toleran terhadap variasi punctuation (`[\.\:\s]+`). Fungsi pembantu `_split_by_ayat()` memisahkan pasal multi-ayat `(\d+)`, `_split_butir()` memisahkan butir bernomor (dengan prepend intro ke butir pertama jika ada), dan `_normalize_letter_numbering()` mengonversi penomoran huruf (a., b., a), b)) menjadi angka agar format konsisten di seluruh dataset.

---

### 3. Follow-up Questions yang Ambigu

**Tantangan:** Pertanyaan lanjutan seperti "apa sanksinya?" atau "jelaskan lebih detail" tidak dapat di-retrieve secara akurat karena tidak mengandung konteks yang cukup untuk vector similarity search.

**Solusi:** Implementasi **Query Rewriting** menggunakan LLM ringan (GPT-3.5-turbo) yang menganalisis riwayat percakapan dan menulis ulang pertanyaan ambigu menjadi _standalone query_ yang spesifik. Query rewritten digunakan untuk retrieval, sementara query asli tetap digunakan untuk generation agar jawaban sesuai dengan apa yang diminta pengguna.

---

### 4. Jawaban Tidak Relevan / Hallucination

**Tantangan:** Model LLM cenderung mengarang jawaban (_hallucinate_) saat tidak ada dokumen yang benar-benar relevan dengan pertanyaan di vector database.

**Solusi:** Implementasi **Confidence Threshold** berbasis skor cross-encoder (MS Marco BAAI/bge-reranker-v2-m3). Jika skor relevansi tertinggi dari hasil re-ranking berada di bawah threshold (-5.0), sistem langsung mengembalikan respons "tidak ditemukan" tanpa meneruskan ke LLM, sehingga hallucination dapat dicegah sepenuhnya.

---

### 5. Integrasi RAFT Model dengan Format Response yang Berbeda

**Tantangan:** Model RAFT fine-tuned memiliki format input/output yang berbeda dari model standar (tidak menggunakan system prompt, menerima daftar dokumen mentah, menghasilkan `jawaban` + `analisis` bukan `choices[0].message.content`).

**Solusi:** Implementasi `HuggingFaceService` dengan **abstraksi multi-model** yang mendeteksi tipe model (`raft`, `openai`, `original`) dan menerapkan routing logic yang sesuai. Response RAFT di-standardisasi ke format OpenAI-compatible sebelum dikembalikan, dan metadata analisis disimpan secara terpisah melalui `_raft_metadata_out` pattern.

---

### 6. Konteks Chunk yang Terpotong

**Tantangan:** Chunk yang diretrive sering kehilangan konteks sekitarnya — misalnya, Pasal 14 Butir 3 merujuk ke definisi di Butir 1-2 yang tidak ikut diretrive.

**Solusi:** Implementasi **Adjacent Chunk Expansion** yang secara otomatis mengambil chunk tetangga (sebelum dan sesudah) dari Supabase menggunakan `(document_id, chunk_index)` sebagai kunci. Query batch dilakukan per dokumen untuk efisiensi, dan hasilnya disusun dengan label "Konteks Sebelumnya" / "Bagian Utama" / "Konteks Berikutnya" agar LLM dapat memahami posisi setiap bagian.

---

# Components and Implementation

Seluruh komponen backend pada proyek ini dirancang, diimplementasikan, dan dikelola secara mandiri, mencakup:

- **Perancangan Arsitektur Sistem:** Merancang keseluruhan arsitektur backend berlapis (_layered architecture_) dengan pemisahan yang jelas antara Handler, Use Case, dan Service layer, serta penetapan dua sistem database (PostgreSQL relasional + Supabase pgvector) sesuai kebutuhan tiap jenis data.

- **Implementasi Advanced RAG Pipeline:** Membangun pipeline RAG 5-tahap dari nol (_scratch_): Query Rewriting → Vector Retrieval → Cross-Encoder Reranking → Adjacent Chunk Expansion → Multi-Model Generation, termasuk seluruh logika _confidence thresholding_ dan _hallucination prevention_.

- **Legal Document Preprocessing Engine:** Membangun `preprocessing_service.py` yang komprehensif untuk menangani variasi format dokumen peraturan desa dari 120+ PDF dengan berbagai layout. Pipeline cleaning 9-tahap mencakup perbaikan OCR spacing, spaced-keyword normalization, `_merge_broken_lines()`, `_normalize_letter_numbering()`, dan metadata extraction berbasis `header_block`. Mendukung **PREVIEW MODE** (`save_to_db=False`) untuk validasi hasil chunking tanpa menyentuh database.

- **Multi-Model Abstraction Layer:** Merancang dan mengimplementasikan `HuggingFaceService` yang mengabstraksi komunikasi ke 3 jenis endpoint LLM (HuggingFace Router, Maia Router/OpenAI-compatible, B200 RAFT Server) di balik antarmuka yang seragam, sehingga penggantian model dapat dilakukan tanpa mengubah logika inti.

- **RAFT Model Integration:** Mengintegrasikan model fine-tuned RAFT (_Retrieval-Augmented Fine-Tuning_) yang dilatih khusus untuk domain peraturan desa, termasuk penanganan format input/output khusus dan ekstraksi _thought process_ analisis dari model.

- **RAGAS Evaluation Framework:** Mengimplementasikan sistem evaluasi otomatis menggunakan RAGAS dengan 5 metrik kualitas RAG, termasuk integrasi ke database untuk persisten penyimpanan hasil evaluasi dan flag per-sesi yang dapat dikontrol frontend.

- **Database Schema Design:** Merancang schema PostgreSQL yang mencakup manajemen user, sesi percakapan, riwayat chat, dan skema evaluasi metrik RAGAS yang efisien, serta SQL function `match_documents` kustom untuk vector similarity search di Supabase.

- **RAFT Dataset Generation:** Merancang dan menjalankan pipeline pembuatan dataset RAFT v4 (`raft_dataset_v4_production.jsonl`) dari dokumen peraturan desa untuk keperluan fine-tuning model. Pipeline mencakup embedding cache, answerability check, 4 jenis distractor (normal, hard*negative, near_miss, completely_absent), quality gate `is_bad_sample()`, dan format sampel dengan \_thought process* + _natural completion_ yang terpisah.

---
