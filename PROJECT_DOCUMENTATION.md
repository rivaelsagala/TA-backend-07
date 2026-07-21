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
> Di dalam Abstrak Laporan TA, disebutkan sistem menggunakan **FAISS** dan **sentence-transformer** dengan jumlah **16 dokumen** (atau 126 di abstrak Inggris). Namun, implementasi kode backend & dokumentasi ini menggunakan **Supabase (pgvector)** dan **OpenAI ext-embedding-3-large**. Anda HARUS merevisi Laporan TA Anda agar sesuai dengan implementasi kode sesungguhnya.

# Technology Stack

| Kategori                    | Teknologi                                    | Fungsi                                                                       |
| --------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| **Web Framework**           | Flask 3.1 + Flask-CORS                       | REST API server dan routing HTTP                                             |
| **Production Server**       | Gunicorn                                     | WSGI server untuk deployment produksi                                        |
| **Vector Database**         | Supabase (pgvector)                          | Menyimpan embedding dokumen dan melakukan similarity search                  |
| **Relational Database**     | PostgreSQL (via Supabase)                    | Menyimpan users, sesi chat, riwayat percakapan, dan metrik evaluasi          |
| **Embedding Model**         | OpenAI `text-embedding-3-large`              | Mengubah teks dokumen/query menjadi vektor numerik                           |
| **LLM — Base Models**       | Llama 3.1 8B, Qwen2.5 7B, DeepSeek-R1 7B     | Model bahasa besar untuk generasi jawaban (via HuggingFace Router)           |
| **LLM — OpenAI-Compatible** | GPT-4o-mini, GPT-3.5-turbo, Gemini 2.0 Flash | Model cloud via Maia Router API                                              |
| **LLM — RAFT Model**        | `model_merged_raft_perdes`                   | Model fine-tuned khusus domain Perdes (di-host di B200 Server)               |
| **Reranker**                | CrossEncoder `cross-encoder/ms-marco-MiniLM-L6-v2` (lokal) | Cross-encoder untuk re-ranking dokumen hasil retrieval (di-load langsung via `sentence-transformers`, bukan API eksternal) |
| **PDF Extraction**          | PyMuPDF (fitz)                               | Ekstraksi teks dari file PDF berbasis teks                                   |
| **OCR**                     | pytesseract + Pillow + OpenCV                | Ekstraksi teks dari PDF hasil scan (gambar)                                  |
| **Orchestration**           | LangChain + LangChain-OpenAI                 | Orkestrasi pipeline RAG dan integrasi vector store                           |
| **Evaluation Framework**    | RAGAS                                        | Evaluasi kualitas sistem RAG secara otomatis (faithfulness, relevancy, dll.) |
| **Reranking Library**       | sentence-transformers (CrossEncoder)          | Re-ranking dokumen hasil retrieval secara lokal                              |
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
│  B200 RAFT Server   │ Reranker Lokal (CrossEncoder)    │
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
              │                  RAFT Fine-Tuning
              │                  (Retrieval-Augmented
              │                   Fine-Tuning)
              │                         │
              │                         ▼
              │                  Fine-Tuned Model
              │                  (model_merged_raft_perdes)
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
 Semantic Similarity, Latency)
              │
              ▼
     Comparative Analysis
```

## A. Alur Ingestion Dokumen (Satu Kali Setup)

### Flowchart: End-to-End Ingestion Pipeline (PDF to Database)

Berikut adalah detail logika alur kerja (workflow) dari API ingestion yang melibatkan `embedding_use_case` dan `preprocessing_service`:

```text
[POST /api/generate-embedding]
  Payload: file (PDF), form: save_to_db (true/false), is_distractor (true/false)
          │
          ▼
handle_generate_embedding()  → simpan ke temp/, panggil ingest_pdf_to_vector_db()
          │
          ▼
ingest_pdf_to_vector_db(file_path, original_filename, save_to_db, is_distractor)
          │
          ▼
extract_and_chunk_pdf(file_path, save_to_db, is_distractor) ──────────┐
          │                                                       │
          ├─► extract_text_from_pdf()                             │
          │   (PyMuPDF / Fallback OCR pytesseract jika < 50 char)  │
          │                                                       │
          ├─► chunk_documents(documents)                          │
          │   (clean_legal_text + _parse_perdes_sections)         │
          │   ↳ Metadata Injection OTOMATIS di sini:              │
          │     inject source_file, document_id, village_name,     │
          │     perdes_number, perdes_year (dari extract_perdes_  │
          │     metadata); prefix "perdes_dis" jika is_distractor │
          │                                                       │
          ├─► save_results_to_folder()                            │
          │   (Simpan file lokal .txt & .json ke data/processed/ │
          │    raw/ atau distraktor/ untuk review)                 │
          │                                                       │
          └───────────────────────────────────────────────────────┘
          │
          ▼
   Jika save_to_db == True:
          │
          ▼
   save_chunks_to_postgres(chunks)
   (Simpan teks asli ke tabel chunks_perdes via psycopg2)
          │
          ▼
   check_document_exists(document_id)  →  jika > 0:
   delete_document_chunks(document_id)  (hapus vektor lama)
          │
          ▼
   store_chunks_to_supabase(chunks)
   (Kirim ke OpenAI text-embedding-3-large, simpan ke
    tabel `documents` di Supabase pgvector)
          │
          ▼
   Return "SUCCESS"  (atau "PREVIEW MODE" jika save_to_db=False)
```

> ⚠️ **Catatan alur sebenarnya:** `save_to_db` mengontrol **dua** penyimpanan sekaligus — baik ke PostgreSQL (`chunks_perdes`) maupun ke Supabase pgvector (`documents`). Jika `save_to_db=False`, keduanya dilewati dan sistem hanya menyimpan hasil ekstraksi (.txt/.json) ke folder lokal sebagai **PREVIEW MODE**. Parameter `is_distractor` (default `false`) mengubah `document_id` menjadi prefix `perdes_dis` untuk dataset distraktor.

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

### Penjelasan Rinci Modul `preprocessing_service.py`

Modul `preprocessing_service.py` bertanggung jawab atas keseluruhan proses persiapan data dokumen hukum (Peraturan Desa) sebelum di-embedding dan disimpan ke dalam database. Modul ini dirancang secara khusus untuk menangani struktur format peraturan perundang-undangan di Indonesia, dan terdiri dari beberapa fungsi krusial:

1. **`clean_legal_text(text: str) -> str`**
   Fungsi ini membersihkan teks mentah hasil ekstraksi PDF melalui 9 tahapan pembersihan khusus dokumen hukum:
   - **Tahap 1 & 1.5:** Normalisasi karakter unicode, perbaikan spasi yang salah akibat OCR (menggabungkan huruf yang terpisah kecuali kata depan valid), dan perbaikan typo spesifik _preamble_ ("menginat" menjadi "mengingat").
   - **Tahap 2:** Konversi teks menjadi huruf kecil (_lowercase_).
   - **Tahap 3 & 4:** Memperbaiki kata kunci hukum yang terpisah spasi (seperti "P A S A L") menjadi satu kata yang utuh, serta menghapus karakter-karakter _noise_ non-alfanumerik yang dihasilkan oleh kesalahan OCR.
   - **Tahap 5:** Normalisasi whitespace dan penggabungan baris yang terputus (_broken lines_). Fungsi akan menganalisis baris-baris yang tidak diakhiri tanda baca dan menggabungkannya jika bukan awal dari bagian struktural baru.
   - **Tahap 6, 7 & 8:** Penghapusan artefak PDF, header/footer berulang, dan blok tanda tangan (seperti "ditetapkan di...", nama desa, tulisan salinan).
   - **Tahap 9 & Akhir:** Menghilangkan kata hubung tunggal di awal baris dan membuang teks lampiran yang berlebihan setelah klausul penutup resmi.

2. **`extract_perdes_metadata(file_path, full_text, is_distractor) -> dict`**
   Mengekstrak metadata penting dari teks lengkap Peraturan Desa menggunakan berbagai pola _Regular Expression_ (Regex). Metadata yang diekstrak meliputi: nama desa, nama kabupaten/kota, nomor peraturan, tahun, dan judul peraturan. Ekstraksi difokuskan pada bagian _header_ (25 baris pertama). Jika flag `is_distractor` bernilai `True`, fungsi akan menambahkan prefix `perdes_dis` pada ID dokumen, yang menandakannya sebagai dokumen _distractor_ untuk keperluan pembuatan dataset RAFT.

3. **`extract_text_from_pdf(file_path, is_distractor) -> list`**
   Fungsi ini membaca file PDF halaman demi halaman menggunakan library `fitz` (PyMuPDF). Jika teks yang diekstrak dari suatu halaman terlalu sedikit (kurang dari 50 karakter), fungsi ini mengasumsikan halaman tersebut adalah hasil scan gambar dan akan **otomatis melakukan _fallback_ menggunakan OCR** (`pytesseract`) untuk mengekstrak teks dari gambar halaman tersebut. Teks yang didapat kemudian digabungkan, dibersihkan dengan `clean_legal_text`, dan diperkaya dengan string _context header_ di awal teks.

4. **`_parse_perdes_sections(text: str) -> list`**
   Ini adalah inti dari proses **_Semantic Legal-Aware Chunking_**. Fungsi ini mengurai teks hukum yang sangat panjang menjadi potongan-potongan terstruktur logis berdasarkan hierarki: BAB, Bagian, Pasal, Ayat, dan Butir.
   - Fungsi mengidentifikasi awal BAB, Bagian, dan Pasal menggunakan Regex.
   - Untuk Pasal yang memiliki pembagian ayat, fungsi memanggil `_split_by_ayat`.
   - Untuk Pasal yang berisi daftar/butir numerik tanpa ayat eksplisit, memanggil `_split_butir`.
   - Huruf penomoran butir yang memakai alfabet (a., b., c.) juga dinormalisasi menjadi angka berurutan (1., 2., 3.) melalui `_normalize_letter_numbering` untuk konsistensi pembacaan oleh LLM.

5. **`chunk_documents(documents: list) -> list`**
   Fungsi ini mengambil dokumen teks lengkap (bersama dengan _context header_ di atasnya) dan membaginya menjadi unit-unit atomik (_chunk_) dengan memanggil fungsi `_parse_perdes_sections`. Tiap potongan teks kemudian dimasukkan ke dalam objek `Document` (dari ekosistem LangChain), dengan menyematkan berbagai atribut `metadata` (Bab, Pasal, Ayat, dll) yang nantinya akan disimpan di dalam Vector Database.

6. **`save_results_to_folder(...)` dan `save_chunks_to_postgres(...)`**
   - `save_results_to_folder`: Menyimpan hasil ekstraksi bersih (`_extracted.txt`) dan detail chunking (`_chunks.json`) ke penyimpanan lokal di direktori `data/processed/raw` atau `data/processed/distraktor`. Sangat berguna untuk keperluan _debugging_ atau verifikasi manual dari hasil ekstraksi.
   - `save_chunks_to_postgres`: Menyimpan representasi teks _chunk_ beserta metadata JSON-nya ke database PostgreSQL pada tabel `chunks_perdes`. Berfungsi sebagai _backup_ data non-vektor atau untuk keperluan _full-text search_ tradisional.

7. **`extract_and_chunk_pdf(...)`**
   Fungsi _orchestrator_ utama yang dipanggil oleh API Layer. Fungsi ini menggabungkan seluruh alur yang dijabarkan di atas: mulai dari ekstraksi teks PDF (`extract_text_from_pdf`), pemotongan teks secara semantik (`chunk_documents`), hingga pemanggilan fungsi untuk menyimpan ke sistem file lokal maupun database relasional.

## B. Alur Chat / Tanya Jawab (Real-time)

```
1. [Request]      User mengirim pertanyaan via POST /api/chat
       │          Payload: {session_id, user_id, question, model_id, evaluate}
       │
2. [Filter]       chat_use_case.py melakukan pre-filter:
       │          ↳ is_chitchat() → jawab salam langsung (rule-based)
       │          ↳ is_off_topic() → tolak pertanyaan di luar domain hukum
       │
3. [History]      Ambil riwayat percakapan terakhir (maks. 6 messages)
       │          dari PostgreSQL untuk konteks follow-up
       │          (MAX_HISTORY_MESSAGES = 6 di chat_use_case.py)
       │          ⚠️ Jika evaluate=True → history DI-BYPASS (kosong)
       │            agar evaluasi tidak terkontaminasi riwayat
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
       │          ↳ Cross-encoder LOKAL cross-encoder/ms-marco-MiniLM-L6-v2
       │            (sentence-transformers, di-load langsung — bukan API eksternal)
       │            menilai ulang relevansi 20 kandidat → pilih top 5
       │          ↳ Confidence threshold check: jika skor < -5.0, jawab
       │            "informasi tidak ditemukan" (tanpa hallucination)
       │
7. [Expand]       Adjacent Chunk Expansion — DINONAKTIFKAN SEMENTARA
       │          ↳ Di rag_service.py, adjacent_map = {} (hardcoded kosong)
       │          ↳ Fitur fetch_adjacent_chunks() tidak dijalankan
       │            (perubahan sementara demi eksperimen/perbandingan)
       │
8. [Generate]     HuggingFaceService.chat_with_context()
       │          ↳ Bangun prompt: system prompt + context + history + question
       │          ↳ Kirim ke model yang dipilih (LLM / RAFT)
       │          ↳ RAFT: kirim raw chunks langsung ke B200 server
       │
9. [Evaluate]     (Opsional, jika evaluate=True pada sesi)
       │          ↳ RAGAS menghitung: faithfulness, answer_relevancy,
       │            context_precision, context_recall
       │          ↳ SemanticAnswerSimilarity (SAS) menghitung cosine similarity
       │            antara answer dan ground_truth menggunakan embedding model
       │            (hanya jika ground_truth diberikan)
       │          ↳ Semua metrik dikembalikan dalam satu dict evaluation_result
       │
10. [Save]        Simpan ke PostgreSQL: pertanyaan, jawaban, sources,
       │          metadata, skor RAGAS + semantic_similarity
       │          ⚠️ CATATAN: di chat_use_case.py, pemanggilan
       │          save_chat_message() sedang DIMATIKAN SEMENTARA
       │          ([SKIP] mode evaluasi batch) — riwayat tidak tersimpan.
       │
11. [Response]    Kembalikan JSON ke frontend:
                  {answer, sources, evaluation, model_used}
                  evaluation: {faithfulness, answer_relevancy,
                               context_precision, context_recall,
                               semantic_similarity}
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
       │  │  URL: {FINETUNED_API_URL}/chat-raft                    │
       │  │  Payload:                                              │
       │  │  {                                                     │
       │  │    "question": "<user_question>",                      │
       │  │    "documents": ["<chunk1>", "<chunk2>", ..., "<chunk5>"]│
       │  │  }                                                     │
       │  └────────────────────────────────────────────────────────┘
       │
       │  ┌────────────────────────────────────────────────────────┐
       │  │         Response dari B200 RAFT Server                 │
       │  │  {                                                     │
       │  │    "answer": "<final_answer>",                          │
       │  │    "thought": "<thought_process / chain-of-thought>",   │
       │  │    "konteks_dipilih": "<...>",                          │
       │  │    "konteks_ditolak": "<...>",                          │
       │  │    "documents_count": 5,                                │
       │  │    "question": "<user_question>",                      │
       │  │    "status": "success"                                 │
       │  │  }                                                     │
       │  └────────────────────────────────────────────────────────┘
       │
       │  ↳ Response di-standardisasi ke format OpenAI-compatible:
       │    {"choices": [{"message": {"content": answer}}],
       │     "raft_metadata": {"analisis": thought, "konteks_dipilih": ...,
       │                      "konteks_ditolak": ..., "num_documents": ...}}
       │
10. [Evaluate]    (Opsional, jika evaluate=True pada sesi)
       │          ↳ RAGAS + SAS menghitung metrik sama seperti Alur B
       │            (faithfulness, answer_relevancy, context_precision,
       │             context_recall, + semantic_similarity jika ada ground_truth)
       │          ↳ contexts diambil dari raw_doc_chunks (bukan expanded)
       │
11. [Save]        Simpan ke PostgreSQL dengan field tambahan:
       │          ↳ metadata["analysis"] = raft_metadata["analisis"]
       │            (thought process RAFT disimpan di kolom metadata JSONB)
       │          ⚠️ CATATAN: save_chat_message() sedang DIMATIKAN
       │          SEMENTARA di chat_use_case.py ([SKIP] mode evaluasi batch)
       │
12. [Response]    Kembalikan JSON ke frontend dengan field tambahan:
                  {
                    answer, sources, evaluation, model_used,
                    "analysis": "<thought_process dari RAFT>"  ← KHUSUS RAFT
                    "konteks_dipilih": "<...>",
                    "konteks_ditolak": "<...>",
                    evaluation: {faithfulness, answer_relevancy,
                                 context_precision, context_recall,
                                 semantic_similarity}
                  }
```

### Perbandingan Alur B (Model Standar) vs Alur C (RAFT)

| Tahapan                    | Model Standar (Alur B)                     | RAFT Model (Alur C)                        |
| -------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Adjacent Expansion**     | ⚠️ DINONAKTIFKAN SEMENTARA (`adjacent_map = {}`) | ❌ Dilewati                                |
| **System Prompt**          | ✅ Dibangun (instruksi + konteks gabungan) | ❌ Tidak ada                               |
| **Format Input ke LLM**    | Joined context string dalam system prompt  | List raw chunks terpisah (`"documents"`)   |
| **Endpoint LLM**           | HuggingFace Router / Maia Router           | B200 RAFT Server (`/chat-raft`)            |
| **Field Respons Tambahan** | —                                          | `"analysis"`, `"konteks_dipilih"`, `"konteks_ditolak"` (RAFT) |
| **Chat History Konteks**   | ✅ Dikirim ke LLM (maks. 6 pesan)         | ❌ Tidak dikirim (RAFT stateless per-call) |

### Penjelasan Rinci Modul `rag_service.py`

Modul `rag_service.py` adalah jantung dari sistem tanya jawab dokumen hukum (RAG) ini. Modul ini bertugas mengorkestrasi seluruh tahapan mulai dari penerimaan kueri pengguna hingga pengembalian jawaban akhir beserta sumber referensinya. Berikut fungsi-fungsi krusial yang ada di dalamnya:

1. **Inisialisasi Database & Embeddings**
   Di bagian paling atas, modul menyiapkan klien Supabase (untuk berkomunikasi dengan _vector store_) dan model _embeddings_ `text-embedding-3-large` dari OpenAI via LangChain. Selain itu, terdapat kamus `AVAILABLE_MODELS` yang memetakan daftar semua LLM yang bisa digunakan, mencakup model Original (HuggingFace), OpenAI/OpenRouter, Google Gemini, hingga model RAFT *fine-tuned*.

2. **`HuggingFaceService` & API Routing**
   Kelas ini (beserta _instance_ global `hf_service`) bertindak sebagai _adapter_ universal yang menjembatani sistem RAG dengan berbagai endpoint API LLM:
   - **`query()`**: Fungsi utama yang menyalurkan permintaan ke URL API yang sesuai berdasarkan tipe model (`"original"`, `"openai"`, atau `"raft"`). Khusus untuk model `"raft"`, payload disesuaikan untuk mengirim array `documents` secara langsung (tidak digabung menjadi string konteks tunggal) serta mengambil metadata evaluatif kembalian seperti _thought process_ (`analisis`), `konteks_dipilih`, dll.
   - **`chat_with_context()`**: Mengkonstruksi _system prompt_ (yang saat ini dikonfigurasi rentan halusinasi secara disengaja untuk pengujian robustnes sistem RAG), menggabungkan _chat history_, dan memanggil fungsi `get_completion`. Khusus RAFT, konstruksi _system prompt_ ini sepenuhnya dilewati.

3. **`rewrite_query_with_history(original_query, chat_history)`**
   Sebuah fungsi pencegat (_interceptor_) untuk Tahap 0 (sebelum pencarian vektor). Fungsi ini membaca _chat history_ (maksimal 6 pesan sebelumnya) dan menggunakan model LLM kecil dan cepat (`gpt-3.5-turbo`) untuk merumuskan ulang pertanyaan lanjutan yang ambigu (misal: "lalu bagaimana hukum untuk hal itu?") menjadi kalimat pertanyaan utuh (_standalone query_) yang optimal digunakan untuk pencarian kemiripan di database vektor.

4. **`_build_expanded_context_block(doc, adjacent_map)`**
   Fungsi pembantu yang bertugas menenun informasi struktural dengan blok teks tetangganya (_adjacent chunk_). Fungsi ini menambahkan keterangan metadata secara rapi (Nama Desa, Nomor/Tahun, dll) dan menyematkan bagian teks penanda `[Konteks Sebelumnya]` dan `[Konteks Berikutnya]` agar LLM memperoleh pandangan utuh hierarki dokumen.

5. **`evaluate_rag_answer(query, context, answer)` (LLM-as-a-Judge)**
   Fungsi evaluasi _inline_ otomatis. Fungsi ini memanfaatkan agen `gpt-4o-mini` (dengan nilai `temperature=0.0`) untuk menilai kualitas jawaban secara seketika. Agen juri ini diminta memverifikasi apakah jawaban "Grounded" (benar bersumber dari teks) dan "Relevant", lalu mengembalikan skor serta alasannya dalam wujud JSON.

6. **`_extract_final_answer(text: str)`**
   Mekanisme keamanan ekstraksi teks (fallback) yang memotong proses _Chain-of-Thought_ panjang dari _output_ mentah LLM, dengan cara mencari penanda teks seperti `[JAWABAN AKHIR]` dan membuang teks internal-evaluasi sebelumnya, memastikan _user_ hanya melihat jawaban akhirnya saja.

7. **`get_answer_from_rag(query, model_id, chat_history)` (Main Orchestrator)**
   Fungsi inti (_entry point_) sistem pipeline RAG yang mengeksekusi seluruh alur dari hulu ke hilir:
   - **Tahap 0**: Rewrite kueri (`rewrite_query_with_history`).
   - **Tahap 1**: Pencarian _top-20_ dokumen awal secara cepat dengan Vector Search di Supabase.
   - **Tahap 2**: Re-ranking _top-20_ menjadi _top-5_ yang sangat presisi menggunakan model Cross-Encoder MS Marco. Tahap ini juga mengevaluasi skor keyakinan (_Confidence Filter_ < -5.0) untuk _early exit_ jika dokumen sama sekali tidak nyambung dengan pertanyaan (pencegahan _hallucination_ mutlak).
   - **Tahap 3**: _Adjacent Chunk Expansion_ (untuk memperluas konteks blok dokumen, walau kode saat ini sedang dinonaktifkan keras untuk uji komparasi).
   - **Tahap 4 & 5**: Formatisasi data referensi (sumber hukum) dan pemanggilan model LLM. Hasil akhir (teks jawaban, _latency_, detail sumber hukum, dan hasil _judge_) disatukan dalam sebuah objek struktur JSON yang siap dikirim ke frontend via API.

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
     │          Model: openai/gpt-4o-mini (temperature=0.0, max 2000)
     │          Role : juri absolut, tidak berubah antar evaluasi
     │
     │  ┌─────────────────────────────────────────────────────────────┐
     │  │ METRIK YANG DIHITUNG (4 metrik RAGAS):                    │
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
     │  │   ⚠️ Hanya dihitung jika ground_truth diberikan           │
     │  │                                                             │
     │  │ ④ context_recall (0–1)                                     │
     │  │   Input : contexts + ground_truth                          │
     │  │   Ukur  : seberapa banyak info ground_truth ada di konteks?│
     │  │   Tinggi = konteks lengkap mencakup referensi pakar        │
     │  │   ⚠️ Akurat hanya jika ground_truth adalah jawaban pakar   │
     │  │   ⚠️ Hanya dihitung jika ground_truth diberikan           │
     │  │                                                             │
     │  │ ⚠️ CATATAN: noise_sensitivity TIDAK digunakan lagi        │
     │  │   (tidak ada di self.metrics pada ragas_service.py)        │
     │  └─────────────────────────────────────────────────────────────┘
     │
3. [Result]    df_results = evaluation_result.to_pandas()
     │          formatted_result = {faithfulness, answer_relevancy,
     │                              context_precision, context_recall}
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
     │          → Satu dict evaluasi lengkap (4 metrik RAGAS + SAS)
     │
─────┴──────────────────────────────────────────────────────────────────
 PERSISTENSI — Simpan ke Database
─────────────────────────────────────────────────────────────────────────
     │
7. [Save]      chat_service.save_chat_message(evaluation=evaluation_result)
     │          ⚠️ CATATAN: di chat_use_case.py pemanggilan ini sedang
     │          DIMATIKAN SEMENTARA ([SKIP] mode evaluasi batch) —
     │          riwayat tidak tersimpan ke PostgreSQL saat ini.
     │
     │  Mapping key Python → kolom PostgreSQL:
     │  ┌──────────────────────┬────────────────────────────────────┐
     │  │ evaluation dict key  │ Kolom chat_history (FLOAT)         │
     │  ├──────────────────────┼────────────────────────────────────┤
     │  │ "faithfulness"       │ faithfulness                       │
     │  │ "answer_relevancy"   │ answer_relevance                   │
     │  │ "context_precision"  │ context_precision                  │
     │  │ "context_recall"     │ context_recall                     │
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
| **semantic_similarity** | 0–1     | answer, ground_truth                     | ✅ Wajib                      | Cosine similarity embedding: kesamaan makna jawaban vs pakar |

> ⚠️ **`noise_sensitivity` TIDAK lagi digunakan** — tidak ada di `self.metrics` pada `ragas_service.py` (sekarang hanya 4 metrik: faithfulness, answer_relevancy, context_precision, context_recall).

### Penanganan Kasus ground_truth Tidak Diberikan

```
Jika ground_truth = None pada request payload:
  → ragas_service memberi WARNING di log
  → ground_truth di-fallback ke answer (jawaban LLM itu sendiri)
  → Konsekuensi:
      • context_precision → TIDAK dihitung (butuh ground_truth)
      • context_recall   → TIDAK VALID (mengukur konteks vs jawaban LLM, bukan pakar)
      • semantic_similarity → SELALU 1.0 (answer == ground_truth)
  → Metrik faithfulness & answer_relevancy TETAP VALID
```

### Konfigurasi LLM Judge & Embedding (ragas_service.py)

| Komponen               | Model                                                | Parameter Kunci             | Tujuan                                |
| ---------------------- | ---------------------------------------------------- | --------------------------- | ------------------------------------- |
| **LLM Judge (RAGAS)**  | `openai/gpt-4o-mini`                                 | `temperature=0.0`, max 2000 | Juri stabil & deterministik           |
| **Embeddings (RAGAS)** | `openai/text-embedding-3-large`                      | 1536 dimensi                | Representasi semantik untuk relevancy |
| **SAS Embedder**       | `openai/text-embedding-3-large`                      | Sama dengan RAGAS embedder  | Cosine similarity answer vs truth     |
| **Wrapper**            | `LangchainLLMWrapper` + `LangchainEmbeddingsWrapper` | —                           | Standardisasi format JSON RAGAS       |

---

## D. Alur Pembuatan Dataset RAFT (`notebooks/generate_raft_dataset.py`)

> **V2 — Retrieval-Grounded Generator:** Berbeda dengan pendekatan lama yang mengambil chunk dari file lokal, V2 ini **mengambil dokumen langsung dari Vector DB (Supabase)** menggunakan pipeline retrieval yang **sama persis dengan inference** (Top-20 → Re-ranking → Top-5). Tujuannya agar distribusi data saat training sama dengan distribusi saat inference (RAFT principle). Generator LLM: `openai/gpt-4o-mini` (`temperature=0.3`, `max_tokens=2048`).

```
[Input]  Daftar pertanyaan (dari raft_dataset_finalv1.jsonl
        via load_questions_from_jsonl, atau QUESTIONS default)
    │
1. [Resume]       Jika output sudah ada → skip pertanyaan yang sama
    │              (mode resume=True, hindari regenerate)
    │
2. [Retrieve]      retrieve_top5_documents(question)
    │              ↳ SupabaseVectorStore.similarity_search(k=INITIAL_K=10)
    │                + filter EXCLUDED_DOCUMENT_IDS (dokumen "gold" dibuang
    │                agar tidak jadi distractor yang terlalu mudah)
    │              ↳ rerank_documents() → Top-5 (cross-encoder lokal)
    │              ↳ Confidence filter: jika top_score < -5.0 → skip pertanyaan
    │              ↳ Hasil: 5 chunk dokumen mentah (raw content)
    │
3. [Generate]     generate_single_raft_entry(question, documents)
    │              ↳ System prompt RAFT_SYSTEM_PROMPT (retrieval-grounded)
    │              ↳ call_llm() → GPT-4o-mini (temp 0.3, retry 3x)
    │              ↳ parse_raft_json() + validate_raft_structure()
    │                (wajib: instruction, thought_process, completion)
    │              ↳ thought_process HARUS berupa dict:
    │                  {"document_analysis": [...], "summary": "..."}
    │              ↳ Jika invalid → retry (maks 3 attempt)
    │
4. [Save]         Simpan ke data/dataset/
                   raft_dataset_retrieval_grounded_YYYYMMDD_HHMMSS.jsonl
                   ↳ Format JSONL | Append per sampel (resume-safe)
                   ↳ Counter: Berhasil | Gagal
```

> ⚠️ **Catatan penting — struktur `thought_process` berubah:** Di V2, `thought_process` **bukan lagi string bebas**, melainkan **objek terstruktur** `{"document_analysis": [{"document": n, "analysis": "..."}], "summary": "..."}`. Setiap dokumen hasil retrieval wajib dianalisis satu per satu, lalu disintesis di `summary`. Aturan ketat diberlakukan agar tidak ada context-mixing: jika pertanyaan menyebut NOMOR PERATURAN spesifik, dokumen dengan nomor BERBEDA **wajib diabaikan** (tidak boleh masuk completion).

### Format Sampel Dataset RAFT v2 (Retrieval-Grounded)

```json
{
  "instruction": "Apa fungsi BPD menurut peraturan desa ini?",
  "documents": [
    "pasal 7\nkewenangan lokal...",
    "pasal 1\nbpd adalah lembaga..."
  ],
  "thought_process": {
    "document_analysis": [
      {"document": 1, "analysis": "Pasal 1 relevan: mendefinisikan BPD sebagai badan permusyawaratan desa."},
      {"document": 2, "analysis": "Pasal 7 membahas kewenangan, tidak relevan dengan pertanyaan fungsi."}
    ],
    "summary": "Berdasarkan Pasal 1, BPD adalah lembaga permusyawaratan desa."
  },
  "completion": "BPD berfungsi sebagai lembaga pemerintahan desa yang anggotanya merupakan wakil penduduk berdasarkan keterwakilan wilayah."
}
```

> Completion **harus natural** (hanya informasi dari Documents, tanpa pengetahuan internal model, tanpa halusinasi). Analisis tiap dokumen ada di `thought_process.document_analysis`.

### Generator Negatif Terpisah (`notebooks/negative_generate_raft_datase.py`)

Untuk membangun set distractor yang kuat, terdapat notebook **terpisah** yang mengambil dokumen dari `EXCLUDED_DOCUMENT_IDS` sebagai sumber negative sampling (INITIAL_K=40, FINAL_K=5, CONFIDENCE_THRESHOLD=-9999.0 agar tidak difilter). Hasilnya digabung dengan dataset utama untuk melatih model membedakan dokumen yang hampir identik.

### Keputusan Desain Penting (V2)

| Keputusan                                          | Alasan                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Retrieval-Grounded**                             | Dokumen diambil dari Vector DB dengan pipeline sama seperti inference, sehingga distribusi train ≈ inference    |
| **Exclude "gold" docs saat retrieve**                | Dokumen jawaban dibuang dari kandidat agar tidak jadi distractor terlalu mudah                              |
| **`thought_process` terstruktur**                  | `document_analysis` + `summary` memaksa LLM menganalisis tiap dokumen eksplisit sebelum menyintesis jawaban           |
| **Larangan context-mixing**                         | Jika pertanyaan sebut nomor peraturan spesifik, dokumen nomor lain wajib diabaikan (hindari kontaminasi)          |
| **Resume mode**                                     | Skip pertanyaan yang sudah ada di output → aman dijalankan berulang kali tanpa duplikasi                       |
| **Validasi JSON ketat**                            | `validate_raft_structure()` menolak output tanpa key wajib / `thought_process` bukan dict → retry otomatis              |

# Key Features

### 1. 🔍 Multi-Stage RAG Pipeline

Pipeline RAG yang berlapis dan sistematis: **Retrieval (k=20) → Re-ranking (k=5) → Generation**.

- **Tahap 1 (Retrieval):** Mengambil 20 kandidat dokumen (chunk) teratas yang secara semantik mirip dengan pertanyaan.
- **Tahap 2 (Re-ranking):** Cross-Encoder **lokal** `cross-encoder/ms-marco-MiniLM-L6-v2` (via `sentence-transformers`, di-load langsung — bukan API eksternal) menilai ulang 20 kandidat tersebut secara ketat, lalu membuang 15 terbawah dan hanya menyisakan **5 pemenang (Top 5)**.
- **Tahap 3 (Expansion):** ⚠️ **DINONAKTIFKAN SEMENTARA** — di `rag_service.py`, `adjacent_map = {}` di-hardcode kosong sehingga `fetch_adjacent_chunks()` tidak dijalankan (perubahan sementara untuk eksperimen/perbandingan).

### 2. 📝 Legal Document-Aware Chunking (Semantic & Recursive)

Chunking tidak dilakukan dengan ukuran karakter biasa (fixed-size), melainkan dengan **memahami struktur hukum dokumen** (BAB → Pasal → Ayat → Butir). Di dalam implementasi (meskipun di laporan TA disebutkan RecursiveCharacterTextSplitter), sistem menggunakan pendekatan hibrida _Semantic Legal-Aware Chunking_ (\_parse_perdes_sections). Setiap chunk merupakan satu unit logis peraturan yang atomic, sehingga retrieval lebih presisi dan jawaban dapat menyebut pasal secara spesifik.

### 3. 🔄 Query Rewriting untuk Follow-up Questions

Sistem mendeteksi dan menyelesaikan pertanyaan lanjutan yang ambigu secara otomatis menggunakan LLM ringan (GPT-3.5-turbo). Query "apa isi pasal itu?" akan diubah menjadi query yang mandiri dan spesifik sebelum masuk ke pipeline retrieval.

### 4. 🤖 Multi-Model Support

Sistem mendukung **7 model LLM yang dapat dipilih** (`AVAILABLE_MODELS` di `rag_service.py`), mencakup model open-source (Llama 3.1 8B, Qwen2.5 7B, DeepSeek-R1 7B), model cloud (GPT-4o-mini, GPT-3.5-turbo, Gemini 2.0 Flash), dan model RAFT fine-tuned khusus domain peraturan desa (`model_merged_raft_perdes`). Frontend dapat memilih model per sesi.

### 5. 🧠 RAFT Model Integration

Terintegrasi dengan model fine-tuned berbasis **RAFT (Retrieval-Augmented Fine-Tuning)** yang di-host pada server GPU dedicated (B200). Model ini dilatih khusus untuk menjawab pertanyaan hukum peraturan desa, mampu melakukan reasoning internal terhadap dokumen, dan menghasilkan analisis berpikir (_thought process_) yang dapat ditampilkan ke pengguna.

### 6. 📊 Automated RAGAS + Semantic Similarity Evaluation

Evaluasi kualitas RAG dilakukan secara otomatis menggunakan dua komponen yang berjalan berurutan:

- **RAGAS Framework** — menghitung **4 metrik** berbasis LLM Judge (`openai/gpt-4o-mini`, `temperature=0.0`): Faithfulness, Answer Relevancy, Context Precision, dan Context Recall. (Metrik `noise_sensitivity` sudah tidak digunakan lagi.)
- **Semantic Answer Similarity (SAS)** — menghitung cosine similarity antara embedding `answer` dan `ground_truth` menggunakan `text-embedding-3-large`, menghasilkan skor kesamaan makna (0–1) yang tidak bergantung pada format atau tanda baca. Hanya dihitung jika `ground_truth` diberikan.

Evaluasi berjalan per sesi (dikontrol flag `evaluate` di `chat_sessions`). ⚠️ **Catatan:** pemanggilan `save_chat_message()` sedang **dimatikan sementara** di `chat_use_case.py` (`[SKIP] mode evaluasi batch`), sehingga riwayat belum tersimpan ke PostgreSQL saat ini.

> 🔕 **Inline LLM-as-a-Judge dinonaktifkan:** Fungsi `evaluate_rag_answer()` di `rag_service.py` (yang memanggil `gpt-4o-mini` untuk menilai jawaban secara inline) telah **di-comment** agar endpoint mengembalikan **respons murni dari model** tanpa penilaian juri. Field `judge_evaluation` ikut di-comment dari response JSON.

### 7. 🔒 Topic Guard & Confidence Threshold

Sistem memiliki dua lapisan filter: **Off-Topic Filter** (berbasis keyword) menolak pertanyaan di luar domain hukum desa, dan **Confidence Threshold** (skor reranker < -5.0) mencegah model mengarang jawaban (_hallucination_) saat tidak ada dokumen yang relevan.

### 8. 📂 Smart Adjacent Chunk Expansion

Sistem mengatasi masalah teks terpotong (akibat proses _chunking_ di awal) dengan fitur **Ekspansi Chunk Bertetangga**. Setelah proses _Re-ranking_ menghasilkan **5 chunk pemenang (Top 5)**, sistem secara otomatis kembali ke database untuk mengambil 1 chunk tepat **sebelum** dan 1 chunk tepat **sesudah** dari masing-masing pemenang tersebut.

> ⚠️ **Status saat ini: DINONAKTIFKAN SEMENTARA.** Di `rag_service.py`, baris `adjacent_map = {}` di-hardcode sehingga `fetch_adjacent_chunks()` tidak dijalankan. Fitur ini masih tersedia sebagai kode (lihat blok di bawah) namun tidak aktif dalam pipeline saat ini — dimatikan untuk keperluan eksperimen/perbandingan.

Hasilnya (jika diaktifkan kembali), LLM tidak hanya membaca 1 potongan kecil teks, melainkan membaca **5 blok teks yang utuh**. Karena masing-masing dari 5 pemenang ini diekspansi menjadi 3 bagian, total teks yang disuapkan ke LLM setara dengan 15 chunk dokumen (5 blok × 3 chunk).

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
| **`chat_history`**  | `id`, `session_id`, `user_id`, `user_query`, `llm_response`, `metadata` (JSONB), `is_evaluated`, `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`, `semantic_similarity` | Menyimpan setiap pasang tanya-jawab beserta metrik evaluasi RAGAS (4 metrik) + SAS dalam kolom bertipe FLOAT yang terstruktur. Kolom evaluasi bernilai NULL jika sesi tidak mengaktifkan evaluasi. |
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
| **B200 RAFT Server**       | HTTP POST (`/chat-raft`)       | Inference model RAFT fine-tuned lokal di server GPU dedicated  |
| **Reranker Lokal**        | In-process (`sentence-transformers`) | Cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` untuk re-ranking dokumen (tanpa API eksternal) |

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

**Solusi:** Implementasi **Confidence Threshold** berbasis skor cross-encoder lokal (`cross-encoder/ms-marco-MiniLM-L6-v2`). Jika skor relevansi tertinggi dari hasil re-ranking berada di bawah threshold (-5.0), sistem langsung mengembalikan respons "tidak ditemukan" tanpa meneruskan ke LLM, sehingga hallucination dapat dicegah sepenuhnya.

---

### 5. Integrasi RAFT Model dengan Format Response yang Berbeda

**Tantangan:** Model RAFT fine-tuned memiliki format input/output yang berbeda dari model standar (tidak menggunakan system prompt, menerima daftar dokumen mentah, menghasilkan `answer` + `thought` bukan `choices[0].message.content`).

**Solusi:** Implementasi `HuggingFaceService` dengan **abstraksi multi-model** yang mendeteksi tipe model (`raft`, `openai`, `original`) dan menerapkan routing logic yang sesuai. Endpoint RAFT adalah `{FINETUNED_API_URL}/chat-raft` dengan payload `{"question", "documents"}`. Response RAFT di-standardisasi ke format OpenAI-compatible sebelum dikembalikan, dan metadata analisis (`analisis`, `konteks_dipilih`, `konteks_ditolak`) disimpan secara terpisah melalui `_raft_metadata_out` pattern.

---

### 6. Konteks Chunk yang Terpotong

**Tantangan:** Chunk yang diretrive sering kehilangan konteks sekitarnya — misalnya, Pasal 14 Butir 3 merujuk ke definisi di Butir 1-2 yang tidak ikut diretrive.

**Solusi:** Implementasi **Adjacent Chunk Expansion** yang secara otomatis mengambil chunk tetangga (sebelum dan sesudah) dari Supabase menggunakan `(document_id, chunk_index)` sebagai kunci. Query batch dilakukan per dokumen untuk efisiensi, dan hasilnya disusun dengan label "Konteks Sebelumnya" / "Bagian Utama" / "Konteks Berikutnya" agar LLM dapat memahami posisi setiap bagian.

> ⚠️ **Status saat ini:** Fitur ini **dinonaktifkan sementara** — di `rag_service.py`, `adjacent_map = {}` di-hardcode sehingga `fetch_adjacent_chunks()` tidak dijalankan. Dimatikan untuk keperluan eksperimen/perbandingan.

---

# Components and Implementation

Seluruh komponen backend pada proyek ini dirancang, diimplementasikan, dan dikelola secara mandiri, mencakup:

- **Perancangan Arsitektur Sistem:** Merancang keseluruhan arsitektur backend berlapis (_layered architecture_) dengan pemisahan yang jelas antara Handler, Use Case, dan Service layer, serta penetapan dua sistem database (PostgreSQL relasional + Supabase pgvector) sesuai kebutuhan tiap jenis data.

- **Implementasi Advanced RAG Pipeline:** Membangun pipeline RAG dari nol (_scratch_): Query Rewriting → Vector Retrieval → Cross-Encoder Reranking (lokal `ms-marco-MiniLM-L6-v2`) → Adjacent Chunk Expansion (saat ini dinonaktifkan sementara) → Multi-Model Generation, termasuk seluruh logika _confidence thresholding_ dan _hallucination prevention_.

- **Legal Document Preprocessing Engine:** Membangun `preprocessing_service.py` yang komprehensif untuk menangani variasi format dokumen peraturan desa dari 120+ PDF dengan berbagai layout. Pipeline cleaning 9-tahap mencakup perbaikan OCR spacing, spaced-keyword normalization, `_merge_broken_lines()`, `_normalize_letter_numbering()`, dan metadata extraction berbasis `header_block`. Mendukung **PREVIEW MODE** (`save_to_db=False`) untuk validasi hasil chunking tanpa menyentuh database.

- **Multi-Model Abstraction Layer:** Merancang dan mengimplementasikan `HuggingFaceService` yang mengabstraksi komunikasi ke 3 jenis endpoint LLM (HuggingFace Router, Maia Router/OpenAI-compatible, B200 RAFT Server) di balik antarmuka yang seragam, sehingga penggantian model dapat dilakukan tanpa mengubah logika inti.

- **RAFT Model Integration:** Mengintegrasikan model fine-tuned RAFT (_Retrieval-Augmented Fine-Tuning_) yang dilatih khusus untuk domain peraturan desa, termasuk penanganan format input/output khusus dan ekstraksi _thought process_ analisis dari model.

- **RAGAS Evaluation Framework:** Mengimplementasikan sistem evaluasi otomatis menggunakan RAGAS dengan 4 metrik kualitas RAG (faithfulness, answer_relevancy, context_precision, context_recall) + SAS, termasuk integrasi ke database untuk persisten penyimpanan hasil evaluasi dan flag per-sesi yang dapat dikontrol frontend. (Inline LLM-as-a-Judge telah dinonaktifkan agar respons murni dari model.)

- **Database Schema Design:** Merancang schema PostgreSQL yang mencakup manajemen user, sesi percakapan, riwayat chat, dan skema evaluasi metrik RAGAS yang efisien, serta SQL function `match_documents` kustom untuk vector similarity search di Supabase.

- **RAFT Dataset Generation:** Merancang dan menjalankan pipeline pembuatan dataset RAFT v2 (`raft_dataset_retrieval_grounded_*.jsonl`) yang **retrieval-grounded** — mengambil dokumen langsung dari Vector DB (Supabase) dengan pipeline sama seperti inference (Top-10 → Re-ranking → Top-5), lalu menghasilkan `instruction` / `thought_process` (terstruktur: `document_analysis` + `summary`) / `completion` via GPT-4o-mini. Terdapat juga notebook negatif terpisah (`negative_generate_raft_datase.py`) untuk distractor kuat.

---
