# RAG Peraturan Desa

Sistem **Retrieval-Augmented Generation (RAG)** untuk dokumen peraturan desa (Perdes, Perkades, SK, dll.).

- 🔢 **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` — lokal, gratis, tanpa API key
- 🗄️ **Vector DB**: ChromaDB (persistent di `data/chroma_db/`)
- ⚡ **API**: Flask
- 🦙 **LLM**: HuggingFace `meta-llama/Llama-3.1-8B-Instruct` *(untuk tahap generation — menyusul)*
- 🐍 **Python**: 3.9+

---

## 📁 Struktur Proyek

```
RAG/
├── .env                        # Konfigurasi (HF_TOKEN, model, dll.)
├── requirements.txt            # Semua dependensi Python
├── run.py                      # Entry point Flask server
├── ingest.py                   # CLI untuk ingest PDF ke ChromaDB
│
├── src/
│   └── config.py               # Konfigurasi terpusat (Pydantic Settings)
│
├── app/
│   ├── __init__.py             # Inisialisasi Flask app
│   ├── routes.py               # Semua endpoint API
│   ├── utils.py                # Helper functions
│   │
│   ├── handler/
│   │   ├── pdf_handler.py      # Ekstraksi teks PDF (text-based + OCR)
│   │   └── llm_handler.py      # Wrapper HuggingFace LLM
│   │
│   ├── services/
│   │   ├── chunking_service.py # Chunking teks untuk dokumen hukum
│   │   ├── embedding_service.py# Embedding + ChromaDB (store & search)
│   │   ├── rag_service.py      # Orkestrasi pipeline RAG
│   │   ├── file_service.py     # Upload file
│   │   ├── conversation_service.py
│   │   └── history_service.py
│   │
│   └── usecases/
│       ├── rag_use_case.py     # Use case: retrieval query
│       └── ingest_use_case.py  # Use case: ingest dokumen
│
└── data/
    ├── raw/                    # Letakkan PDF di sini
    ├── chroma_db/              # Auto-generated oleh ChromaDB
    └── processed/
```

---

## 🚀 Setup dari Awal (Fresh Clone)

### Prasyarat
- Python **3.9+** sudah terinstall
- Git sudah terinstall

---

### Langkah 1 — Clone & Masuk Direktori

```powershell
git clone <URL_REPOSITORY>
cd RAG
```

---

### Langkah 2 — Buat Virtual Environment

```powershell
python -m venv .venv
```

---

### Langkah 3 — Aktifkan Virtual Environment

```powershell
# Windows PowerShell
.venv\Scripts\activate 

# Windows CMD
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

> **Tanda berhasil**: prompt terminal berubah menjadi `(.venv) PS E:\...\RAG>`

---

### Langkah 4 — Install Dependensi

```powershell
pip install -r requirements.txt
```

> ⏳ Proses ini membutuhkan waktu beberapa menit karena mengunduh `sentence-transformers`, `chromadb`, dll.

---

### Langkah 5 — Siapkan File `.env`

File `.env` sudah ada di root proyek. Pastikan isinya:

```env
# HuggingFace Token (untuk LLM — nanti)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Model LLM (untuk tahap generation — nanti)
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Model Embedding (lokal, gratis, tidak butuh token)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=rag_collection

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retriever
TOP_K_RESULTS=5

# API Flask
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
```

> **HF_TOKEN** hanya dibutuhkan saat menggunakan LLM. Untuk ingest & retrieval, tidak diperlukan.

---

### Langkah 6 — Letakkan File PDF

Salin semua file PDF peraturan desa ke folder `data/raw/`:

```
data/raw/
├── perdes_001_2023.pdf
├── perdes_002_2023.pdf
├── perkades_001_2022.pdf
└── ... (120 file)
```

---

### Langkah 7 — Ingest PDF ke ChromaDB

**Ingest semua PDF sekaligus (rekomendasi untuk 120 file):**

```powershell
python ingest.py --all
```

**Ingest satu file saja:**

```powershell
python ingest.py --file ./data/raw/perdes_001.pdf
```

**Cek statistik hasil ingest:**

```powershell
python ingest.py --stats
```

**Reset ChromaDB (hapus semua data, mulai dari awal):**

```powershell
python ingest.py --reset
```

> ⏳ Ingest pertama kali akan lama karena model embedding (~80MB) diunduh otomatis.
> Proses selanjutnya jauh lebih cepat karena model sudah ter-cache lokal.

---

### Langkah 8 — Jalankan Flask Server

```powershell
python run.py
```

Server berjalan di: **http://127.0.0.1:5000**

---

## 🌐 API Endpoints

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/rag-query` | Cari dokumen relevan |
| POST | `/api/rag-query/scored` | Cari dokumen + skor relevansi |
| POST | `/api/ingest` | Ingest satu file PDF |
| POST | `/api/ingest/batch` | Ingest seluruh folder PDF |
| GET | `/api/stats` | Statistik ChromaDB |
| POST | `/generate-embedding` | Generate embedding satu teks |

### Contoh Request

```powershell
# Cari dokumen relevan
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:5000/api/rag-query" `
  -ContentType "application/json" `
  -Body '{"query": "apa syarat pembentukan BUMDes", "k": 5}'

# Statistik ChromaDB
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:5000/api/stats"

# Ingest batch
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:5000/api/ingest/batch" `
  -ContentType "application/json" `
  -Body '{"directory": "./data/raw"}'
```

---

## 🔑 Penjelasan Pipeline

```
PDF (data/raw/)
    │
    ▼ [1] handler/pdf_handler.py
    │   → PyMuPDF  : PDF text-based (cepat, akurat)
    │   → pytesseract OCR : PDF scanned (fallback otomatis)
    │
    ▼ [2] services/chunking_service.py
    │   → Split per BAB → Pasal → Ayat → Huruf
    │   → Chunk ~1000 chars, overlap 200 chars
    │   → Metadata otomatis: nama file, nomor pasal, bab
    │
    ▼ [3] services/embedding_service.py — store_documents()
    │   → Model: all-MiniLM-L6-v2 (384 dimensi, lokal)
    │   → Simpan ke ChromaDB (data/chroma_db/)
    │
    ▼ [4] services/embedding_service.py — similarity_search()
        → Embed query → cosine similarity → top-k chunks
        → Return dokumen paling relevan
```

---

## ⚙️ Variabel `.env`

| Variabel | Default | Keterangan |
|---|---|---|
| `HF_TOKEN` | — | Token HuggingFace (untuk LLM nanti) |
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model LLM |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model embedding lokal |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Lokasi penyimpanan ChromaDB |
| `CHROMA_COLLECTION_NAME` | `rag_collection` | Nama koleksi ChromaDB |
| `CHUNK_SIZE` | `1000` | Ukuran chunk (karakter) |
| `CHUNK_OVERLAP` | `200` | Overlap antar chunk |
| `TOP_K_RESULTS` | `5` | Jumlah dokumen yang diambil |
| `API_HOST` | `0.0.0.0` | Host Flask |
| `API_PORT` | `8000` | Port Flask |

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (lokal) |
| Vector DB | ChromaDB |
| Framework | LangChain + Flask |
| Config | Pydantic Settings + python-dotenv |
| PDF Parser | PyMuPDF + pytesseract (OCR fallback) |
| Logging | Loguru |
| LLM *(menyusul)* | HuggingFace `meta-llama/Llama-3.1-8B-Instruct` |

- 🔗 **Framework**: LangChain dengan LCEL pattern  
- 🗄️ **Vector DB**: ChromaDB (persistent)
- 🔢 **Embedding**: nomic-embed-text (via Ollama)
- ⚡ **API**: Flask dengan endpoints lengkap
- 📄 **Processing**: Batch processing untuk 120+ PDF files
- 🚀 **Performance**: Multi-threading untuk processing paralel

---

## 🆕 Fitur Utama

### ✨ Batch PDF Processing
- Proses ratusan file PDF secara paralel
- Multi-threading untuk efisiensi maksimal
- Progress tracking dan error handling
- Metadata enrichment untuk setiap dokumen

### 📊 Advanced Chunking
- Recursive character text splitting
- Token-based chunking (opsional)  
- Configurable chunk size dan overlap
- Chunk statistics dan optimization

### 🎯 Smart Embedding
- Batch embedding untuk efisiensi
- Persistent ChromaDB storage
- Collection management (create/append/reset)
- Similarity search dengan scoring

### 🔧 Comprehensive API
- Document processing endpoints
- Batch processing management
- Statistics dan monitoring
- Error handling dan logging

---

## 📁 Struktur Proyek

```
RAG/
├── src/                          # Core RAG modules  
│   ├── config.py                 # Pydantic settings & konfigurasi
│   ├── loader.py                 # Document loaders (PDF, DOCX, TXT)
│   ├── chunker.py                # Text chunking & splitting
│   ├── embedder.py               # Embedding service + ChromaDB
│   ├── retriever.py              # Similarity search & retrieval
│   └── rag_chain.py              # RAG pipeline (LangChain LCEL)
├── app/                          # Flask application
│   ├── routes.py                 # API endpoints
│   ├── services/
│   │   └── document_processing_service.py  # Enhanced PDF processing
│   └── ...
├── data/
└── requirements.txt              # Dependencies
```

---

## 🚀 Quick Start

### 1. Automated Setup

```bash
# Clone repository dan masuk ke direktori
cd RAG

# Jalankan setup otomatis
python setup.py

# Atau step-by-step:
python setup.py --install      # Install dependencies
python setup.py --pull-models  # Pull Ollama models  
python setup.py --check        # Check semua dependencies
```

### 2. Manual Setup (jika diperlukan)

#### Prasyarat
- **Python 3.8+**
- **Ollama** ([Download](https://ollama.ai))

#### Install Ollama Models
```bash
ollama pull llama3.1:8b         # LLM model
ollama pull nomic-embed-text    # Embedding model
```

#### Virtual Environment  
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
```

#### Dependencies
```bash
pip install -r requirements.txt
```

#### Environment Variables
```bash
copy .env.example .env          # Windows  
# cp .env.example .env          # Linux/Mac
```

---

## 📄 Processing PDF Files

### 🔥 Batch Processing (Recommended untuk 120+ files)

#### CLI Commands
```bash
# Process seluruh direktori PDF
python ingest.py --batch-dir ./data/raw/ --workers 8 --new-collection

# Process dengan pattern spesifik
python ingest.py --path ./data/raw/ --pattern "**/*.pdf" --workers 4

# Append ke collection existing
python ingest.py --batch-dir ./data/raw/new_pdfs/ --workers 4

# Check statistics
python ingest.py --stats

# Reset collection
python ingest.py --reset
```

#### Parameters
- `--workers`: Jumlah thread untuk parallel processing (default: 4)
- `--new-collection`: Buat collection baru (hapus yang lama)
- `--pattern`: Pattern file (default: `**/*.pdf`)

### 📊 Expected Performance
- **120 PDF files** (~50MB total): **~10-15 menit** (8 workers)
- **Throughput**: ~8-12 files per menit
- **Memory usage**: ~500MB-1GB peak

---

## 🌐 API Server

### Start Server
```bash
python run.py
```

Server berjalan di: **http://localhost:5000**

### 🔗 API Endpoints

#### Document Processing
```bash
# Process single PDF
POST /api/documents/process-single
{
    "pdf_path": "/path/to/file.pdf",
    "metadata": {"category": "regulation"}
}

# Batch process multiple PDFs
POST /api/documents/process-batch  
{
    "pdf_paths": ["/path/1.pdf", "/path/2.pdf"],
    "max_workers": 4,
    "create_new_collection": false
}

# Process entire directory
POST /api/documents/process-directory
{
    "directory_path": "/path/to/pdfs/",
    "pattern": "**/*.pdf", 
    "max_workers": 8,
    "create_new_collection": false
}

# Get processing statistics
GET /api/documents/stats

# Reset collection  
DELETE /api/documents/reset-collection
```

#### RAG Query
```bash
# Query documents
POST /api/rag-query
{
    "query": "Apa itu peraturan desa tentang..."
}
```

---

## 🧪 Testing & Validation

### Run Tests
```bash
# Test semua komponen
python test_rag.py --test-all

# Test spesifik
python test_rag.py --test-processing   # Document processing
python test_rag.py --test-embedding    # Embedding system  
python test_rag.py --test-api         # API endpoints
python test_rag.py --test-performance  # Performance test
```

---

## 📈 Performance Tips

### Untuk 120+ PDF Files

#### Optimal Settings
```env
# .env file
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=10
MAX_WORKERS=8               # Adjust based on CPU cores
MAX_FILE_SIZE_MB=50
```

#### Processing Strategy
1. **Batch processing** dengan `--workers 6-8`
2. **New collection** untuk reset penuh: `--new-collection`
3. **Monitor memory** usage selama processing
4. **Check logs** di `./logs/` untuk troubleshooting

#### Hardware Recommendations
- **CPU**: 6+ cores untuk optimal parallel processing
- **RAM**: 8GB+ (peak usage ~1-2GB untuk 120 files)
- **Storage**: SSD untuk ChromaDB performance

---

## 📝 Workflow untuk 120 PDF Files

### Step 1: Preparation
```bash
# 1. Setup sistem
python setup.py

# 2. Letakkan semua PDF di ./data/raw/
# 3. Verify files
ls ./data/raw/*.pdf | wc -l  # Should show 120
```

### Step 2: Processing  
```bash
# Start batch processing (estimated 10-15 minutes)
python ingest.py --batch-dir ./data/raw/ --workers 8 --new-collection

# Monitor progress di terminal dan logs/ingest.log
```

### Step 3: Validation
```bash
# Check results
python ingest.py --stats

# Test search
python test_rag.py --test-embedding

# Start API
python run.py
```

### Step 4: Usage
```bash
# Query via API
curl -X POST http://localhost:5000/api/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "peraturan tentang..."}'
```

---

## 🔧 Configuration

### Key Settings (src/config.py)
```python
CHUNK_SIZE = 1000           # Chunk size in characters
CHUNK_OVERLAP = 200         # Overlap between chunks
EMBEDDING_MODEL = "nomic-embed-text"  
LLM_MODEL = "llama3.1:8b"
BATCH_SIZE = 10             # Embedding batch size
```

### Directory Structure for 120 PDFs
```
data/
├── raw/                    # Your 120+ PDF files here
│   ├── doc1.pdf
│   ├── doc2.pdf  
│   ├── ...
│   └── doc120.pdf
├── chroma_db/              # Auto-generated vector DB
└── processed/              # Optional preprocessing results
```

---

## 🔍 Contoh Query via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Apa isi dari dokumen ini?"}'
```

---

## ⚙️ Konfigurasi (.env)

| Variable               | Default               | Deskripsi                        |
|------------------------|-----------------------|----------------------------------|
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | URL Ollama server             |
| `LLM_MODEL`            | `llama3.1:8b`         | Model LLM yang digunakan         |
| `EMBEDDING_MODEL`      | `nomic-embed-text`    | Model embedding                  |
| `CHROMA_PERSIST_DIR`   | `./data/chroma_db`    | Direktori penyimpanan ChromaDB   |
| `CHROMA_COLLECTION_NAME` | `rag_collection`   | Nama collection ChromaDB         |
| `CHUNK_SIZE`           | `1000`                | Ukuran chunk (karakter)          |
| `CHUNK_OVERLAP`        | `200`                 | Overlap antar chunk              |
| `TOP_K_RESULTS`        | `5`                   | Jumlah dokumen yang diambil      |

---

## 🛠️ Tech Stack

| Komponen     | Teknologi                   |
|--------------|-----------------------------|
| LLM          | Llama 3.1 8B via Ollama     |
| Framework    | LangChain (LCEL)            |
| Vector DB    | ChromaDB                    |
| Embedding    | nomic-embed-text via Ollama |
| API Server   | FastAPI + Uvicorn           |
| Config       | Pydantic Settings + dotenv  |
| Logging      | Loguru                      |
