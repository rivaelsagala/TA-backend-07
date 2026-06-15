# ============================================================
# Dockerfile — RAG Peraturan Desa (Flask + Python 3.11)
# ============================================================

# ---------- Base image ----------
FROM python:3.11-slim AS base

# Mencegah interactive prompt saat install package system
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- System dependencies ----------
# - libgomp1                          → dependency sentence-transformers (XGBoost)
# - build-essential, pkg-config       → compile psycopg2-binary
#
# [OCR & OpenCV - uncomment jika butuh OCR Tesseract / OpenCV]
# - tesseract-ocr + bahasa Indonesia  → untuk OCR PDF
# - libgl1, libglib2.0-0             → untuk OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        pkg-config \
        curl \
        # tesseract-ocr \
        # tesseract-ocr-ind \
        # tesseract-ocr-eng \
        # libgl1 \
        # libglib2.0-0 \
        # libsm6 \
        # libxext6 \
        # libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# [OCR] Path tesseract di Linux (container) - uncomment jika butuh OCR
# ENV TESSERACT_CMD=/usr/bin/tesseract
# Flask harus bind ke 0.0.0.0 agar bisa diakses dari luar container
ENV FLASK_RUN_HOST=0.0.0.0

# ---------- Working directory ----------
WORKDIR /app

# ---------- Python dependencies (cached layer) ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Application code ----------
COPY . .

# ---------- Port yang digunakan Flask ----------
EXPOSE 5000

# ---------- Default command (development) ----------
# Gunakan gunicorn untuk production:
#   CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "1060"]
CMD ["python", "run.py"]
