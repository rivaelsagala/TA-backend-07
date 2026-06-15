# Docker Guide — RAG Peraturan Desa

Panduan lengkap menjalankan aplikasi RAG Peraturan Desa menggunakan Docker.

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Instalasi Docker](#2-instalasi-docker)
3. [Clone Repository](#3-clone-repository)
4. [Setup File .env](#4-setup-file-env)
5. [Build & Jalankan Aplikasi](#5-build--jalankan-aplikasi)
6. [Verifikasi Aplikasi Berjalan](#6-verifikasi-aplikasi-berjalan)
7. [Perintah Docker Sehari-hari](#7-perintah-docker-sehari-hari)
8. [Troubleshooting](#8-troubleshooting)
9. [Struktur File Docker](#9-struktur-file-docker)
10. [FAQ](#10-faq)

---

## 1. Prasyarat

Pastikan komputer kamu sudah terinstall:

| Software       | Versi Minimum | Cek Versi         |
|---------------|--------------|-------------------|
| **Docker Desktop** | v24+         | `docker --version` |
| **Git**            | v2.30+       | `git --version`    |

> **Catatan**: Docker Desktop sudah termasuk Docker Engine + Docker Compose. Tidak perlu install terpisah.

---

## 2. Instalasi Docker

### Windows
1. Download **Docker Desktop for Windows**: https://www.docker.com/products/docker-desktop/
2. Jalankan installer, ikuti langkah-langkahnya
3. Pastikan **WSL 2** sudah aktif (Docker Desktop akan otomatis meminta ini)
4. Restart komputer jika diminta
5. Buka Docker Desktop, tunggu sampai icon Docker di system tray menunjukkan **"Docker Desktop is running"**

### Verifikasi Instalasi
Buka terminal (PowerShell / CMD) dan jalankan:
```bash
docker --version
docker compose version
docker run hello-world
```
Jika muncul pesan "Hello from Docker!", berarti Docker sudah terinstall dengan benar.

---

## 3. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

> Ganti `YOUR_USERNAME/YOUR_REPO` dengan URL repository yang benar.

---

## 4. Setup File .env

File `.env` berisi API keys dan konfigurasi database. File ini **tidak di-commit** ke Git (masuk `.gitignore`) karena berisi data sensitif.

### Langkah-langkah:

**4.1. Salin template `.env.example` menjadi `.env`:**

```bash
# Windows (PowerShell)
copy .env.example .env

# Windows (CMD)
copy .env.example .env

# Linux / Mac
cp .env.example .env
```

**4.2. Edit file `.env`** menggunakan text editor (VS Code, Notepad, dll):

```bash
# Contoh menggunakan VS Code
code .env

# Atau Notepad (Windows)
notepad .env
```

**4.3. Isi nilai-nilai berikut** (minta ke pemilik repo jika tidak tahu):

```env
# WAJIB diisi:
OPENAI_API_KEY=sk-xxxxxxxxxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGci...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
HF_TOKEN=hf_xxxxxxxxx

# Boleh dikosongkan / pakai default:
DB_HOST=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

> **PENTING**: Jangan pernah commit file `.env` ke Git!

---

## 5. Build & Jalankan Aplikasi

### Build pertama kali (butuh waktu ~5-10 menit)

```bash
docker compose up --build
```

Proses yang terjadi:
1. Download base image Python 3.11 (~150 MB)
2. Install system packages (Tesseract OCR, OpenCV libs, dll)
3. Install Python dependencies dari `requirements.txt` (~2-3 GB termasuk PyTorch)
4. Copy source code ke dalam container
5. Jalankan aplikasi Flask

> **Tips**: Build pertama kali akan memakan waktu lama karena download dependencies. Build selanjutnya akan jauh lebih cepat karena Docker caching.

### Jalankan di background (tanpa melihat log)

```bash
docker compose up -d --build
```

### Lihat log aplikasi

```bash
docker compose logs -f
```

Tekan `Ctrl+C` untuk keluar dari log (container tetap berjalan di background).

---

## 6. Verifikasi Aplikasi Berjalan

### Cek container status
```bash
docker compose ps
```
Pastikan status menunjukkan **"Up"** atau **"healthy"**.

### Test API endpoint
Buka browser atau gunakan curl:

```bash
# Cek info model
curl http://localhost:5000/api/models

# Atau buka di browser:
# http://localhost:5000/api/models
```

Jika mendapat response JSON, berarti aplikasi sudah berjalan dengan baik!

### Test via Postman / Thunder Client
```
POST http://localhost:5000/api/chat
Content-Type: application/json

{
  "message": "Apa isi peraturan desa nomor 7?",
  "session_id": 1
}
```

---

## 7. Perintah Docker Sehari-hari

| Aksi | Perintah |
|------|----------|
| **Jalankan** (tanpa rebuild) | `docker compose up` |
| **Jalankan** (rebuild jika ada perubahan) | `docker compose up --build` |
| **Jalankan di background** | `docker compose up -d` |
| **Hentikan** container | `docker compose down` |
| **Hentikan + hapus volume** | `docker compose down -v` |
| **Lihat log** | `docker compose logs -f` |
| **Lihat log 100 baris terakhir** | `docker compose logs --tail=100` |
| **Status container** | `docker compose ps` |
| **Restart container** | `docker compose restart` |
| **Masuk ke dalam container** (shell) | `docker compose exec rag-app bash` |
| **Rebuild ulang dari awal** | `docker compose build --no-cache` |

---

## 8. Troubleshooting

### Problem: "port is already allocated"
Port 5000 sudah dipakai program lain.

**Solusi**: Edit port mapping di `docker-compose.yml`:
```yaml
ports:
  - "5001:5000"   # Ganti 5000 → 5001 di sisi host
```
Akses aplikasi di `http://localhost:5001`

---

### Problem: "Cannot connect to the Docker daemon"
Docker Desktop belum berjalan.

**Solusi**: Buka Docker Desktop dan tunggu sampai running.

---

### Problem: Build gagal karena kehabisan memori
Sentence-transformers + PyTorch membutuhkan RAM yang cukup besar.

**Solusi**:
1. Buka Docker Desktop → Settings → Resources
2. Naikkan Memory minimal **4 GB** (disarankan 6-8 GB)
3. Klik "Apply & Restart"

---

### Problem: `.env: file not found`
File `.env` belum dibuat.

**Solusi**: Ulangi langkah [Setup File .env](#4-setup-file-env).

---

### Problem: Model HuggingFace download gagal / timeout
Koneksi internet lambat atau HuggingFace sedang down.

**Solusi**:
```bash
# Coba rebuild ulang
docker compose build --no-cache
docker compose up
```

Volume `hf-cache` akan menyimpan model yang sudah didownload, jadi tidak perlu download ulang.

---

### Problem: "ImportError: libGL.so.1: cannot open shared object file"
System dependency OpenCV belum terinstall (seharusnya otomatis oleh Dockerfile).

**Solusi**: Rebuild ulang
```bash
docker compose build --no-cache
docker compose up
```

---

### Problem: Tesseract OCR error
**Solusi**: Pastikan tesseract sudah terinstall di container:
```bash
docker compose exec rag-app tesseract --version
```

---

### Masuk ke dalam container untuk debugging
```bash
docker compose exec rag-app bash

# Di dalam container, kamu bisa:
python run.py                  # Jalankan app manual
pip list                        # Cek package terinstall
tesseract --list-langs          # Cek bahasa OCR tersedia
```

---

## 9. Struktur File Docker

```
RAG/
├── Dockerfile           ← Instruksi build image (Python 3.11 + deps)
├── docker-compose.yml   ← Konfigurasi container + volume + port
├── .dockerignore        ← File yang di-exclude dari image
├── .env.example         ← Template variabel environment
├── .env                 ← API keys (TIDAK di-commit, buat sendiri)
└── run.py               ← Entry point aplikasi
```

---

## 10. FAQ

### Q: Apakah saya perlu install Python di komputer saya?
**Tidak!** Docker sudah menyediakan Python 3.11 di dalam container. Semua dependencies diinstall otomatis.

### Q: Apakah saya perlu install Tesseract OCR di Windows?
**Tidak!** Tesseract OCR sudah diinstall di dalam container Docker.

### Q: Apakah data di Supabase akan hilang jika container dihapus?
**Tidak!** Data tersimpan di Supabase cloud, bukan di container. Container hanya menjalankan aplikasi Flask.

### Q: Berapa ukuran Docker image-nya?
Sekitar **3-5 GB** karena includes PyTorch + sentence-transformers + OpenCV. Ini normal.

### Q: Build pertama sangat lambat, apakah normal?
**Ya, sangat normal.** Build pertama harus download semua dependencies. Build selanjutnya akan jauh lebih cepat (1-2 menit) berkat Docker layer caching.

### Q: Bagaimana jika ada update code di repo?
```bash
git pull origin main
docker compose up --build
```

### Q: Bagaimana cara menghapus semua data Docker untuk mulai dari awal?
```bash
# Hapus container + image + volume
docker compose down -v --rmi all

# Kemudian build ulang
docker compose up --build
```

---

## Quick Start (Ringkasan Cepat)

Untuk yang tidak mau baca panjang-panjang:

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Buat file .env
copy .env.example .env
# Edit .env dan isi API keys

# 3. Jalankan
docker compose up --build

# 4. Buka browser
# http://localhost:5000/api/models
```

---

*Jika masih ada kendala, hubungi pemilik repository.*
