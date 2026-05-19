# API Documentation - RAG System

Dokumentasi lengkap untuk REST API sistem RAG (Retrieval-Augmented Generation) dengan fitur chat history, embedding dokumen PDF, dan manajemen model.

## Base URL
```
http://localhost:5000
```

---

## 📑 Table of Contents
1. [Document & Embedding](#1-document--embedding)
2. [Model Management](#2-model-management)
3. [Chat Management](#3-chat-management)
4. [Chat Session Management](#4-chat-session-management)

---

## 1. Document & Embedding

### Generate Embedding from PDF
Upload dokumen PDF dan generate embedding untuk disimpan ke vector database.

**Endpoint:** `POST /api/generate-embedding`

**Content-Type:** `multipart/form-data`

**Request Body:**
- `file` (file, required): File PDF yang akan diproses

**Response Success (200):**
```json
{
  "status": "success",
  "message": "PDF berhasil diproses dan embedding disimpan",
  "filename": "dokumen_hukum.pdf",
  "chunks_count": 150
}
```

**Response Error (400):**
```json
{
  "error": "Format file tidak didukung. Harap unggah format PDF."
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X POST http://localhost:5000/api/generate-embedding \
  -F "file=@/path/to/dokumen_hukum.pdf"
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/generate-embedding"
files = {'file': open('dokumen_hukum.pdf', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/api/generate-embedding', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 2. Model Management

### Load Fine-tuned Model
Memuat model fine-tuned ke dalam memori server.

**Endpoint:** `POST /api/load-finetuned-model`

**Content-Type:** `application/json`

**Response Success (200):**
```json
{
  "status": "success",
  "message": "Model fine-tuned berhasil dimuat ke memori"
}
```

**Response Error (500):**
```json
{
  "status": "error",
  "message": "Gagal memuat model fine-tuned"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X POST http://localhost:5000/api/load-finetuned-model \
  -H "Content-Type: application/json"
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/load-finetuned-model"
response = requests.post(url)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
fetch('http://localhost:5000/api/load-finetuned-model', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  }
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### Get Model Information
Mendapatkan informasi tentang model yang tersedia.

**Endpoint:** `GET /api/models`

**Response Success (200):**
```json
{
  "status": "success",
  "data": {
    "original_model": {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "description": "Llama 3.1 8B Instruct (Original)",
      "use_finetuned_model": false
    },
    "finetuned_model": {
      "name": "your-username/llama-3.1-8b-finetuned",
      "description": "Llama 3.1 8B Instruct Fine-tuned for Legal Documents",
      "use_finetuned_model": true
    }
  }
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X GET http://localhost:5000/api/models
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/models"
response = requests.get(url)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
fetch('http://localhost:5000/api/models')
.then(response => response.json())
.then(data => console.log(data));
```

---

## 3. Chat Management

### Send Chat Message
Mengirim pesan chat dan mendapatkan response dari sistem RAG.

**Endpoint:** `POST /api/chat`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "message": "Apa itu hukum perdata?",
  "session_id": 1,
  "user_id": 123,
  "use_finetuned_model": false
}
```

**Parameters:**
- `message` (string, required): Pertanyaan/pesan dari user
- `session_id` (integer, required): ID session chat
- `user_id` (integer, required): ID user
- `use_finetuned_model` (boolean, optional): Gunakan model fine-tuned (default: false)

**Response Success (200):**
```json
{
  "status": "success",
  "data": {
    "answer": "Hukum perdata adalah...",
    "sources": [
      {
        "content": "...",
        "metadata": {
          "filename": "dokumen_hukum.pdf",
          "page": 5
        }
      }
    ],
    "session_id": 1,
    "message_id": 42
  }
}
```

**Response Error (400):**
```json
{
  "error": "Format request salah. Butuh message, session_id, dan user_id"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Apa itu hukum perdata?",
    "session_id": 1,
    "user_id": 123,
    "use_finetuned_model": false
  }'
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/chat"
data = {
    "message": "Apa itu hukum perdata?",
    "session_id": 1,
    "user_id": 123,
    "use_finetuned_model": False
}
response = requests.post(url, json=data)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const data = {
  message: "Apa itu hukum perdata?",
  session_id: 1,
  user_id: 123,
  use_finetuned_model: false
};

fetch('http://localhost:5000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 4. Chat Session Management

### Create New Session
Membuat session chat baru untuk user.

**Endpoint:** `POST /api/chat-sessions`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "user_id": 123,
  "session_name": "Diskusi Hukum Perdata"
}
```

**Parameters:**
- `user_id` (integer, required): ID user
- `session_name` (string, optional): Nama session (default: "New Chat")

**Response Success (200):**
```json
{
  "status": "success",
  "data": {
    "session_id": 5,
    "user_id": 123,
    "session_name": "Diskusi Hukum Perdata",
    "created_at": "2026-05-10T10:30:00Z"
  }
}
```

**Response Error (400):**
```json
{
  "error": "Butuh field user_id"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X POST http://localhost:5000/api/chat-sessions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "session_name": "Diskusi Hukum Perdata"
  }'
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/chat-sessions"
data = {
    "user_id": 123,
    "session_name": "Diskusi Hukum Perdata"
}
response = requests.post(url, json=data)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const data = {
  user_id: 123,
  session_name: "Diskusi Hukum Perdata"
};

fetch('http://localhost:5000/api/chat-sessions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### Get All Sessions
Mendapatkan semua session chat untuk user tertentu.

**Endpoint:** `GET /api/chat-sessions?user_id={user_id}`

**Query Parameters:**
- `user_id` (integer, required): ID user

**Response Success (200):**
```json
{
  "status": "success",
  "data": [
    {
      "session_id": 1,
      "user_id": 123,
      "session_name": "Diskusi Hukum Perdata",
      "created_at": "2026-05-10T10:30:00Z",
      "updated_at": "2026-05-10T11:15:00Z"
    },
    {
      "session_id": 2,
      "user_id": 123,
      "session_name": "Pertanyaan Hukum Pidana",
      "created_at": "2026-05-09T14:20:00Z",
      "updated_at": "2026-05-09T15:00:00Z"
    }
  ]
}
```

**Response Error (400):**
```json
{
  "error": "Butuh query parameter user_id"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X GET "http://localhost:5000/api/chat-sessions?user_id=123"
```

**Contoh Penggunaan (Python):**
```python
import requests

url = "http://localhost:5000/api/chat-sessions"
params = {"user_id": 123}
response = requests.get(url, params=params)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const userId = 123;
fetch(`http://localhost:5000/api/chat-sessions?user_id=${userId}`)
.then(response => response.json())
.then(data => console.log(data));
```

---

### Get Chat History
Mendapatkan riwayat chat dari session tertentu.

**Endpoint:** `GET /api/chat-history/{session_id}`

**Path Parameters:**
- `session_id` (integer, required): ID session

**Response Success (200):**
```json
{
  "status": "success",
  "data": {
    "session_id": 1,
    "session_name": "Diskusi Hukum Perdata",
    "messages": [
      {
        "message_id": 1,
        "role": "user",
        "content": "Apa itu hukum perdata?",
        "timestamp": "2026-05-10T10:35:00Z"
      },
      {
        "message_id": 2,
        "role": "assistant",
        "content": "Hukum perdata adalah...",
        "timestamp": "2026-05-10T10:35:05Z",
        "sources": [...]
      }
    ]
  }
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X GET http://localhost:5000/api/chat-history/1
```

**Contoh Penggunaan (Python):**
```python
import requests

session_id = 1
url = f"http://localhost:5000/api/chat-history/{session_id}"
response = requests.get(url)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const sessionId = 1;
fetch(`http://localhost:5000/api/chat-history/${sessionId}`)
.then(response => response.json())
.then(data => console.log(data));
```

---

### Update Session Name
Memperbarui nama session chat.

**Endpoint:** `PUT /api/chat-sessions/{session_id}`

**Content-Type:** `application/json`

**Path Parameters:**
- `session_id` (integer, required): ID session

**Request Body:**
```json
{
  "session_name": "Hukum Perdata - Update"
}
```

**Parameters:**
- `session_name` (string, required): Nama session baru

**Response Success (200):**
```json
{
  "status": "success",
  "message": "Session name berhasil diupdate",
  "data": {
    "session_id": 1,
    "session_name": "Hukum Perdata - Update"
  }
}
```

**Response Error (400):**
```json
{
  "error": "Butuh field session_name"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X PUT http://localhost:5000/api/chat-sessions/1 \
  -H "Content-Type: application/json" \
  -d '{
    "session_name": "Hukum Perdata - Update"
  }'
```

**Contoh Penggunaan (Python):**
```python
import requests

session_id = 1
url = f"http://localhost:5000/api/chat-sessions/{session_id}"
data = {
    "session_name": "Hukum Perdata - Update"
}
response = requests.put(url, json=data)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const sessionId = 1;
const data = {
  session_name: "Hukum Perdata - Update"
};

fetch(`http://localhost:5000/api/chat-sessions/${sessionId}`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### Delete Session
Menghapus session chat dan semua history-nya.

**Endpoint:** `DELETE /api/chat-sessions/{session_id}`

**Path Parameters:**
- `session_id` (integer, required): ID session

**Response Success (200):**
```json
{
  "status": "success",
  "message": "Session berhasil dihapus"
}
```

**Response Error (404):**
```json
{
  "status": "error",
  "message": "Session tidak ditemukan"
}
```

**Contoh Penggunaan (cURL):**
```bash
curl -X DELETE http://localhost:5000/api/chat-sessions/1
```

**Contoh Penggunaan (Python):**
```python
import requests

session_id = 1
url = f"http://localhost:5000/api/chat-sessions/{session_id}"
response = requests.delete(url)
print(response.json())
```

**Contoh Penggunaan (JavaScript/Fetch):**
```javascript
const sessionId = 1;
fetch(`http://localhost:5000/api/chat-sessions/${sessionId}`, {
  method: 'DELETE'
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 🔄 Complete Workflow Example

### Skenario: User baru menggunakan sistem RAG

```python
import requests

BASE_URL = "http://localhost:5000"
USER_ID = 123

# 1. Upload dokumen PDF untuk generate embedding
print("1. Uploading PDF document...")
with open('dokumen_hukum.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/api/generate-embedding", files=files)
    print(response.json())

# 2. Buat session chat baru
print("\n2. Creating new chat session...")
session_data = {
    "user_id": USER_ID,
    "session_name": "Konsultasi Hukum Perdata"
}
response = requests.post(f"{BASE_URL}/api/chat-sessions", json=session_data)
session = response.json()['data']
session_id = session['session_id']
print(f"Session created with ID: {session_id}")

# 3. Kirim pertanyaan pertama
print("\n3. Sending first question...")
chat_data = {
    "message": "Apa yang dimaksud dengan hukum perdata?",
    "session_id": session_id,
    "user_id": USER_ID,
    "use_finetuned_model": False
}
response = requests.post(f"{BASE_URL}/api/chat", json=chat_data)
print(response.json())

# 4. Kirim pertanyaan lanjutan
print("\n4. Sending follow-up question...")
chat_data['message'] = "Bisakah Anda jelaskan lebih detail tentang perjanjian dalam hukum perdata?"
response = requests.post(f"{BASE_URL}/api/chat", json=chat_data)
print(response.json())

# 5. Ambil semua history chat
print("\n5. Getting chat history...")
response = requests.get(f"{BASE_URL}/api/chat-history/{session_id}")
print(response.json())

# 6. Update nama session
print("\n6. Updating session name...")
update_data = {
    "session_name": "Konsultasi Hukum Perdata - Perjanjian"
}
response = requests.put(f"{BASE_URL}/api/chat-sessions/{session_id}", json=update_data)
print(response.json())

# 7. Ambil semua session user
print("\n7. Getting all user sessions...")
response = requests.get(f"{BASE_URL}/api/chat-sessions", params={"user_id": USER_ID})
print(response.json())
```

---

## 📝 Error Handling

Semua endpoint mengikuti format error response yang konsisten:

```json
{
  "status": "error",
  "error": "Deskripsi error"
}
```

**HTTP Status Codes:**
- `200 OK` - Request berhasil
- `400 Bad Request` - Parameter request tidak valid
- `404 Not Found` - Resource tidak ditemukan
- `500 Internal Server Error` - Error di server

---

## 🔐 Notes

1. **Session Management**: Setiap user bisa memiliki multiple session. Setiap session menyimpan history chat secara terpisah.

2. **Model Selection**: Gunakan parameter `use_finetuned_model: true` untuk menggunakan model fine-tuned. Pastikan model sudah di-load terlebih dahulu menggunakan endpoint `/api/load-finetuned-model`.

3. **File Upload**: Endpoint generate embedding hanya menerima file PDF. Maksimal ukuran file tergantung konfigurasi server.

4. **User ID**: Semua endpoint yang berhubungan dengan session membutuhkan `user_id` untuk identifikasi user.

5. **Session Name**: Jika tidak dispesifikasikan saat create session, default nama adalah "New Chat".

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python run.py

# 3. Test dengan cURL
curl http://localhost:5000/api/models
```

---

**Last Updated:** May 10, 2026
