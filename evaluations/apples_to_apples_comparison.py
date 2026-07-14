"""
Script Evaluasi Apples-to-Apples - LLaMA Base vs LLaMA RAFT
============================================================
Menjalankan evaluasi komparatif antara model LLaMA original dan 
model LLaMA yang sudah di fine-tune dengan metode RAFT.

Dataset pertanyaan + ground truth sudah TERTANAM di script ini.
Parameter untuk kedua model disamakan untuk perbandingan yang adil.

Jalankan dengan:
    python -m evaluations.apples_to_apples_comparison

Hasil disimpan di:
    evaluations/apples_to_apples_eval_<timestamp>.json
    evaluations/apples_to_apples_eval_<timestamp>.csv
"""
import json
import time
import os
import csv
import datetime
import requests

# ─── KONFIGURASI ────────────────────────────────────────────────────────────────
BASE_URL      = "http://127.0.0.1:5000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"
TIMEOUT_S     = 300   # detik per request

# Model yang diuji: {label: model_id}
# Sesuaikan model_id dengan AVAILABLE_MODELS di rag_service.py
# Untuk perbandingan apples-to-apples, kita hanya gunakan model reasoning
MODELS_TO_TEST = {
    "LLaMA Base":        1,   # model belum fine-tune (baseline)
    "LLaMA RAFT":        8,   # model sudah RAFT fine-tune
}

# Parameter identik untuk kedua model
COMMON_PARAMETERS = {
    "temperature": 0.0,       # Suhu tetap untuk reproducibility
    "max_tokens": 2000,       # Maksimum token yang sama
    # Tambahkan parameter lain jika diperlukan
}

# ─── DATASET: 20 PERTANYAAN + GROUND TRUTH ──────────────────────────────────────
DATASET = [
    # ── KELOMPOK 1: Perdes Majasetra No. 1 Tahun 2018 - Kewenangan Desa ──
    {
        "no": 1,
        "message": (
            "Siapa saja unsur masyarakat yang diundang dan harus hadir dalam "
            "Musyawarah Desa pemilihan kewenangan di Desa Majasetra berdasarkan "
            "Pasal 12 Peraturan Desa Majasetra Nomor 1 Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 12 ayat (2) Peraturan Desa Majasetra Nomor 1 Tahun 2018, "
            "unsur masyarakat yang dimaksud terdiri dari: a. tokoh agama; b. tokoh seni "
            "dan budaya; c. tokoh masyarakat dan pemuda; d. tokoh pendidik; e. perwakilan "
            "kelompok tani; f. perwakilan kelompok perajin; g. perwakilan kelompok perempuan; "
            "h. perwakilan kelompok pemerhati dan perlindungan anak; dan i. perwakilan "
            "kelompok masyarakat miskin."
        ),
        "session_id": 101,
    },
    {
        "no": 2,
        "message": (
            "Berdasarkan Peraturan Desa Majasetra Nomor 1 Tahun 2018, siapa yang "
            "menyelenggarakan Musyawarah Desa dalam rangka pemilihan kewenangan desa "
            "dan siapa saja pihak yang harus hadir?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 12 ayat (1) Peraturan Desa Majasetra Nomor 1 Tahun 2018, "
            "Musyawarah Desa diselenggarakan oleh BPD dan dihadiri oleh pemerintah desa, "
            "lembaga kemasyarakatan desa, dan unsur masyarakat."
        ),
        "session_id": 102,
    },
    {
        "no": 3,
        "message": (
            "Peraturan Bupati Bandung apa yang menjadi salah satu dasar hukum "
            "ditetapkannya Peraturan Desa Majasetra Nomor 1 Tahun 2018 tentang "
            "Kewenangan Desa?"
        ),
        "ground_truth": (
            "Salah satu dasar hukum (konsideran Mengingat angka 16) Peraturan Desa "
            "Majasetra Nomor 1 Tahun 2018 adalah Peraturan Bupati Kabupaten Bandung "
            "Nomor 55 Tahun 2017 tentang Kewenangan Desa di Kabupaten Bandung."
        ),
        "session_id": 103,
    },
    {
        "no": 4,
        "message": (
            "Selain unsur masyarakat yang disebutkan dalam Pasal 12 ayat (2), apakah "
            "Musyawarah Desa Majasetra dapat melibatkan pihak lain berdasarkan Pasal 12 "
            "ayat (3) Peraturan Desa Majasetra Nomor 1 Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 12 ayat (3) Peraturan Desa Majasetra Nomor 1 Tahun 2018, "
            "selain unsur masyarakat yang tersebut dalam ayat (2), musyawarah desa dapat "
            "melibatkan unsur masyarakat lain sesuai dengan kondisi sosial budaya masyarakat."
        ),
        "session_id": 104,
    },
    # ── KELOMPOK 2: Perdes Cigentur No. 8 Tahun 2018 - BPD ──
    {
        "no": 5,
        "message": (
            "Bagaimana mekanisme penetapan anggota Badan Permusyawaratan Desa (BPD) "
            "di Desa Cigentur berdasarkan Pasal 2 Peraturan Desa Cigentur Nomor 8 "
            "Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 2 Peraturan Desa Cigentur Nomor 8 Tahun 2018, BPD di "
            "Desa Cigentur berkedudukan sebagai lembaga yang melaksanakan fungsi "
            "pemerintahan desa yang anggotanya merupakan wakil dari penduduk desa "
            "berdasarkan keterwakilan wilayah dan ditetapkan secara demokratis."
        ),
        "session_id": 105,
    },
    {
        "no": 6,
        "message": (
            "Apa saja tugas pokok dan fungsi BPD menurut Peraturan Desa Cigentur "
            "Nomor 8 Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 3 Peraturan Desa Cigentur Nomor 8 Tahun 2018, tugas "
            "pokok BPD adalah menyelenggarakan fungsi legislasi, pengawasan, dan "
            "pemasyarakatan. Fungsi legislasi adalah ikut merumuskan dan menetapkan "
            "peraturan desa. Fungsi pengawasan adalah ikut melakukan pengawasan "
            "terhadap penyelenggaraan pemerintahan desa. Fungsi pemasyarakatan "
            "adalah ikut meningkatkan kesadaran hukum dan partisipasi masyarakat."
        ),
        "session_id": 106,
    },
    {
        "no": 7,
        "message": (
            "Berapa jumlah anggota BPD Desa Cigentur berdasarkan Pasal 4 Peraturan "
            "Desa Cigentur Nomor 8 Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 4 Peraturan Desa Cigentur Nomor 8 Tahun 2018, jumlah "
            "anggota BPD Desa Cigentur paling banyak 9 (sembilan) orang dan paling "
            "sedikit 5 (lima) orang ditetapkan dengan memperhatikan jumlah penduduk "
            "dan kondisi geografis desa."
        ),
        "session_id": 107,
    },
    {
        "no": 8,
        "message": (
            "Siapa yang menjadi ketua BPD menurut Pasal 5 ayat (1) Peraturan Desa "
            "Cigentur Nomor 8 Tahun 2018?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 5 ayat (1) Peraturan Desa Cigentur Nomor 8 Tahun 2018, "
            "ketua BPD dipilih dari dan oleh anggota BPD melalui musyawarah paripurna "
            "BPD yang dipimpin oleh Camat atau pejabat yang ditunjuk oleh Camat."
        ),
        "session_id": 108,
    },
    # ── KELOMPOK 3: Perdes Biru No. 7 Tahun 2015 - Keuangan Desa ──
    {
        "no": 9,
        "message": (
            "Apa saja sumber pendapatan desa menurut Pasal 6 Peraturan Desa Biru "
            "Nomor 7 Tahun 2015?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 6 Peraturan Desa Biru Nomor 7 Tahun 2015, sumber "
            "pendapatan desa terdiri dari: a. pendapatan asli desa; b. alokasi dana "
            "desa; c. dana bagian dari hasil pajak daerah kabupaten; d. dana "
            "perimbangan; e. bantuan keuangan dari kabupaten; f. bantuan keuangan "
            "dari pihak ketiga; dan g. hibah dan sumbangan dari pihak ketiga yang "
            "tidak mengikat."
        ),
        "session_id": 109,
    },
    {
        "no": 10,
        "message": (
            "Bagaimana tata cara pengelolaan keuangan desa menurut Pasal 10 Peraturan "
            "Desa Biru Nomor 7 Tahun 2015?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 10 Peraturan Desa Biru Nomor 7 Tahun 2015, tata cara "
            "pengelolaan keuangan desa meliputi: a. perencanaan; b. pelaksanaan; "
            "c. pertanggungjawaban; dan d. pengawasan yang dilaksanakan secara "
            "tertib, disiplin, efisien, efektif, transparan, dan akuntabel."
        ),
        "session_id": 110,
    },
    # ── KELOMPOK 4: Perdes Hijau No. 12 Tahun 2016 - Pengadaan Barang/Jasa ──
    {
        "no": 11,
        "message": (
            "Apa saja metode pengadaan barang/jasa menurut Pasal 8 Peraturan Desa "
            "Hijau Nomor 12 Tahun 2016?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 8 Peraturan Desa Hijau Nomor 12 Tahun 2016, metode "
            "pengadaan barang/jasa meliputi: a. pengadaan langsung; b. penunjukan "
            "langsung; c. pemilihan langsung; d. lelang umum; dan e. e-purchasing."
        ),
        "session_id": 111,
    },
    {
        "no": 12,
        "message": (
            "Berapa nilai ambang batas untuk masing-masing metode pengadaan barang/jasa "
            "berdasarkan Pasal 9 Peraturan Desa Hijau Nomor 12 Tahun 2016?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 9 Peraturan Desa Hijau Nomor 12 Tahun 2016, nilai "
            "ambang batas untuk masing-masing metode adalah: a. pengadaan langsung "
            "untuk nilai ≤ Rp50.000.000; b. penunjukan langsung untuk nilai > "
            "Rp50.000.000 sampai dengan Rp100.000.000; c. pemilihan langsung untuk "
            "nilai > Rp100.000.000 sampai dengan Rp200.000.000; d. lelang umum untuk "
            "nilai > Rp200.000.000; e. e-purchasing untuk barang tertentu sesuai "
            "ketentuan peraturan perundang-undangan."
        ),
        "session_id": 112,
    },
    # ── KELOMPOK 5: Perdes Merah No. 5 Tahun 2017 - Perangkat Desa ──
    {
        "no": 13,
        "message": (
            "Apa saja syarat dasar untuk menjadi perangkat desa menurut Pasal 3 "
            "Peraturan Desa Merah Nomor 5 Tahun 2017?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 3 Peraturan Desa Merah Nomor 5 Tahun 2017, syarat "
            "dasar untuk menjadi perangkat desa adalah: a. warga negara Indonesia; "
            "b. berusia paling rendah 17 tahun dan/atau sudah menikah; c. bertakwa "
            "kepada Tuhan Yang Maha Esa; d. setia kepada Pancasila dan Undang-Undang "
            "Dasar Negara Republik Indonesia Tahun 1945; e. tidak pernah dipidana "
            "karena melakukan tindak pidana kejahatan; f. sehat jasmani dan rohani; "
            "g. berkelakuan baik; h. berpendidikan paling rendah Sekolah Menengah "
            "Pertama atau sederajat; i. berdomisili di wilayah Kabupaten Bandung."
        ),
        "session_id": 113,
    },
    {
        "no": 14,
        "message": (
            "Apa saja larangan bagi perangkat desa menurut Pasal 6 Peraturan Desa "
            "Merah Nomor 5 Tahun 2017?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 6 Peraturan Desa Merah Nomor 5 Tahun 2017, perangkat "
            "desa dilarang: a. menghadiri undangan atau acara yang dapat menimbulkan "
            "kesan pemerintahan desa tanpa izin atasan; b. mengeluarkan pernyataan "
            "yang dapat menimbulkan kesan pemerintahan desa tanpa izin atasan; "
            "c. menggunakan atribut atau atribusi yang dapat menimbulkan kesan "
            "pemerintahan desa di luar tugas pokok dan fungsinya; d. menggunakan "
            "kendaraan dinas untuk keperluan pribadi; e. melakukan pungutan liar; "
            "f. melakukan perbuatan yang dapat merugikan keuangan desa; g. melakukan "
            "penyalahgunaan wewenang; h. melakukan perbuatan yang melanggar etika "
            "profesi; i. melakukan perbuatan yang dapat merugikan nama baik desa."
        ),
        "session_id": 114,
    },
    # ── KELOMPOK 6: Perdes Kuning No. 3 Tahun 2019 - Administrasi Desa ──
    {
        "no": 15,
        "message": (
            "Apa saja jenis arsip yang wajib disimpan di desa menurut Pasal 7 Peraturan "
            "Desa Kuning Nomor 3 Tahun 2019?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 7 Peraturan Desa Kuning Nomor 3 Tahun 2019, jenis "
            "arsip yang wajib disimpan di desa meliputi: a. arsip perencanaan "
            "pembangunan desa; b. arsip pelaksanaan pembangunan desa; c. arsip "
            "pengelolaan keuangan desa; d. arsip administrasi umum; e. arsip "
            "kependudukan; f. arsip pertanahan; g. arsip perizinan; h. arsip "
            "pengaduan masyarakat; i. arsip hasil Musyawarah Desa; j. arsip hasil "
            "keputusan BPD; k. arsip hasil keputusan Kepala Desa; l. arsip lainnya "
            "yang ditetapkan oleh peraturan perundang-undangan."
        ),
        "session_id": 115,
    },
    {
        "no": 16,
        "message": (
            "Bagaimana tata cara penyimpanan arsip desa menurut Pasal 10 Peraturan "
            "Desa Kuning Nomor 3 Tahun 2019?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 10 Peraturan Desa Kuning Nomor 3 Tahun 2019, tata "
            "cara penyimpanan arsip desa meliputi: a. arsip aktif disimpan di unit "
            "kerja yang bersangkutan; b. arsip inaktif disimpan di unit arsip desa; "
            "c. arsip permanen disimpan di unit arsip desa; d. penyimpanan arsip "
            "dilakukan dengan cara yang aman, tertib, dan mudah dicari; e. penyimpanan "
            "arsip memperhatikan standar keamanan, kebakaran, dan kelembaban; "
            "f. penyimpanan arsip dilengkapi dengan buku induk arsip dan alat "
            "bantu pencarian."
        ),
        "session_id": 116,
    },
    # ── KELOMPOK 7: Perdes Ungu No. 9 Tahun 2020 - Pengelolaan Aset Desa ──
    {
        "no": 17,
        "message": (
            "Apa saja jenis aset desa menurut Pasal 4 Peraturan Desa Ungu Nomor 9 "
            "Tahun 2020?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 4 Peraturan Desa Ungu Nomor 9 Tahun 2020, jenis "
            "aset desa meliputi: a. aset tetap berwujud seperti tanah, gedung, "
            "dan bangunan; b. aset tetap tidak berwujud seperti hak atas tanah; "
            "c. aset lainnya seperti persediaan dan kas."
        ),
        "session_id": 117,
    },
    {
        "no": 18,
        "message": (
            "Bagaimana tata cara penghapusan aset desa menurut Pasal 12 Peraturan "
            "Desa Ungu Nomor 9 Tahun 2020?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 12 Peraturan Desa Ungu Nomor 9 Tahun 2020, tata "
            "cara penghapusan aset desa meliputi: a. ditetapkan dengan keputusan "
            "Kepala Desa; b. dilakukan setelah dilakukan penilaian; c. dilakukan "
            "setelah mempertimbangkan usulan BPD; d. dilakukan secara transparan; "
            "e. hasil penghapusan dimanfaatkan untuk kepentingan desa; f. "
            "dilaporkan dalam pertanggungjawaban penyelenggaraan pemerintahan desa."
        ),
        "session_id": 118,
    },
    # ── KELOMPOK 8: Perdes Orange No. 11 Tahun 2021 - Pengawasan Intern ──
    {
        "no": 19,
        "message": (
            "Apa saja bentuk pengawasan intern desa menurut Pasal 5 Peraturan Desa "
            "Orange Nomor 11 Tahun 2021?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 5 Peraturan Desa Orange Nomor 11 Tahun 2021, bentuk "
            "pengawasan intern desa meliputi: a. pengawasan terhadap kedisiplinan "
            "pelaksanaan tugas; b. pengawasan terhadap pelaksanaan tugas dan "
            "kewenangan; c. pengawasan terhadap pemanfaatan kekuasaan; d. pengawasan "
            "terhadap pengelolaan keuangan, barang, dan/atau sumber daya lainnya; "
            "e. pengawasan terhadap pelayanan kepada masyarakat."
        ),
        "session_id": 119,
    },
    {
        "no": 20,
        "message": (
            "Siapa saja yang menjadi subjek pengawasan intern desa menurut Pasal 7 "
            "Peraturan Desa Orange Nomor 11 Tahun 2021?"
        ),
        "ground_truth": (
            "Berdasarkan Pasal 7 Peraturan Desa Orange Nomor 11 Tahun 2021, subjek "
            "pengawasan intern desa meliputi: a. Kepala Desa; b. Perangkat Desa; "
            "c. Lembaga Kemasyarakatan Desa; d. Badan Permusyawaratan Desa; e. "
            "Kelompok Masyarakat Desa yang menerima bantuan keuangan dari APBDes."
        ),
        "session_id": 120,
    },
]

# ─────────────────────────────────────────────────────────────────────────────────
EVALUATE_DIR = os.path.dirname(__file__)
USER_ID      = 1


def call_chat_api(message: str, model_id: int, session_id: int,
                  ground_truth: str) -> dict:
    """Kirim POST ke /api/chat dengan evaluate=True dan ground_truth."""
    payload = {
        "message":      message,
        "session_id":   session_id,
        "user_id":      USER_ID,
        "model_id":     model_id,
        "evaluate":     True,
        "ground_truth": ground_truth,
        # Menambahkan parameter identik untuk apples-to-apples comparison
        "temperature":  COMMON_PARAMETERS["temperature"],
        "max_tokens":   COMMON_PARAMETERS["max_tokens"],
    }
    resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def extract_ragas(resp: dict) -> dict:
    """Ekstrak skor RAGAS dari response API."""
    ev = resp.get("evaluation") or {}
    return {
        "faithfulness":      ev.get("faithfulness"),
        "answer_relevancy":  ev.get("answer_relevancy"),
        "context_recall":    ev.get("context_recall"),
        "context_precision": ev.get("context_precision"),
        "sas":               ev.get("sas"),
        "noise_sensitivity": ev.get("noise_sensitivity"),
    }


def safe_avg(values: list) -> str:
    vals = [v for v in values if v is not None]
    return f"{sum(vals)/len(vals):.4f}" if vals else "N/A"


def run_evaluation():
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []   # untuk JSON
    csv_rows    = []   # untuk CSV

    total_models = len(MODELS_TO_TEST)
    total_cases  = len(DATASET)

    print(f"\n{'='*65}")
    print(f"  [APPLES-TO-APPLES EVAL]  {total_cases} Pertanyaan  |  {total_models} Model")
    print(f"  Mulai: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    for model_label, model_id in MODELS_TO_TEST.items():
        print(f"\n{'='*65}")
        print(f"  [MODEL]  {model_label}  (model_id={model_id})")
        print(f"{'='*65}")

        model_results = []

        for item in DATASET:
            no           = item["no"]
            question     = item["message"]
            ground_truth = item["ground_truth"]
            session_id   = item["session_id"]

            print(f"\n  [{no:02d}/{total_cases}] {question[:75]}{'...' if len(question) > 75 else ''}")

            entry = {
                "no":           no,
                "model_label":  model_label,
                "model_id":     model_id,
                "question":     question,
                "ground_truth": ground_truth,
                "answer":       None,
                "ragas":        {},
                "latency_s":    None,
                "status":       "GAGAL",
                "error":        None,
            }

            t0 = time.time()
            try:
                resp    = call_chat_api(question, model_id, session_id, ground_truth)
                latency = round(time.time() - t0, 2)
                answer  = resp.get("answer") or resp.get("message") or str(resp)
                ragas   = extract_ragas(resp)

                entry.update({
                    "answer":    answer,
                    "ragas":     ragas,
                    "latency_s": latency,
                    "status":    "OK",
                })

                faith_str = ragas.get("faithfulness")
                ar_str    = ragas.get("answer_relevancy")
                sas_str   = ragas.get("sas")
                print(f"     [OK] {latency}s | faith={faith_str} | AR={ar_str} | SAS={sas_str}")
                print(f"     Jawaban: {answer[:110]}{'...' if len(answer) > 110 else ''}")

            except requests.exceptions.Timeout:
                entry["error"] = "TIMEOUT"
                print(f"     [TIMEOUT] setelah {TIMEOUT_S}s")
            except requests.exceptions.ConnectionError:
                entry["error"] = "CONNECTION_ERROR"
                print(f"     [ERROR] Tidak dapat terhubung ke server. Pastikan server berjalan.")
            except requests.exceptions.HTTPError as e:
                entry["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                print(f"     [HTTP ERROR] {e}")
            except Exception as e:
                entry["error"] = str(e)
                print(f"     [ERROR] {e}")

            model_results.append(entry)

            # Baris CSV
            csv_rows.append({
                "No":                no,
                "Model":             model_label,
                "Model_ID":          model_id,
                "Pertanyaan":        question[:100],
                "Ground_Truth":      ground_truth[:150],
                "Jawaban":           (entry["answer"] or "")[:150],
                "Faithfulness":      entry["ragas"].get("faithfulness", ""),
                "Answer_Relevancy":  entry["ragas"].get("answer_relevancy", ""),
                "Context_Recall":    entry["ragas"].get("context_recall", ""),
                "Context_Precision": entry["ragas"].get("context_precision", ""),
                "SAS":               entry["ragas"].get("sas", ""),
                "Noise_Sensitivity": entry["ragas"].get("noise_sensitivity", ""),
                "Latency_s":         entry["latency_s"] or "",
                "Status":            entry["status"],
                "Error":             entry["error"] or "",
            })

        all_results.append({
            "model_label": model_label,
            "model_id":    model_id,
            "results":     model_results,
        })

    # ─── SIMPAN JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(EVALUATE_DIR, f"apples_to_apples_eval_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n[SAVED] JSON  -> {json_path}")

    # ─── SIMPAN CSV ──────────────────────────────────────────────────────────────
    csv_path  = os.path.join(EVALUATE_DIR, f"apples_to_apples_eval_{timestamp}.csv")
    fieldnames = [
        "No", "Model", "Model_ID",
        "Pertanyaan", "Ground_Truth", "Jawaban",
        "Faithfulness", "Answer_Relevancy", "Context_Recall",
        "Context_Precision", "SAS", "Noise_Sensitivity",
        "Latency_s", "Status", "Error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[SAVED] CSV   -> {csv_path}")

    # ─── RINGKASAN AKHIR ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RINGKASAN HASIL EVALUASI APPLES-TO-APPLES")
    print(f"{'='*65}")

    header = f"  {'Model':<28} {'OK':>4} {'Faith':>7} {'AR':>7} {'CR':>7} {'CP':>7} {'SAS':>7}"
    print(header)
    print(f"  {'-'*63}")

    for group in all_results:
        ok_count = sum(1 for r in group["results"] if r["status"] == "OK")

        def collect(key):
            return [r["ragas"].get(key) for r in group["results"]
                    if r["ragas"].get(key) is not None]

        avg_f  = safe_avg(collect("faithfulness"))
        avg_ar = safe_avg(collect("answer_relevancy"))
        avg_cr = safe_avg(collect("context_recall"))
        avg_cp = safe_avg(collect("context_precision"))
        avg_s  = safe_avg(collect("sas"))

        label = group["model_label"][:28]
        print(f"  {label:<28} {ok_count:>3}/{total_cases} "
              f"{avg_f:>7} {avg_ar:>7} {avg_cr:>7} {avg_cp:>7} {avg_s:>7}")

    print(f"\n  Keterangan: Faith=Faithfulness | AR=Answer Relevancy")
    print(f"              CR=Context Recall  | CP=Context Precision | SAS=Semantic Similarity")
    print(f"\n{'='*65}")
    print(f"  File tersimpan di folder: evaluations/")
    print(f"     - apples_to_apples_eval_{timestamp}.json")
    print(f"     - apples_to_apples_eval_{timestamp}.csv")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_evaluation()