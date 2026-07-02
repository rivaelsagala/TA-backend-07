"""
Script Evaluasi RAGAS - 20 Pertanyaan, 2 Model
================================================
Menjalankan evaluasi RAGAS secara batch untuk 20 pertanyaan
terhadap DUA model yang berbeda secara langsung.

Dataset pertanyaan + ground truth sudah TERTANAM di script ini.

Jalankan dengan:
    python -m evaluations.run_ragas_eval

Hasil disimpan di:
    evaluations/ragas_eval_<timestamp>.json
    evaluations/ragas_eval_<timestamp>.csv
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
MODELS_TO_TEST = {
    "LLaMA Base":        1,   # model belum fine-tune (baseline)
    "LLaMA RAFT":        8,   # model sudah RAFT fine-tune
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
    # {
    #     "no": 3,
    #     "message": (
    #         "Peraturan Bupati Bandung apa yang menjadi salah satu dasar hukum "
    #         "ditetapkannya Peraturan Desa Majasetra Nomor 1 Tahun 2018 tentang "
    #         "Kewenangan Desa?"
    #     ),
    #     "ground_truth": (
    #         "Salah satu dasar hukum (konsideran Mengingat angka 16) Peraturan Desa "
    #         "Majasetra Nomor 1 Tahun 2018 adalah Peraturan Bupati Kabupaten Bandung "
    #         "Nomor 55 Tahun 2017 tentang Kewenangan Desa di Kabupaten Bandung."
    #     ),
    #     "session_id": 103,
    # },
    # {
    #     "no": 4,
    #     "message": (
    #         "Selain unsur masyarakat yang disebutkan dalam Pasal 12 ayat (2), apakah "
    #         "Musyawarah Desa Majasetra dapat melibatkan pihak lain berdasarkan Pasal 12 "
    #         "ayat (3) Peraturan Desa Majasetra Nomor 1 Tahun 2018?"
    #     ),
    #     "ground_truth": (
    #         "Berdasarkan Pasal 12 ayat (3) Peraturan Desa Majasetra Nomor 1 Tahun 2018, "
    #         "selain unsur masyarakat yang tersebut dalam ayat (2), musyawarah desa dapat "
    #         "melibatkan unsur masyarakat lain sesuai dengan kondisi sosial budaya masyarakat."
    #     ),
    #     "session_id": 104,
    # },
    # # ── KELOMPOK 2: Perdes Cigentur No. 8 Tahun 2018 - BPD ──
    # {
    #     "no": 5,
    #     "message": (
    #         "Bagaimana mekanisme penetapan anggota Badan Permusyawaratan Desa (BPD) "
    #         "di Desa Cigentur berdasarkan Pasal 2 Peraturan Desa Cigentur Nomor 8 "
    #         "Tahun 2018?"
    #     ),
    #     "ground_truth": (
    #         "Berdasarkan Pasal 2 Peraturan Desa Cigentur Nomor 8 Tahun 2018, BPD di "
    #         "Desa Cigentur berkedudukan sebagai lembaga yang melaksanakan fungsi "
    #         "pemerintahan desa yang anggotanya merupakan wakil dari penduduk desa "
    #         "berdasarkan keterwakilan wilayah dan ditetapkan secara demokratis."
    #     ),
    #     "session_id": 105,
    # },
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
    print(f"  [RAGAS EVAL]  {total_cases} Pertanyaan  |  {total_models} Model")
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
    json_path = os.path.join(EVALUATE_DIR, f"ragas_eval_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n[SAVED] JSON  -> {json_path}")

    # ─── SIMPAN CSV ──────────────────────────────────────────────────────────────
    csv_path  = os.path.join(EVALUATE_DIR, f"ragas_eval_{timestamp}.csv")
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
    print("  RINGKASAN HASIL EVALUASI RAGAS")
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
    print(f"     - ragas_eval_{timestamp}.json")
    print(f"     - ragas_eval_{timestamp}.csv")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_evaluation()
