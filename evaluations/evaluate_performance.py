# python -m evaluations.evaluate_performance



import time
import pandas as pd
from app.services.rag_service import get_answer_from_rag, AVAILABLE_MODELS

def run_performance_benchmark():
    # Daftar ID model yang akan diuji (sesuai AVAILABLE_MODELS di rag_service.py)
    # Anda bisa menghapus ID model yang tidak digunakan.
    # 1: Llama-3, 2: Qwen, 3: DeepSeek, 5: GPT-4o-mini, 6: GPT-3.5, 7: Gemini
    test_models = [1, 2, 3]
    
    # 5 pertanyaan contoh (Anda bisa tambahkan lebih banyak atau load dari tes_case.txt)
    queries = [
        "Apa tugas pokok kepala desa?",
        "Bagaimana pengelolaan keuangan desa dilakukan?",
        "Sebutkan larangan bagi perangkat desa menurut peraturan!",
        "Bagaimana prosedur pengangkatan perangkat desa?",
        "Apa saja hak dari BPD?"
    ]
    
    results = []
    
    for model_id in test_models:
        # Skip jika ID tidak ada di AVAILABLE_MODELS
        if model_id not in AVAILABLE_MODELS:
            continue
            
        model_name = AVAILABLE_MODELS[model_id]["name"]
        print(f"\n" + "="*50)
        print(f"🚀 MENGUJI MODEL: {model_name}")
        print("="*50)
        
        total_retrieval = 0
        total_inference = 0
        total_latency = 0
        sukses_count = 0
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] Query: '{query}'")
            t0 = time.time()
            
            try:
                # Memanggil core pipeline backend secara langsung (tanpa overhead HTTP/Flask)
                response = get_answer_from_rag(query=query, model_id=model_id, chat_history=[])
                
                t1 = time.time()
                latency = t1 - t0
                
                # Mengambil metrics yang sudah kita tambahkan ke rag_service.py
                retrieval_time = response.get("retrieval_time") or 0
                inference_time = response.get("inference_time") or 0
                
                total_retrieval += retrieval_time
                total_inference += inference_time
                total_latency += latency
                sukses_count += 1
                
                print(f"     ✅ Sukses | Latency Total: {latency:.2f}s | Retrieval: {retrieval_time:.2f}s | Inferensi LLM: {inference_time:.2f}s")
                
            except Exception as e:
                print(f"     ❌ GAGAL: {e}")
                
        # Hitung nilai rata-rata (hanya dari yang sukses)
        if sukses_count > 0:
            avg_retrieval = total_retrieval / sukses_count
            avg_inference = total_inference / sukses_count
            avg_latency = total_latency / sukses_count
            
            results.append({
                "Model LLM": model_name,
                "Rata-rata Waktu Retrieval (s)": round(avg_retrieval, 2),
                "Rata-rata Waktu Inferensi (s)": round(avg_inference, 2),
                "Rata-rata Waktu Respons Total (s)": round(avg_latency, 2)
            })
        else:
            print(f"  ⚠️ Semua query gagal untuk model {model_name}.")
        
    # Simpan ke CSV untuk dimasukkan ke tabel Word
    if results:
        df = pd.DataFrame(results)
        csv_filename = "hasil_pengujian_kinerja.csv"
        df.to_csv(csv_filename, index=False)
        print("\n\n" + "="*50)
        print("✅ PENGUJIAN SELESAI")
        print(f"Data tersimpan di: {csv_filename}")
        print("="*50)
        print("\nData Siap Diisi ke Tabel Dokumen Word Anda:")
        print(df.to_markdown(index=False))
        print("="*50)

if __name__ == "__main__":
    run_performance_benchmark()
