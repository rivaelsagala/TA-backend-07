import json
from pathlib import Path

INPUT_FILE = Path(r"E:\MATA KULIAH\SEMESTER 8\TA\Code\BE\RAG\data\dataset\raft_dataset_final.jsonl")
ORACLE_FILE = INPUT_FILE.parent /"raft_dataset_final/" "raft_dataset_oracle.jsonl"
DISTRACTOR_FILE = INPUT_FILE.parent / "raft_dataset_final/" "raft_dataset_distractor.jsonl"

def split_dataset():
    oracle_count = 0
    distractor_count = 0
    
    if not INPUT_FILE.exists():
        print(f"File {INPUT_FILE} tidak ditemukan.")
        return
        
    print(f"Membaca dataset dari {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(ORACLE_FILE, "w", encoding="utf-8") as oracle_out, \
         open(DISTRACTOR_FILE, "w", encoding="utf-8") as distractor_out:
         
        for line in infile:
            if not line.strip():
                continue
            sample = json.loads(line)
            is_oracle_present = sample.get("metadata_extra", {}).get("oracle_present", False)
            
            if is_oracle_present:
                oracle_out.write(line)
                oracle_count += 1
            else:
                distractor_out.write(line)
                distractor_count += 1
                
    print(f"\nSelesai! Berhasil memisahkan dataset menjadi 2 file:")
    print(f"1. Data Oracle (oracle_present=true): {oracle_count} baris tersimpan di {ORACLE_FILE}")
    print(f"2. Data Distraktor (oracle_present=false): {distractor_count} baris tersimpan di {DISTRACTOR_FILE}")

if __name__ == "__main__":
    split_dataset()
