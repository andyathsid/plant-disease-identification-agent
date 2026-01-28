import json
import os

def filter_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    filtered_data = []
    removed_count = 0
    
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            pests = data.get('pests_and_diseases', '')
            
            # Check if pests_and_diseases is empty (string, list, or "NA")
            is_empty = False
            if isinstance(pests, str):
                p_clean = pests.strip().lower()
                if not p_clean or p_clean == "na" or p_clean == "none":
                    is_empty = True
            elif isinstance(pests, list):
                if len(pests) == 0:
                    is_empty = True
            
            if not is_empty:
                filtered_data.append(data)
            else:
                removed_count += 1

    # Overwrite with filtered data
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Done. Kept {len(filtered_data)} items, removed {removed_count} items.")

if __name__ == "__main__":
    files_to_filter = [
        "llm/data/gardenology.jsonl",
        "llm/data/plantvillage.jsonl"
    ]
    
    for file in files_to_filter:
        filter_jsonl(file)
