import pandas as pd
import json
import hashlib
from pathlib import Path

def generate_deterministic_id(content: str, metadata: dict) -> str:
    """Generate deterministic ID based on content and metadata."""
    # Combine key metadata fields with content for uniqueness
    source = metadata.get('source', '')
    plant = metadata.get('plant', '')
    doc_type = metadata.get('type', '')
    disease = metadata.get('disease', '')
    
    # Create a unique string
    unique_str = f"{source}|{plant}|{doc_type}|{disease}|{content}"
    
    # Hash it to get a deterministic UUID-like string
    hash_obj = hashlib.md5(unique_str.encode('utf-8'))
    return hash_obj.hexdigest()

def generate_corpus():
    corpus_rows = []

    # --- Process PlantVillage ---
    pv_path = Path("data/plantvillage.jsonl")
    if pv_path.exists():
        with open(pv_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                plant = data['plant']
                
                # 1. Propagation Point
                if data.get('propagation'):
                    content = data['propagation']
                    metadata = {
                        'plant': plant,
                        'type': 'cultivation/propagation',
                        'source': 'plantvillage'
                    }
                    corpus_rows.append({
                        'doc_id': generate_deterministic_id(content, metadata),
                        'contents': f"Cultivation and propagation of {plant}: {content}",
                        'metadata': metadata
                    })
                
                # 2. Disease Points
                for disease in data.get('pests_and_diseases', []):
                    disease_name = disease.get('name', 'Unknown Disease')
                    content = f"Symptoms: {disease.get('symptoms', '')}. Management: {disease.get('management', '')}"
                    metadata = {
                        'plant': plant,
                        'type': 'disease',
                        'category': disease.get('category', 'unknown'),
                        'disease': disease_name,
                        'source': 'plantvillage'
                    }
                    corpus_rows.append({
                        'doc_id': generate_deterministic_id(content, metadata),
                        'contents': f"Symptoms and management of {disease_name} affecting {plant}: {content}",
                        'metadata': metadata
                    })

    # --- Process Gardenology ---
    gar_path = Path("data/gardenology.jsonl")
    if gar_path.exists():
        with open(gar_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                plant = data['plant']
                
                # 1. Cultivation/Propagation Point
                cult = data.get('cultivation', '')
                prop = data.get('propagation', '')
                if cult or prop:
                    content = f"{cult} {prop}".strip()
                    metadata = {
                        'plant': plant,
                        'type': 'cultivation/propagation',
                        'source': 'gardenology'
                    }
                    corpus_rows.append({
                        'doc_id': generate_deterministic_id(content, metadata),
                        'contents': f"Cultivation and propagation of {plant}: {content}",
                        'metadata': metadata
                    })
                
                # 2. Disease Point (Generic)
                if data.get('pests_and_diseases'):
                    content = data['pests_and_diseases']
                    metadata = {
                        'plant': plant,
                        'type': 'disease',
                        'category': 'unknown',
                        'disease': 'General Pests and Diseases',
                        'source': 'gardenology'
                    }
                    corpus_rows.append({
                        'doc_id': generate_deterministic_id(content, metadata),
                        'contents': f"Symptoms and management of pests and diseases affecting {plant}: {content}",
                        'metadata': metadata
                    })

    # Save to Parquet
    df = pd.DataFrame(corpus_rows)
    df.to_parquet('data/corpus.parquet')
    print(f"✓ Created corpus with {len(df)} points.")
    return df

if __name__ == "__main__":
    generate_corpus()
