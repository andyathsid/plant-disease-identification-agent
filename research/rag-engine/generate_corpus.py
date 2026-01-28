import pandas as pd
import json
import hashlib
import re
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Document

load_dotenv()

def clean_disease_name(disease_name: str) -> str:
    """Clean and normalize disease names by removing underscores and improving formatting."""
    if not disease_name:
        return "Unknown Disease"

    # Remove underscores used for formatting (e.g., _Fusarium graminearum_)
    # This pattern matches underscores around text that should be italicized
    cleaned = re.sub(r'_([^_]+)_', r'\1', disease_name)

    # Remove any remaining underscores and replace with spaces
    cleaned = cleaned.replace('_', ' ')

    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Capitalize first letter of each word for proper nouns (keep common words lowercase)
    # This is a simple heuristic - we can refine it based on specific needs
    words = cleaned.split()
    if words:
        # Keep common disease-related words lowercase
        common_lower = {'and', 'or', 'of', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}
        cleaned_words = []
        for i, word in enumerate(words):
            if word.lower() in common_lower and i > 0:
                cleaned_words.append(word.lower())
            else:
                # Capitalize first letter, keep rest as is (preserves scientific names)
                cleaned_words.append(word[0].upper() + word[1:] if word else word)
        cleaned = ' '.join(cleaned_words)

    return cleaned

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

# Initialize the Semantic Splitter once
# breakpoint_percentile_threshold: higher means more chunks (more sensitive to changes)
embed_model = GoogleGenAIEmbedding(model="genmini-embedding-001")
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

def chunk_content(content: str) -> list[str]:
    """Split text based on semantic meaning changes."""
    if not content.strip():
        return []

    doc = Document(text=content)
    nodes = semantic_splitter.get_nodes_from_documents([doc])
    chunks = [node.get_content() for node in nodes]

    return chunks

def count_lines_in_file(file_path):
    """Count the number of lines in a file for progress bar"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def generate_corpus(batch_size=1000):
    """
    Generate corpus with batching to manage memory usage.
    Process data in batches and save intermediate results to avoid memory issues.
    """
    # Initialize variables for batch processing
    corpus_rows_batch = []
    total_processed = 0
    batch_count = 0

    # Process PlantVillage data
    pv_path = Path("data/plantvillage.jsonl")
    if pv_path.exists():
        total_lines = count_lines_in_file(pv_path)
        print(f"Processing PlantVillage data ({total_lines} records)...")

        with open(pv_path, 'r', encoding='utf-8') as f:
            # Create a tqdm progress bar for PlantVillage data
            pbar = tqdm(total=total_lines, desc="PlantVillage", unit="record")

            for idx, line in enumerate(f):
                data = json.loads(line)
                plant = data['plant']

                # 1. Propagation Point
                if data.get('propagation'):
                    full_content = data['propagation'].strip()
                    chunks = chunk_content(full_content)

                    for i, content in enumerate(chunks):
                        metadata = {
                            'plant': plant,
                            'type': 'cultivation/propagation',
                            'source': 'plantvillage',
                            'chunk': i
                        }

                        page_content = f"Cultivation and propagation of {plant} (Part {i+1})"
                        page_content += f"\n\n{content}"

                        corpus_rows_batch.append({
                            'doc_id': generate_deterministic_id(content, metadata),
                            'contents': page_content,
                            'metadata': metadata
                        })

                # 2. Disease Points
                for disease in data.get('pests_and_diseases', []):
                    disease_name = disease.get('name', 'Unknown Disease')
                    cleaned_disease_name = clean_disease_name(disease_name)

                    symptoms = disease.get('symptoms', '').strip()
                    management = disease.get('management', '').strip()

                    content_parts = []
                    if symptoms:
                        content_parts.append(f"Symptoms: {symptoms}")
                    if management:
                        content_parts.append(f"Management: {management}")

                    full_content = ". ".join(content_parts) if content_parts else ""
                    chunks = chunk_content(full_content)

                    for i, content in enumerate(chunks):
                        metadata = {
                            'plant': plant,
                            'type': 'disease',
                            'category': disease.get('category', 'unknown'),
                            'disease': cleaned_disease_name,
                            'source': 'plantvillage',
                            'chunk': i
                        }

                        suffix = f" (Part {i+1})" if len(chunks) > 1 else ""
                        page_content = f"Disease: {cleaned_disease_name} affecting {plant}{suffix}"
                        if content:
                            page_content += f"\n\n{content}"

                        corpus_rows_batch.append({
                            'doc_id': generate_deterministic_id(content, metadata),
                            'contents': page_content,
                            'metadata': metadata
                        })

                total_processed += 1
                pbar.update(1)

                # Check if we need to save a batch
                if len(corpus_rows_batch) >= batch_size:
                    # Save current batch to temporary file
                    batch_df = pd.DataFrame(corpus_rows_batch)
                    batch_filename = f'data/corpus_batch_{batch_count}.parquet'
                    batch_df.to_parquet(batch_filename)

                    # Clear the batch to free memory
                    corpus_rows_batch.clear()
                    batch_count += 1

            pbar.close()

    # Process Gardenology data
    gar_path = Path("data/gardenology.jsonl")
    if gar_path.exists():
        total_lines = count_lines_in_file(gar_path)
        print(f"Processing Gardenology data ({total_lines} records)...")

        with open(gar_path, 'r', encoding='utf-8') as f:
            # Create a tqdm progress bar for Gardenology data
            pbar = tqdm(total=total_lines, desc="Gardenology", unit="record")

            for idx, line in enumerate(f):
                data = json.loads(line)
                plant = data['plant']

                # 1. Cultivation/Propagation Point
                cult = data.get('cultivation', '')
                prop = data.get('propagation', '')
                if cult or prop:
                    full_content = f"{cult}\n\n{prop}".strip()
                    chunks = chunk_content(full_content)

                    for i, content in enumerate(chunks):
                        metadata = {
                            'plant': plant,
                            'type': 'cultivation/propagation',
                            'source': 'gardenology',
                            'chunk': i
                        }

                        page_content = f"Cultivation and propagation of {plant} (Part {i+1})"
                        page_content += f"\n\n{content}"

                        corpus_rows_batch.append({
                            'doc_id': generate_deterministic_id(content, metadata),
                            'contents': page_content,
                            'metadata': metadata
                        })

                # 2. Disease Point (Generic)
                if data.get('pests_and_diseases'):
                    full_content = data['pests_and_diseases'].strip()
                    disease_name = 'General Pests and Diseases'

                    chunks = chunk_content(full_content)
                    for i, content in enumerate(chunks):
                        metadata = {
                            'plant': plant,
                            'type': 'disease',
                            'category': 'unknown',
                            'disease': disease_name,
                            'source': 'gardenology',
                            'chunk': i
                        }

                        suffix = f" (Part {i+1})" if len(chunks) > 1 else ""
                        page_content = f"Disease: {disease_name} affecting {plant}{suffix}"
                        page_content += f"\n\n{content}"

                        corpus_rows_batch.append({
                            'doc_id': generate_deterministic_id(content, metadata),
                            'contents': page_content,
                            'metadata': metadata
                        })

                total_processed += 1
                pbar.update(1)

                # Check if we need to save a batch
                if len(corpus_rows_batch) >= batch_size:
                    # Save current batch to temporary file
                    batch_df = pd.DataFrame(corpus_rows_batch)
                    batch_filename = f'data/corpus_batch_{batch_count}.parquet'
                    batch_df.to_parquet(batch_filename)

                    # Clear the batch to free memory
                    corpus_rows_batch.clear()
                    batch_count += 1

            pbar.close()

    # Save any remaining items in the last batch
    if corpus_rows_batch:
        batch_df = pd.DataFrame(corpus_rows_batch)
        batch_filename = f'data/corpus_batch_{batch_count}.parquet'
        batch_df.to_parquet(batch_filename)
        batch_count += 1

    # Now combine all batches into the final corpus
    print(f"Combining {batch_count} batches into final corpus...")

    all_dataframes = []
    for i in range(batch_count):
        batch_filename = f'data/corpus_batch_{i}.parquet'
        if os.path.exists(batch_filename):
            df_batch = pd.read_parquet(batch_filename)
            all_dataframes.append(df_batch)

            # Optionally remove the batch file after loading to save disk space
            # os.remove(batch_filename)

    # Concatenate all dataframes
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Save final corpus
    final_df.to_parquet('data/corpus.parquet')
    print(f"✓ Created corpus with {len(final_df)} points.")

    # Optionally remove temporary batch files
    for i in range(batch_count):
        batch_filename = f'data/corpus_batch_{i}.parquet'
        if os.path.exists(batch_filename):
            os.remove(batch_filename)

    return final_df

if __name__ == "__main__":
    generate_corpus()
