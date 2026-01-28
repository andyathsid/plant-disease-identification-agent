import torch
import stamina
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import os
import numpy as np

# Configuration
DATA_DIR = Path(r"d:\Workspace\Repository\thesis\research\rag-engine\data")
IMAGES_DIR = DATA_DIR / "knowlege_base"

R2_PUBLIC_DOMAIN = "https://thesis-assets.andyathsid.com"
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "colmodernvbert_collection"

MODEL_NAME = "ModernVBERT/colmodernvbert"
DEVICE = "cuda:0" 

# Print device name
if DEVICE.startswith("cuda"):
    print(f"Using device: {torch.cuda.get_device_name(int(DEVICE.split(':')[1]))}")

@stamina.retry(on=Exception, attempts=10, wait_initial=5, wait_max=60)
def setup_qdrant(client):
    """Setup collection with retries if Qdrant is restarting."""
    start_point_id = 0
    existing_pdfs = set()

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} exists. Resuming...")
        collection_info = client.get_collection(COLLECTION_NAME)
        start_point_id = collection_info.points_count
        
        # Fetch existing PDFs to skip
        offset = None
        while True:
            scroll_result, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in scroll_result:
                if point.payload and "pdf_name" in point.payload:
                    existing_pdfs.add(point.payload["pdf_name"])
            if offset is None:
                break
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "initial": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        ),
                    ),
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
    return start_point_id, existing_pdfs

def main():
    # 1. Load Model
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = ColModernVBert.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True
    ).eval()
    processor = ColModernVBertProcessor.from_pretrained(MODEL_NAME)
    
    # 2. Setup Qdrant
    client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
    start_point_id, existing_pdfs = setup_qdrant(client)
    print(f"Resuming from ID: {start_point_id}, Skipping {len(existing_pdfs)} PDFs.")

    # 4. Ingestion
    pdf_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    embedding_batch_size = 4 
    
    def generate_points():
        point_id = start_point_id
        for pdf_dir in tqdm(pdf_dirs, desc="Processing PDFs"):
            if pdf_dir.name in existing_pdfs:
                continue
            
            image_files = sorted(list(pdf_dir.glob("*.jpg")), key=lambda x: int(x.stem.split('_')[1]))
            if not image_files: continue

            images = [Image.open(img_path) for img_path in image_files]
            for i in range(0, len(images), embedding_batch_size):
                batch_images = images[i:i + embedding_batch_size]
                with torch.no_grad():
                    processed_images = processor.process_images(batch_images).to(model.device)
                    image_embeddings = model(**processed_images)
                
                vecs_initial = image_embeddings.cpu().float().numpy()
                for j in range(len(batch_images)):
                    page_num = i + j + 1
                    yield models.PointStruct(
                        id=point_id,
                        payload={"pdf_name": pdf_dir.name, "page": page_num, 
                                 "image_url": f"{R2_PUBLIC_DOMAIN}/knowlege_base/{pdf_dir.name}/page_{page_num}.jpg"},
                        vector={"initial": vecs_initial[j].tolist()}
                    )
                    point_id += 1

    # Wrap the upload and post-indexing in a retry block
    @stamina.retry(on=Exception, attempts=10, wait_initial=10, wait_max=120)
    def start_upload():
        print("Starting ingestion with parallel uploads...")
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=generate_points(),
            batch_size=8,
            parallel=2,
        )
        print("Re-enabling indexing...")
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )

    start_upload()
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Ingestion complete. Total points: {collection_info.points_count}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
