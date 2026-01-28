import os
from pathlib import Path
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
# filepath: d:\Workspace\Repository\thesis\research\rag-engine\extract_images.py
PDF_DIR = r"D:\Workspace\Repository\thesis\research\rag-engine\corpus"
OUTPUT_DIR = r"D:\Workspace\Repository\thesis\research\rag-engine\data\knowlege_base"
MAX_WORKERS = 8 

def process_pdf(pdf_path):
    try:
        pdf_name = pdf_path.stem
        save_dir = Path(OUTPUT_DIR) / pdf_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Get total page count
        try:
            info = pdfinfo_from_path(str(pdf_path))
            total_pages = int(info["Pages"])
        except Exception:
            total_pages = -1

        # 2. Check existing images
        existing_images = list(save_dir.glob("page_*.jpg"))
        
        # 3. Graceful Skip
        if len(existing_images) == total_pages and total_pages > 0:
             return 

        # 4. Convert
        # thread_count=1 is important when using ProcessPoolExecutor to avoid contention
        images = convert_from_path(str(pdf_path), thread_count=1)
        
        for i, image in enumerate(images):
            image_path = save_dir / f"page_{i+1}.jpg"
            image.save(image_path, "JPEG", quality=80)
            
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")

if __name__ == "__main__":
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs. Starting extraction with {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks first
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        
        # Process them as they complete (out of order)
        for future in tqdm(as_completed(futures), total=len(pdf_files)):
            try:
                future.result()
            except Exception as e:
                # This catches errors that crashed the worker process itself
                pdf = futures[future]
                print(f"Worker crashed on {pdf.name}: {e}")
        
    print(f"Extraction complete! Images saved to {OUTPUT_DIR}")