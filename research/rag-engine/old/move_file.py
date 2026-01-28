import shutil
from pathlib import Path

def move_pdfs():
    source_dir = Path(r"D:\Workspace\Repository\thesis\research\rag-engine\all_pages")
    dest_dir = Path(r"D:\Workspace\Repository\thesis\research\rag-engine\corpus")
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(source_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to move.")
    
    for pdf_file in pdf_files:
        try:
            shutil.move(str(pdf_file), str(dest_dir / pdf_file.name))
            print(f"Moved: {pdf_file.name}")
        except Exception as e:
            print(f"Error moving {pdf_file.name}: {e}")

if __name__ == "__main__":
    move_pdfs()
