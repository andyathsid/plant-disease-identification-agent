import os
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

# Load configuration
load_dotenv()

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = "thesis-bucket"

# Local configuration
DATA_DIR = Path(r"d:\Workspace\Repository\thesis\research\rag-engine\data")
IMAGES_DIR = DATA_DIR / "knowlege_base"

# Initialize R2 Session
session = boto3.Session(
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY
)

def upload_file(local_path: Path):
    """Uploads a single file to R2 using a thread-local client."""
    try:
        # Create a new client instance for this thread (safer than sharing)
        client = session.client(
            service_name="s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            region_name="auto",
        )
        
        # Structure: knowlege_base/pdf_name/page_n.jpg
        relative_path = local_path.relative_to(IMAGES_DIR.parent)
        r2_key = relative_path.as_posix() # Ensure forward slashes for R2
        
        client.upload_file(
            str(local_path), 
            R2_BUCKET, 
            r2_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        return True
    except Exception as e:
        return False

def main():
    if not IMAGES_DIR.exists():
        print(f"Error: Directory {IMAGES_DIR} not found.")
        return

    # 1. Collect all images
    print(f"Scanning {IMAGES_DIR} for images...")
    image_files = list(IMAGES_DIR.rglob("*.jpg"))
    print(f"Found {len(image_files)} images to upload.")

    # 2. Upload in parallel
    # Using 8 workers as requested (safe for 16-core CPU)
    max_workers = 8 
    
    print(f"Starting upload to R2 bucket '{R2_BUCKET}' with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(upload_file, image_files), total=len(image_files), desc="Uploading"))

    print("\nUpload process finished.")

if __name__ == "__main__":
    main()
