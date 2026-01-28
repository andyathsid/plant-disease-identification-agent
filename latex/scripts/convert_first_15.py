from pdf2image import convert_from_path
import os
from pathlib import Path

def pdf_to_images_limit(pdf_path, max_pages=15, output_folder=None, dpi=300, fmt='png'):
    """
    Convert PDF pages to images up to a limit.
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    # Set output folder
    if output_folder is None:
        output_folder = pdf_path.parent / f"{pdf_path.stem}_images"
    else:
        output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {pdf_path.name} (first {max_pages} pages) to images...")
    print(f"Output folder: {output_folder}")
    
    try:
        # Convert PDF to images with limit
        # Using first_page and last_page parameters
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
        
        # Save each page as an image
        for i, image in enumerate(images, start=1):
            image_path = output_folder / f"page_{i:03d}.{fmt}"
            image.save(image_path, fmt.upper())
            print(f"Saved: {image_path.name}")
        
        print(f"\nSuccessfully converted {len(images)} pages!")
        print(f"Images saved to: {output_folder}")
        
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("\nNote: pdf2image requires poppler-utils to be installed.")

if __name__ == "__main__":
    # PDF file name
    pdf_file = "IMPLEMENTASI FUSI MODEL PADA APLIKASI CHATBOT UNTUK DETEKSI PENYAKIT DAUN TANAMAN ANGGUR BERBASIS.pdf"
    
    # Run conversion
    pdf_to_images_limit(
        pdf_path=pdf_file,
        max_pages=15
    )
