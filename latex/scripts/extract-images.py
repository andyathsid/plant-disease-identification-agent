from pdf2image import convert_from_path
import os
from pathlib import Path


def pdf_to_images(pdf_path, output_folder=None, dpi=300, fmt='png'):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save images (default: same as PDF)
        dpi: Image resolution (default: 300)
        fmt: Image format - 'png', 'jpg', etc. (default: 'png')
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
    
    print(f"Converting {pdf_path.name} to images...")
    print(f"Output folder: {output_folder}")
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
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
        print("Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
        print("Add the bin folder to your PATH or specify poppler_path parameter.")


if __name__ == "__main__":
    # Path to your PDF
    pdf_file = "Lampiran 17 - Template Proposal Skripsi (1).pdf"
    
    # Convert to images
    pdf_to_images(
        pdf_path=pdf_file,
        output_folder=None,  # Will create folder based on PDF name
        dpi=300,  # High quality
        fmt='png'  # PNG format
    )
