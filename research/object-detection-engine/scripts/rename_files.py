from pathlib import Path
import hashlib

def generate_short_name(original_name: str, index: int) -> str:
    """Generate a short, unique filename using hash and index"""
    hash_obj = hashlib.md5(original_name.encode())
    hash_str = hash_obj.hexdigest()[:8]
    return f"img_{index:06d}_{hash_str}"

def rename_dataset_files(dataset_root: Path, max_name_length: int = 50):
    """Rename all images and labels to shorter names"""
    
    splits = ["train", "valid", "test"]
    rename_map = {}
    
    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        
        if not images_dir.exists():
            print(f"Skipping {split}: images directory not found")
            continue
        
        image_files = sorted(images_dir.glob("*"))
        print(f"\nProcessing {split} split: {len(image_files)} files")
        
        for idx, img_path in enumerate(image_files, 1):
            if len(img_path.name) <= max_name_length:
                continue
            
            original_name = img_path.stem
            extension = img_path.suffix
            
            new_name = generate_short_name(original_name, idx)
            new_img_name = f"{new_name}{extension}"
            new_label_name = f"{new_name}.txt"
            
            new_img_path = images_dir / new_img_name
            old_label_path = labels_dir / f"{original_name}.txt"
            new_label_path = labels_dir / new_label_name
            
            img_path.rename(new_img_path)
            print(f"Renamed: {img_path.name} -> {new_img_name}")
            
            if old_label_path.exists():
                old_label_path.rename(new_label_path)
                print(f"Renamed: {old_label_path.name} -> {new_label_name}")
            
            rename_map[original_name] = new_name
    
    return rename_map

if __name__ == "__main__":
    DATA_ROOT = Path(r"D:\Workspace\Repository\thesis\research\object-detection-engine\data\PlantDoc\txt")
    
    print(f"Starting file renaming in: {DATA_ROOT}")
    rename_map = rename_dataset_files(DATA_ROOT, max_name_length=50)
    print(f"\nRenaming complete. Total files renamed: {len(rename_map)}")