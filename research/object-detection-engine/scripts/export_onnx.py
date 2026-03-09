from ultralytics import YOLO
import os
import io
import numpy as np
import requests
from pathlib import Path

# Optional imports with fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not available. ONNX inference testing will be skipped.")
    ONNX_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: cv2 not available. Using PIL for image processing.")
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not available. Image processing may be limited.")
    PIL_AVAILABLE = False

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

# Base project directory (thesis). Use parents to make paths robust when script
# is run from any working directory.
BASE_DIR = Path(__file__).resolve().parents[3]

def preprocess_image_for_onnx(image_path, input_size=(640, 640)):
    """Preprocess image for ONNX inference"""
    if not (CV2_AVAILABLE and PIL_AVAILABLE):
        print("Required image processing libraries not available")
        return None

    if isinstance(image_path, str) and image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(image_path))

    # Expected input_size is (H, W) or (W, H); ensure tuple (W, H) for cv2
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        # make width,height for cv2.resize
        target_size = (int(input_size[1]), int(input_size[0])) if input_size[0] != input_size[1] and (input_size[0] > 0 and input_size[1] > 0) else (int(input_size[1]), int(input_size[0]))
        # fall back to (640,640) if weird
        try:
            img_resized = cv2.resize(img, (target_size[0], target_size[1]))
        except Exception:
            img_resized = cv2.resize(img, (640, 640))
    else:
        img_resized = cv2.resize(img, (640, 640))

    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Add batch dimension and transpose to CHW format
    img_input = np.transpose(img_normalized, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    return img_input

def validate_onnx_file(onnx_path):
    """Validate that an ONNX file is properly formed"""
    if not onnx_path.exists():
        return False, "File does not exist"
    
    if onnx_path.stat().st_size == 0:
        return False, "File is empty"
    
    if not ONNX_AVAILABLE:
        return True, "onnxruntime not available for validation"
    
    try:
        # Try to load the model
        session = ort.InferenceSession(str(onnx_path))
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()
        
        if len(output_meta) == 0:
            return False, "No outputs found in model"
            
        return True, "Valid ONNX file"
    except Exception as e:
        return False, f"Invalid ONNX file: {str(e)}"

def test_onnx_inference(onnx_path, test_image_url="https://thesis-assets.andyathsid.com/plantwild/gallery/images/eggplant%20leaf/1.jpg"):
    """Test ONNX model inference with onnxruntime"""
    if not ONNX_AVAILABLE:
        print("onnxruntime not available, skipping ONNX inference test")
        return None
        
    # First validate the file
    is_valid, message = validate_onnx_file(onnx_path)
    if not is_valid:
        print(f"✗ ONNX validation failed: {message}")
        return None
    else:
        print(f"✓ ONNX validation passed: {message}")
        
    try:
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input details
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape  # e.g. [1,3,640,640] or [None,3,None,None]

        # Determine input H,W from input_shape (fallback to 640)
        try:
            # assume CHW
            h = 640
            w = 640
            if len(input_shape) >= 4:
                # input_shape may be [batch, channels, height, width]
                h = input_shape[2] if input_shape[2] is not None else 640
                w = input_shape[3] if input_shape[3] is not None else 640
            elif len(input_shape) == 3:
                # channels, height, width
                h = input_shape[1] if input_shape[1] is not None else 640
                w = input_shape[2] if input_shape[2] is not None else 640
            h = int(h)
            w = int(w)
        except Exception:
            h, w = 640, 640

        # Preprocess image using the model's input size
        img_input = preprocess_image_for_onnx(test_image_url, input_size=(h, w))
        if img_input is None:
            return None
        
        # Run inference
        outputs = session.run(None, {input_name: img_input})
        
        print(f"✓ ONNX inference successful for {onnx_path.name}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shapes: {[out.shape for out in outputs]}")
        
        return outputs
        
    except Exception as e:
        print(f"✗ ONNX inference failed for {onnx_path.name}: {e}")
        return None

def compare_outputs(output1, output2, tolerance=1e-3):
    """Compare outputs between simplified and non-simplified models"""
    if output1 is None or output2 is None:
        return False
    
    max_diff = 0.0
    for i, (o1, o2) in enumerate(zip(output1, output2)):
        diff = np.abs(o1 - o2)
        max_diff_tensor = np.max(diff)
        max_diff = max(max_diff, max_diff_tensor)
        mean_diff = np.mean(diff)
        
        print(f"  Output {i}: max_diff={max_diff_tensor:.6f}, mean_diff={mean_diff:.6f}")
    
    within_tolerance = max_diff < tolerance
    print(f"  Overall max difference: {max_diff:.6f} (tolerance: {tolerance})")
    print(f"  Within tolerance: {'✓' if within_tolerance else '✗'}")
    
    return within_tolerance
def export_rf_detr_models():
    if not RFDETR_AVAILABLE:
        print("rfdetr library not available, skipping RF-DETR export")
        return
        
    MODELS_DIR = BASE_DIR / "research" / "object-detection-engine" / "models" / "rfdetr" / "models"
    
    rf_variants = {
        'nano': MODELS_DIR / 'train_nano' / 'checkpoint_best_total.pth',
        'small': MODELS_DIR / 'train_small' / 'checkpoint_best_total.pth',
        'medium': MODELS_DIR / 'train_medium' / 'checkpoint_best_total.pth',
    }

    # RF-DETR input sizes per variant (H, W)
    rf_input_sizes = {
        'nano': (384, 384),
        'small': (512, 512),
        'medium': (576, 576),
    }
    
    for variant, model_path in rf_variants.items():
        if not model_path.exists():
            print(f"Warning: Model file '{model_path}' not found, skipping...")
            continue

        print(f"Loading RF-DETR {variant} model from: {model_path}")    
        model = RFDETRNano(pretrain_weights=str(model_path)) if variant == 'nano' else \
                RFDETRSmall(pretrain_weights=str(model_path)) if variant == 'small' else \
                RFDETRMedium(pretrain_weights=str(model_path))
        
        # Move model to CPU explicitly
        model.model.cpu()
                
        out_dir = MODELS_DIR / f"rf_detr_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_size = rf_input_sizes.get(variant, None)
            # Attempt to pass input_size if supported by rfdetr export
            if input_size is not None:
                try:
                    model.export(output_dir=str(out_dir), simplify=True, shape=input_size, device='cpu')
                except TypeError:
                    # Fallback if device parameter not supported
                    try:
                        model.export(output_dir=str(out_dir), simplify=True, device='cpu')
                    except TypeError:
                        model.export(output_dir=str(out_dir), simplify=True)
            else:
                try:
                    model.export(output_dir=str(out_dir), simplify=True, device='cpu')
                except TypeError:
                    model.export(output_dir=str(out_dir), simplify=True)
            print(f"✓ Exported RF-DETR {variant} to: {out_dir}")
            
            # Test ONNX inference
            onnx_files = list(out_dir.glob("*.onnx"))
            for onnx_file in onnx_files:
                print(f"Testing ONNX inference for {onnx_file.name}:")
                test_onnx_inference(onnx_file)
                
        except Exception as e:
            print(f"Error exporting RF-DETR {variant}: {e}")

    

def export_yolo_models():
    """Export all YOLO trained models to ONNX format with both simplified and non-simplified versions"""
    
    # Define model variants and their paths to trained weights
    models = {
        'yolo11n': 'train_yolov11n',
        'yolo11s': 'train_yolov11s', 
        'yolo11m': 'train_yolov11m'
    }
    model_dir = BASE_DIR / 'research' / 'object-detection-engine' / 'models' / 'yolov11'

    # Ensure model directory exists
    if not model_dir.exists():
        print(f"Error: Model directory '{model_dir}' not found")
        return

    for model_name, train_dir in models.items():
        # Path to the trained model weights
        model_path = model_dir / 'runs' / 'detect' / train_dir / 'weights' / 'best.pt'

        if not model_path.exists():
            print(f"Warning: Trained model file '{model_path}' not found, skipping...")
            continue
            
        try:
            print(f"Loading trained model: {model_path}")
            # Load the model and explicitly set to CPU
            model = YOLO(str(model_path))
            model.to('cpu')

            base_name = model_name
            
            # Create separate directories for each version
            non_simplified_dir = model_dir / "non_simplified"
            simplified_dir = model_dir / "simplified"
            
            non_simplified_dir.mkdir(parents=True, exist_ok=True)
            simplified_dir.mkdir(parents=True, exist_ok=True)
            
            # Export non-simplified version to its own directory
            print(f"Exporting {model_name} to non-simplified version...")
            model.export(format="onnx", simplify=False, device="cpu", imgsz=(416, 416), dynamic=False)
            
            # Move the exported file to the non-simplified directory
            default_onnx = model_path.with_suffix('.onnx')
            non_simplified_path = non_simplified_dir / f"{base_name}.onnx"
            
            if default_onnx.exists():
                import shutil
                shutil.move(str(default_onnx), str(non_simplified_path))
                print(f"✓ Successfully exported to: {non_simplified_path}")
            else:
                print(f"✗ Warning: Expected ONNX file not found after export: {default_onnx}")

            # Export simplified version to its own directory
            print(f"Exporting {model_name} to simplified version...")
            model.export(format="onnx", simplify=True, device="cpu", imgsz=(416, 416), dynamic=False)
            
            # Move the exported file to the simplified directory
            default_onnx = model_path.with_suffix('.onnx')
            simplified_path = simplified_dir / f"{base_name}.onnx"
            
            if default_onnx.exists():
                import shutil
                shutil.move(str(default_onnx), str(simplified_path))
                print(f"✓ Successfully exported to: {simplified_path}")
            else:
                print(f"✗ Warning: Expected ONNX file not found after export: {default_onnx}")

            # Test both versions with onnxruntime
            outputs_original = None
            outputs_simplified = None
            
            # Test non-simplified version
            if non_simplified_path.exists():
                print(f"✓ Successfully exported: {non_simplified_path.name}")
                # Validate the ONNX file
                is_valid, message = validate_onnx_file(non_simplified_path)
                if is_valid:
                    print(f"Testing ONNX inference for non-simplified version:")
                    outputs_original = test_onnx_inference(non_simplified_path)
                else:
                    print(f"✗ Error: {non_simplified_path.name} - {message}")
            else:
                print(f"✗ Failed to export: {non_simplified_path.name}")
                
            # Test simplified version
            if simplified_path.exists():
                print(f"✓ Successfully exported: {simplified_path.name}")
                # Validate the ONNX file
                is_valid, message = validate_onnx_file(simplified_path)
                if is_valid:
                    print(f"Testing ONNX inference for simplified version:")
                    outputs_simplified = test_onnx_inference(simplified_path)
                else:
                    print(f"✗ Error: {simplified_path.name} - {message}")
            else:
                print(f"✗ Failed to export: {simplified_path.name}")
            
            # Compare outputs
            if outputs_original is not None and outputs_simplified is not None:
                print(f"Comparing outputs between non-simplified and simplified versions:")
                compare_outputs(outputs_original, outputs_simplified)
            
            # Also test with YOLO wrapper for verification (test non-simplified version first)
            if non_simplified_path.exists() and non_simplified_path.stat().st_size > 0:
                print(f"Testing with YOLO wrapper for non-simplified version")
                try:
                    onnx_model = YOLO(str(non_simplified_path), task="detect")
                    test_image = "https://thesis-assets.andyathsid.com/plantwild/gallery/images/eggplant%20leaf/1.jpg"
                    results = onnx_model(test_image, verbose=False, device="cpu")
                    print(f"✓ YOLO wrapper inference test completed for non-simplified version")
                except Exception as e:
                    print(f"✗ YOLO wrapper test failed for non-simplified version: {e}")
            elif simplified_path.exists() and simplified_path.stat().st_size > 0:
                print(f"Testing with YOLO wrapper for simplified version")
                try:
                    onnx_model = YOLO(str(simplified_path), task="detect")
                    test_image = "https://thesis-assets.andyathsid.com/plantwild/gallery/images/eggplant%20leaf/1.jpg"
                    results = onnx_model(test_image, verbose=False, device="cpu")
                    print(f"✓ YOLO wrapper inference test completed for simplified version")
                except Exception as e:
                    print(f"✗ YOLO wrapper test failed for simplified version: {e}")
            else:
                print(f"Skipping YOLO wrapper test - no valid ONNX files available")
                
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
        
        # Clean up any leftover default .onnx files
        default_onnx = model_path.with_suffix('.onnx')
        if default_onnx.exists():
            try:
                default_onnx.unlink()
                print(f"Cleaned up leftover file: {default_onnx.name}")
            except Exception as e:
                print(f"Warning: Could not clean up {default_onnx.name}: {e}")
        
        print("-" * 50)

def export_summary():
    """Generate a summary of the export process"""
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    
    # Check YOLOv12 exports
    yolo_dir = BASE_DIR / 'research' / 'object-detection-engine' / 'models' / 'yolov11'
    
    if yolo_dir.exists():
        print("\nYOLOv11 Models:")
        
        # Check non-simplified versions
        non_simplified_dir = yolo_dir / "non_simplified"
        if non_simplified_dir.exists():
            print("  Non-simplified versions:")
            for model_file in non_simplified_dir.glob("*.onnx"):
                print(f"    {model_file.name} ({'✓' if model_file.exists() else '✗'})")
        
        # Check simplified versions
        simplified_dir = yolo_dir / "simplified"
        if simplified_dir.exists():
            print("  Simplified versions:")
            for model_file in simplified_dir.glob("*.onnx"):
                print(f"    {model_file.name} ({'✓' if model_file.exists() else '✗'})")
    
    # Check RF-DETR exports
    rfdetr_dir = BASE_DIR / "research" / "object-detection-engine" / "models" / "rfdetr" / "models"
    if rfdetr_dir.exists():
        print("\nRF-DETR Models:")
        for variant_dir in rfdetr_dir.glob("rf_detr_*"):
            if variant_dir.is_dir():
                onnx_files = list(variant_dir.glob("*.onnx"))
                if onnx_files:
                    print(f"  {variant_dir.name}:")
                    for model_file in onnx_files:
                        print(f"    {model_file.name} ({'✓' if model_file.exists() else '✗'})")
    
    print("\n" + "="*60)
    print("Export process completed!")
    print("="*60)

if __name__ == "__main__":
    # Run both exporters
    # print("Starting RF-DETR export...")
    # export_rf_detr_models()
    
    # print("\nStarting YOLO export...")
    export_yolo_models()
    
    # Generate summary
    export_summary()