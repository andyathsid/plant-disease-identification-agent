# SCOLD Model to ONNX Conversion

This guide explains how to convert the SCOLD model to ONNX format for deployment.

## Files Created

1. **`convert_to_onnx.py`** - Main conversion script
2. **`requirements_conversion.txt`** - Required dependencies
3. **`scold_onnx_inference.py`** - Generated inference script (created automatically)

## Installation

First, install the required dependencies:

```bash
pip install -r requirements_conversion.txt
```

**Note**: The dependencies have been optimized to be lighter by removing torchvision and using PIL for image preprocessing instead.

## Usage

### Basic Conversion

Run the conversion script:

```bash
python convert_to_onnx.py
```

This will:
- Load the SCOLD model from `scold.pt`
- Convert it to ONNX format
- Save as `scold.onnx`
- Create an inference script `scold_onnx_inference.py`
- Verify the conversion by comparing outputs

### Custom Parameters

You can customize the conversion:

```python
from convert_to_onnx import convert_scold_to_onnx

# Convert with custom parameters
convert_scold_to_onnx(
    model_path="path/to/your/scold.pt",
    output_path="custom_scold.onnx",
    text_sample="Your sample text for dynamic axes"
)
```

## Output Files

- **`scold.onnx`** - The converted ONNX model
- **`scold_onnx_inference.py`** - Ready-to-use inference script

## ONNX Model Details

### Input Names
- `image_input`: Image tensor (batch_size, 3, 224, 224)
- `input_ids`: Tokenized text input_ids (batch_size, sequence_length)
- `attention_mask`: Attention mask for text (batch_size, sequence_length)

### Output Names
- `image_embeddings`: Image feature embeddings (batch_size, 512)
- `text_embeddings`: Text feature embeddings (batch_size, 512)

### Dynamic Axes
- `batch_size`: Variable batch size
- `sequence_length`: Variable sequence length for text inputs

## Inference Example

```python
from scold_onnx_inference import SCOLDONNX

# Load the ONNX model
model = SCOLDONNX("scold.onnx")

# Predict similarity
similarity = model.predict(
    image_path="path/to/image.jpg",
    text="A maize leaf with bacterial spot"
)

print(f"Similarity score: {similarity}")
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: If you encounter memory errors, try reducing the batch size in the conversion script.

2. **ONNX Version Compatibility**: The script uses ONNX opset version 14. If you need compatibility with older versions, modify the `opset_version` parameter.

3. **Missing Dependencies**: Ensure all packages in `requirements_conversion.txt` are installed.

### Verification

The script automatically verifies the conversion by:
- Running inference with both PyTorch and ONNX models
- Comparing the output embeddings
- Reporting any significant differences

## Performance Notes

- The ONNX model should provide similar performance to the original PyTorch model
- ONNX format allows for better deployment options (TensorRT, OpenVINO, etc.)
- Dynamic axes enable flexible batch sizes and sequence lengths