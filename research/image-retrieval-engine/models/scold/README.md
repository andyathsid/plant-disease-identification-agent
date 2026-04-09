---
license: cc-by-4.0
language:
- en
metrics:
- accuracy
- recall
pipeline_tag: image-to-text
tags:
- agriculture
- leaf
- disease
datasets:
- enalis/LeafNet
library_name: transformers
---

# 🌿 SCOLD: A Vision-Language Foundation Model for Leaf Disease Identification

**SCOLD** is a multimodal model that maps **images** and **text descriptions** into a shared embedding space. This model is developed for **cross-modal retrieval**, **few-shot classification**, and **explainable AI in agriculture**, especially for plant disease diagnosis from both images and domain-specific text prompts.

---

### ✅ Intended Use
- Vision-language embedding for classification or retrieval tasks
- Few-shot learning in agricultural or medical datasets
- Multimodal interpretability or zero-shot transfer
---

## 🧪 How to Use

First clone our repository:

```bash
 git clone https://huggingface.co/enalis/scold
```

Please find detail to load and use our model in *inference.py*

```python

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text = "A maize leaf with bacterial blight"
inputs = tokenizer(text, return_tensors="pt")

# Image preprocessing
image = Image.open("path_to_leaf.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    image_emb, text_emb = model(image_tensor, inputs["input_ids"], inputs["attention_mask"])
    similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb)
    print(f"Similarity score: {similarity.item():.4f}")
```
Please cite this paper if this code is useful for you!

```
@article{quoc2025vision,
  title={A Vision-Language Foundation Model for Leaf Disease Identification},
  author={Quoc, Khang Nguyen and Thu, Lan Le Thi and Quach, Luyl-Da},
  journal={arXiv preprint arXiv:2505.07019},
  year={2025}
}

```