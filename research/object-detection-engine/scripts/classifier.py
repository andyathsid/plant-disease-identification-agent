import asyncio
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
from typing import List, Dict, Union, Optional, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import RobertaTokenizer
import aiohttp
import pandas as pd
from tqdm.auto import tqdm

class AsyncImageHandler:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """Load image from base64 string, URL, or bytes"""
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif image_input.startswith("http"):
            return await self._load_from_url(image_input)
        elif image_input.startswith("data:image"):
            return self._load_from_base64(image_input)
        else:
            # Assume file path
            return Image.open(image_input).convert("RGB")
    
    async def _load_from_url(self, url: str) -> Image.Image:
        async with self.session.get(url) as response:
            response.raise_for_status()
            image_bytes = await response.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    def _load_from_base64(self, base64_str: str) -> Image.Image:
        # Extract base64 data from data URL
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

class SCOLDClassifier:
    def __init__(self, 
                 model_path: str,
                 tokenizer_name: str = "roberta-base",
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "leaf_disease_collection"):
        
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Load ONNX model
        self.session = None
        self.tokenizer = None
        self.qdrant_client = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and tokenizer"""
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant at {self.qdrant_url}: {e}")
            self.qdrant_client = None
    
    def setup_collection(self):
        """Setup Qdrant collection with text and image vector configs"""
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized. Check connection.")
        
        # Delete existing collection if it exists
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        
        # Create collection with both text and image vectors
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
            }
        )
        print(f"Created collection: {self.collection_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using ONNX model"""
        # Preprocess text
        text_inputs = self._preprocess_text(text)
        
        # Run inference with dummy image input
        outputs = self.session.run(None, {
            'image_input': np.zeros((1, 3, 224, 224), dtype=np.float32),  # Dummy image
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        })
        
        text_emb = outputs[1]  # Text embeddings are the second output
        
        # Normalize
        text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
        return text_emb[0]
    
    def encode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Encode image from bytes using ONNX model"""
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self._preprocess_image(image)
        
        # Run inference with dummy text input
        outputs = self.session.run(None, {
            'image_input': image_tensor,
            'input_ids': np.zeros((1, 77), dtype=np.int64),  # Dummy text
            'attention_mask': np.zeros((1, 77), dtype=np.int64)  # Dummy attention
        })
        
        image_emb = outputs[0]  # Image embeddings are the first output
        
        # Normalize
        image_emb = image_emb / np.linalg.norm(image_emb, axis=-1, keepdims=True)
        return image_emb[0]
    
    def ingest_support_set(self,
                          data: Union[pd.DataFrame, str],
                          batch_size: int = 10,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Ingest support set data into Qdrant collection
        
        Args:
            data: DataFrame with columns ['image', 'caption', 'label'] or path to parquet file
            batch_size: Batch size for insertion
            progress_callback: Optional callback function for progress updates
        
        Returns:
            Dictionary with ingestion statistics
        """
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized. Check connection.")
        
        # Load data if path is provided
        if isinstance(data, str):
            if data.endswith('.parquet'):
                df = pd.read_parquet(data)
            else:
                raise ValueError("Only parquet files are supported for path input")
        else:
            df = data.copy()
        
        # Validate required columns
        required_columns = ['image', 'caption', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Setup collection
        self.setup_collection()
        
        # Ingest data in batches
        total_points = 0
        successful_batches = 0
        failed_batches = 0
        
        for i in tqdm(range(0, len(df), batch_size), desc="Ingesting batches"):
            batch_df = df.iloc[i:i+batch_size]
            batch_points = []
            
            for idx, row in batch_df.iterrows():
                try:
                    caption_text = str(row['caption'])
                    
                    # Encode text
                    text_vec = self.encode_text(caption_text)
                    
                    # Encode image
                    img_vec = self.encode_image_from_bytes(row['image']['bytes'])
                    
                    # Create point with both vector types
                    batch_points.append(models.PointStruct(
                        id=idx,
                        vector={
                            "text": text_vec.tolist(),
                            "image": img_vec.tolist(),
                        },
                        payload={
                            "caption": caption_text,
                            "label": row['label'],
                            "source_id": idx,
                            **{k: v for k, v in row.items() if k not in ['image', 'caption', 'label']}
                        }
                    ))
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
            
            # Insert batch into Qdrant
            if batch_points:
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    successful_batches += 1
                    total_points += len(batch_points)
                    
                    if progress_callback:
                        progress_callback(i + len(batch_points), len(df))
                        
                except Exception as e:
                    failed_batches += 1
                    print(f"Batch {i//batch_size + 1} error: {e}")
                    continue
        
        # Return statistics
        stats = {
            "total_points": total_points,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_batches": successful_batches + failed_batches,
            "collection_name": self.collection_name
        }
        
        print(f"Ingestion completed: {total_points} points inserted")
        return stats
    
    def cross_modal_search(self,
                          query_text: Optional[str] = None,
                          image_input: Optional[Union[str, bytes]] = None,
                          limit: int = 5,
                          search_type: str = "text") -> List[dict]:
        """
        Perform cross-modal search using Qdrant
        
        Args:
            query_text: Text query for text-to-image search
            image_input: Image path, bytes, or URL for image-to-image search
            limit: Number of results to return
            search_type: "text" or "image" search type
        
        Returns:
            List of search results with scores and payloads
        """
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized. Check connection.")
        
        if search_type == "text":
            if query_text is None:
                raise ValueError("query_text is required for text search")
            
            # Encode text query
            query_vector = self.encode_text(query_text)
            
            # Search using text vector
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                using="text",
                limit=limit,
                with_payload=True
            )
            
        elif search_type == "image":
            if image_input is None:
                raise ValueError("image_input is required for image search")
            
            # Load image
            if isinstance(image_input, str):
                if image_input.startswith("http"):
                    # Download from URL
                    import requests
                    response = requests.get(image_input)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    # Read from file
                    with open(image_input, "rb") as f:
                        image_bytes = f.read()
            else:
                image_bytes = image_input
            
            # Encode image query
            query_vector = self.encode_image_from_bytes(image_bytes)
            
            # Search using image vector
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                using="image",
                limit=limit,
                with_payload=True
            )
            
        else:
            raise ValueError("search_type must be 'text' or 'image'")
        
        # Format results
        results = []
        for point in search_result.points:
            results.append({
                'id': point.id,
                'score': point.score,
                'payload': point.payload
            })
        
        return results
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        image_array = np.transpose(image_array, (2, 0, 1))
        
        image_tensor = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        return image_tensor
    
    def _preprocess_text(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocess text for ONNX model"""
        # Tokenize text
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        
        return {
            'input_ids': tokens['input_ids'].numpy(),
            'attention_mask': tokens['attention_mask'].numpy()
        }
    
    async def predict(self, 
                     image_input: Union[str, bytes],
                     candidate_boxes: Optional[List[dict]] = None,
                     query_text: Optional[str] = None,
                     top_k: int = 5,
                     method: str = "zero-shot") -> Dict:
        if method == "zero-shot" and query_text is None:
            raise ValueError("query_text is required for zero-shot classification")
        elif method == "few-shot" and query_text is not None:
            raise ValueError("query_text should not be provided for few-shot classification")
        elif method == "image-to-text" and query_text is not None:
            raise ValueError("query_text should not be provided for image-to-text classification")
        
        image = await self._load_image(image_input)
        
        if candidate_boxes is None:
            results = await self._process_full_image(image, query_text, top_k, method)
        else:
            results = await self._process_candidate_boxes(image, candidate_boxes, query_text, top_k, method)
        
        return results
    
    async def _process_full_image(self, image: Image.Image, 
                                 query_text: Optional[str],
                                 top_k: int,
                                 method: str) -> Dict:
        if method == "text-to-image":
            if query_text is None:
                raise ValueError("query_text is required text-to-image classification")
            
            text_embedding = await self._encode_text(query_text)
            
            if self.qdrant_client:
                search_results = await self._search_by_text(text_embedding, top_k)
            else:
                search_results = self._get_fallback_results(top_k)
        elif method == "image-to-text":
            if image is None:
                raise ValueError("image is required image-to-text classification")
            image_embedding = await self._encode_image(image)
            
            if self.qdrant_client:
                search_results = await self._search_image_against_text(image_embedding, top_k)
            else:
                search_results = self._get_fallback_results(top_k)
        else:
            image_embedding = await self._encode_image(image)
            
            if self.qdrant_client:
                search_results = await self._search_by_image(image_embedding, top_k)
            else:
                search_results = self._get_fallback_results(top_k)
        
        return self._format_results(search_results, method)
    
    async def _process_candidate_boxes(self, image: Image.Image,
                                     candidate_boxes: List[dict],
                                     query_text: Optional[str],
                                     top_k: int,
                                     method: str) -> Dict:
        """Process candidate boxes for classification"""
        results = []
        
        for box in candidate_boxes:
            # Crop image to box
            cropped_image = self._crop_image(image, box['box'])
            
            if method == "zero-shot":
                if query_text is None:
                    raise ValueError("query_text is required for zero-shot classification")
                
                # Get text embedding
                text_embedding = await self._encode_text(query_text)
                
                # Search in Qdrant
                if self.qdrant_client:
                    search_results = await self._search_by_text(text_embedding, top_k)
                else:
                    search_results = self._get_fallback_results(top_k)
            else:
                # Get image embedding
                image_embedding = await self._encode_image(cropped_image)
                
                # Search in Qdrant
                if self.qdrant_client:
                    search_results = await self._search_by_image(image_embedding, top_k)
                else:
                    search_results = self._get_fallback_results(top_k)
            
            box_result = {
                'box': box['box'],
                'score': box.get('score', 1.0),
                'classification': self._format_results(search_results, method)
            }
            results.append(box_result)
        
        return {'boxes': results}
    
    async def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image using ONNX model"""
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {
            'image_input': image_tensor,
            'input_ids': np.zeros((1, 77), dtype=np.int64),  # Dummy input for text
            'attention_mask': np.zeros((1, 77), dtype=np.int64)  # Dummy input for attention
        })
        
        image_embedding = outputs[0]
        
        # Normalize
        image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
        return image_embedding[0]
    
    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using ONNX model"""
        # Preprocess text
        text_inputs = self._preprocess_text(text)
        
        # Run inference
        outputs = self.session.run(None, {
            'image_input': np.zeros((1, 3, 224, 224), dtype=np.float32),  # Dummy input for image
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        })
        
        text_embedding = outputs[1]
        
        # Normalize
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=-1, keepdims=True)
        return text_embedding[0]
    
    async def _search_by_text(self, text_embedding: np.ndarray, top_k: int) -> List[dict]:
        """Search Qdrant by text embedding"""
        if not self.qdrant_client:
            return self._get_fallback_results(top_k)
        
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=text_embedding.tolist(),
                using="text",
                limit=top_k,
                with_payload=True
            )
            return [self._format_point(point) for point in search_result.points]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return self._get_fallback_results(top_k)
    
    async def _search_by_image(self, image_embedding: np.ndarray, top_k: int) -> List[dict]:
        if not self.qdrant_client:
            return self._get_fallback_results(top_k)
        
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=image_embedding.tolist(),
                using="image",
                limit=top_k,
                with_payload=True
            )
            return [self._format_point(point) for point in search_result.points]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return self._get_fallback_results(top_k)
    
    async def _search_image_against_text(self, image_embedding: np.ndarray, top_k: int) -> List[dict]:
        if not self.qdrant_client:
            return self._get_fallback_results(top_k)
        
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=image_embedding.tolist(),
                using="text",
                limit=top_k,
                with_payload=True
            )
            return [self._format_point(point) for point in search_result.points]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return self._get_fallback_results(top_k)
    
    def _format_point(self, point) -> dict:
        """Format Qdrant point result"""
        return {
            'id': point.id,
            'score': point.score,
            'payload': point.payload
        }
    
    def _format_results(self, search_results: List[dict], method: str) -> dict:
        """Format search results"""
        if not search_results:
            return {'label': 'unknown', 'confidence': 0.0, 'top_k': []}
        
        # Extract labels and scores
        labels = [result['payload']['label'] for result in search_results]
        scores = [result['score'] for result in search_results]
        
        # Weighted voting (sum of scores)
        label_votes = {}
        for label, score in zip(labels, scores):
            label_votes[label] = label_votes.get(label, 0) + score
        
        # Get best label
        best_label = max(label_votes.items(), key=lambda x: x[1])
        total_score = sum(label_votes.values())
        confidence = best_label[1] / total_score if total_score > 0 else 0
        
        return {
            'label': best_label[0],
            'confidence': float(confidence),
            'label_scores': label_votes,
            'top_k': list(zip(labels, scores))
        }
    
    def _crop_image(self, image: Image.Image, box: List[float]) -> Image.Image:
        """Crop image to bounding box"""
        x1, y1, x2, y2 = map(int, box)
        return image.crop((x1, y1, x2, y2))
    
    async def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """Async image loading"""
        async with AsyncImageHandler() as handler:
            return await handler.load_image(image_input)
    
    def _get_fallback_results(self, top_k: int) -> List[dict]:
        """Get fallback results when Qdrant is not available"""
        fallback_labels = [
            "healthy_leaf", "bacterial_spot", "leaf_mold", "target_spot", 
            "mosaic_virus", "septoria_leaf_spot", "spider_mites", "unknown"
        ]
        
        results = []
        for i in range(min(top_k, len(fallback_labels))):
            results.append({
                'id': i,
                'score': 1.0 / (i + 1), 
                'payload': {
                    'label': fallback_labels[i]
                }
            })
        
        return results

def progress_callback(current: int, total: int):
    """Progress callback for ingestion"""
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

