# src/utils/embedder.py

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class Embedder:
    """
    FAISS-Ready Embedder.
    Automatically detects model dimensions to prevent FAISS shape errors.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        """
        Args:
            device: 'cuda', 'mps', or 'cpu'.
        """
        # 1. Hardware Optimization
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps" 
            else:
                device = "cpu"
        
        print(f"Loading embedding model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.model_name = model_name
        
        # 2. Dynamic Dimension Capture (CRITICAL FOR FAISS)
        # We ask the model: "How big are your vectors?"
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension detected: {self.dimension}")

    def get_dimension(self) -> int:
        """Returns the vector size (e.g., 768 or 1024) for FAISS initialization."""
        return self.dimension

    def create_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Embeds document chunks. 
        normalize_embeddings=True is ESSENTIAL for FAISS IndexFlatIP (Cosine Similarity).
        """
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Encoding {len(texts)} chunks (Batch: {batch_size})...")
        
        # FAISS expects numpy float32. 
        vectors = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True # Ensures dot product = cosine similarity
        )

        # Attach embeddings to chunks
        # We keep them as standard Lists for safety, FAISS will convert to float32 numpy later
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = vectors[i].tolist() 
        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds the user query.
        Adds specific instructions for BGE models to improve retrieval accuracy.
        """
        # BGE models work best when you tell them it's a search query
        instruction = ""
        if "bge" in self.model_name:
             instruction = "Represent this sentence for searching relevant passages: "
        elif "nomic" in self.model_name:
             instruction = "search_query: " # Nomic specific prefix
        
        full_query = instruction + query
        
        # Output must be a List[float] for your FAISS search method
        vec = self.model.encode(
            [full_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )[0]
        
        return vec.tolist()