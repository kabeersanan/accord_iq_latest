# utils/vectorstore_faiss.py
import os
import json
import numpy as np
import uuid

class FaissVectorStore:
    def __init__(self, embedding_fn, dim=384, metric="cosine"):
        """
        embedding_fn: A function that takes [str] and returns [list of floats]
        """
        import faiss
        self.embedding_fn = embedding_fn  # <--- LINK 1: Store the embedder
        self.dim = dim
        self.metric = metric.lower()
        self._normalize = self.metric == "cosine"
        self.index = faiss.IndexFlatIP(dim) if self._normalize else faiss.IndexFlatL2(dim)
        self.id_map = {}
        self._next_id = 0
        print(f"FAISS index initialized with dimension {dim}")

    def _maybe_normalize(self, arr: np.ndarray):
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr.astype("float32")

    # --- NEW METHOD: This links your Chunker to FAISS ---
    def add_texts(self, texts, metadatas=None):
        """
        Takes a list of strings (from your chunker), embeds them, and stores them.
        """
        if not texts: return
        
        # 1. Generate Embeddings internally
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedding_fn(texts) # Returns list of lists
        
        # 2. Format data for internal storage
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        chunks_formatted = []
        for i, (text, vec) in enumerate(zip(texts, embeddings)):
            chunks_formatted.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "embedding": vec,
                "metadata": metadatas[i]
            })
            
        # 3. Pass to the existing low-level method
        self.add_embeddings(chunks_formatted)

    def add_embeddings(self, chunks):
        """Low-level add: Expects pre-computed embedding dicts."""
        if not chunks: return
        vecs = []
        for c in chunks:
            emb = np.array(c["embedding"], dtype="float32")
            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding dim mismatch: {emb.shape[0]} != {self.dim}")
            vecs.append(emb)
            self.id_map[str(self._next_id)] = {
                "id": c["id"],
                "text": c["text"],
                "metadata": c.get("metadata", {})
            }
            self._next_id += 1

        mat = np.vstack(vecs)
        mat = self._maybe_normalize(mat)
        self.index.add(mat)
        print(f"Added {len(vecs)} vectors. Total now: {self.index.ntotal}")

    # --- UPDATED METHOD: Search by text directly ---
    def search(self, query, top_k=5):
        """
        query: Can be a string (text) OR a list of floats (vector)
        """
        # 1. Handle Text Query (Auto-embed)
        if isinstance(query, str):
            query_vec = self.embedding_fn([query])[0]
        else:
            query_vec = query

        # 2. Standard Search Logic
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec, dtype="float32")
        q = query_vec.reshape(1, -1)
        q = self._maybe_normalize(q)

        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            doc = self.id_map.get(str(idx))
            if not doc: continue
            score = float(dist) if self._normalize else float(1.0 / (1.0 + dist))
            results.append({
                "text": doc["text"],
                "score": score,
                "metadata": doc["metadata"]
            })
        return results