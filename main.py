import os
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory setup
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(os.path.dirname(__file__), "indexes"))
os.makedirs(INDEX_DIR, exist_ok=True)

# Config paths
METADATA_PATH = os.path.join(INDEX_DIR, "id_map.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMBED_DIM = 384  # must match your embedding model

# FastAPI init
app = FastAPI(title="RAG PDF Chat")

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utils imports
from utils.extractor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.embedder import Embedder
from utils.vectorstore_faiss import FaissVectorStore
from utils.generator_gemini import GeminiGenerator

# Initialize core components
embedder = Embedder()
generator = GeminiGenerator()
vector_store = FaissVectorStore(dim=EMBED_DIM)

# Load persisted FAISS index and metadata
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            vector_store.id_map = json.load(f)
        if os.path.exists(FAISS_INDEX_PATH):
            import faiss
            vector_store.index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ Loaded FAISS index and metadata from disk")
    except Exception as e:
        print("⚠️ Failed to load persisted index:", e)


def persist_index_and_map():
    """Save FAISS index and id_map to disk."""
    try:
        import faiss
        faiss.write_index(vector_store.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(vector_store.id_map, f, ensure_ascii=False, indent=2)
        print("✅ Persisted FAISS index and metadata")
    except Exception as e:
        print("❌ Error persisting index:", e)


# -------------------------------
# 📄 PDF Upload Endpoint
# -------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, chunk, embed, and add to FAISS."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Generate unique file_id
    file_id = str(uuid.uuid4())
    tmp_path = os.path.join(INDEX_DIR, f"{file_id}.pdf")

    # Save uploaded file temporarily
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text per page
    pages_text = extract_text_from_pdf(tmp_path)
    pages = [{"page": i + 1, "text": p} for i, p in enumerate(pages_text)]

    # Chunk the text
    full_text = " ".join(pages_text)
    chunks_texts = chunk_text(full_text)

    chunks = []
    for i, t in enumerate(chunks_texts):
        cid = f"{file_id}_c{i}"
        chunks.append({
            "id": cid,
            "text": t,
            "metadata": {"file_id": file_id, "chunk_index": i}
        })

    # Create embeddings and add to FAISS
    chunks = embedder.create_embeddings(chunks)
    vector_store.add_embeddings(chunks)
    persist_index_and_map()

    return JSONResponse({
        "file_id": file_id,
        "num_pages": len(pages_text),
        "num_chunks": len(chunks_texts)
    })


# -------------------------------
# 💬 Ask Question Endpoint
# -------------------------------
class AskRequest(BaseModel):
    question: str
    file_id: Optional[str] = None
    top_k: Optional[int] = 3


""" @app.post("/ask")
async def ask(req: AskRequest):
    question = req.question
    file_id = req.file_id
    top_k = req.top_k or 3

    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # Embed the question
    q_vec = embedder.embed_query(question)
    print("DEBUG q_vec_len =", len(q_vec))

    # --- Normalize query for cosine/IP FAISS ---
    import numpy as np
    q_arr = np.array(q_vec, dtype=np.float32).reshape(1, -1)
    norm = float(np.linalg.norm(q_arr))
    if norm != 0.0:
        q_arr = q_arr / norm
    q_arr_np = q_arr.flatten().astype('float32')

    print(f"[DEBUG] normalized q_vec len={len(q_arr_np)} (norm was {norm:.4f})")

    # Perform search
    debug_top_k = max(10, top_k)
    results = vector_store.search(q_arr_np, top_k=debug_top_k)
    for r in results[:3]:
        print("[DEBUG TEXT SAMPLE]", r["text"][:300].replace("\n", " "), "...\n")

    print(f"[DEBUG] search returned {len(results)} results (top_k={debug_top_k})")

    # Optional: restrict by file_id
    if file_id:
        results = [r for r in results if r.get("metadata", {}).get("file_id") == file_id]
        if not results:
            return JSONResponse({
                "answer": None,
                "retrieved": [],
                "note": "no chunks for that file_id"
            }, status_code=200)

    # Generate answer with Gemini
    answer_text = generator.generate_answer(question, results)
    print(f"[DEBUG] Final answer length: {len(answer_text)}")

    return {"answer": answer_text, "retrieved": results}
 """
@app.post("/ask")
async def ask(req: AskRequest):
    question = req.question
    file_id = req.file_id
    top_k = req.top_k or 3

    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try: # <--- Start of new try block
        # Embed the question
        q_vec = embedder.embed_query(question)
        print("DEBUG q_vec_len =", len(q_vec))

        # --- Normalize query for cosine/IP FAISS ---
        import numpy as np
        q_arr = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q_arr))
        if norm != 0.0:
            q_arr = q_arr / norm
        q_arr_np = q_arr.flatten().astype('float32')

        print(f"[DEBUG] normalized q_vec len={len(q_arr_np)} (norm was {norm:.4f})")

        # Perform search
        debug_top_k = max(10, top_k)
        results = vector_store.search(q_arr_np, top_k=debug_top_k)
        print(f"[DEBUG] search returned {len(results)} results (top_k={debug_top_k})")

        # Optional: restrict by file_id
        if file_id:
            results = [r for r in results if r.get("metadata", {}).get("file_id") == file_id]
            if not results:
                return JSONResponse({
                    "answer": "The requested file ID has no indexed content to answer the question.",
                    "retrieved": [],
                    "note": "no chunks for that file_id"
                }, status_code=200)

        # Generate answer with Gemini
        answer_text = generator.generate_answer(question, results)

        return {"answer": answer_text, "retrieved": results}

    except Exception as e: # <--- Catch block
        # This will print the exception trace to your terminal for debugging
        import traceback
        traceback.print_exc()
        
        # This returns an HTTP 500 response to the client with the specific error
        raise HTTPException(status_code=500, detail=f"Internal Server Error during RAG processing: {str(e)}. Check backend console for full trace.")


# -------------------------------
# 🧠 Debug Info Endpoint
# -------------------------------
@app.get("/debug/index_info")
def debug_index_info():
    try:
        idx = getattr(vector_store, "index", None)
        id_map = getattr(vector_store, "id_map", {})
        info = {
            "index_exists": idx is not None,
            "id_map_size": len(id_map),
        }
        try:
            import faiss
            info["ntotal"] = int(idx.ntotal) if idx is not None else 0
        except Exception:
            info["ntotal"] = None
        info["sample_ids"] = list(id_map.keys())[:10]
        return info
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG PDF Chat backend is running"}

