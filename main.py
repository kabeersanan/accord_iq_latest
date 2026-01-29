import os
import uuid
import json
import faiss  # Added top-level import to prevent runtime errors
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Utils imports
from utils.extractor import extract_text_from_pdf
# FIXED: Updated import to match the function name in utils/chunker.py
from utils.chunker import recursive_chunk_text 
from utils.embedder import Embedder
from utils.vectorstore_faiss import FaissVectorStore
from utils.generator_gemini import GeminiGenerator

# Load environment variables
load_dotenv()

# Directory setup
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(os.path.dirname(__file__), "indexes"))
os.makedirs(INDEX_DIR, exist_ok=True)

# Config paths
METADATA_PATH = os.path.join(INDEX_DIR, "id_map.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# FastAPI init
app = FastAPI(title="RAG PDF Chat")

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# INITIALIZATION (Fixed Order & Logic)
# ---------------------------------------------------------
print("Initializing RAG components...")

# 1. Initialize Embedder first to get the correct dimension
embedder = Embedder()
EMBED_DIM = embedder.get_dimension() # FIXED: Dynamically gets 1024 for BGE-M3 (was hardcoded to 384)

# 2. Initialize Generator
generator = GeminiGenerator()

# 3. Initialize Vector Store with the embedding function
# FIXED: Passed 'embedding_fn' which is required by your updated vectorstore_faiss.py
vector_store = FaissVectorStore(embedding_fn=embedder.embed_query, dim=EMBED_DIM)

# Load persisted FAISS index and metadata
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            vector_store.id_map = json.load(f)
        if os.path.exists(FAISS_INDEX_PATH):
            vector_store.index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ Loaded FAISS index and metadata from disk")
    except Exception as e:
        print(f"⚠️ Failed to load persisted index: {e}")


def persist_index_and_map():
    """Save FAISS index and id_map to disk."""
    try:
        faiss.write_index(vector_store.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(vector_store.id_map, f, ensure_ascii=False, indent=2)
        print("✅ Persisted FAISS index and metadata")
    except Exception as e:
        print(f"❌ Error persisting index: {e}")


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

    try:
        # Extract text per page
        pages_text = extract_text_from_pdf(tmp_path)
        
        # Chunk the text
        full_text = " ".join(pages_text)
        
        # FIXED: Calling the correct function name 'recursive_chunk_text'
        chunks_texts = recursive_chunk_text(full_text)

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
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        
    finally:
        # Cleanup: Remove the temp PDF file to save space
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------------
# 💬 Ask Question Endpoint
# -------------------------------
class AskRequest(BaseModel):
    question: str
    file_id: Optional[str] = None
    top_k: Optional[int] = 3

@app.post("/ask")
async def ask(req: AskRequest):
    question = req.question
    file_id = req.file_id
    top_k = req.top_k or 3

    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        # 1. Generate the query embedding
        q_vec = embedder.embed_query(question)
        
        # 2. Search for more chunks than needed (to account for filtering)
        # We search for at least 15 chunks so that if we filter by file_id, 
        # we still have enough relevant ones left.
        search_k = max(15, top_k * 2) 
        results = vector_store.search(q_vec, top_k=search_k)
        
        print(f"[DEBUG] Initial vector search found {len(results)} total chunks.")

        # 3. Handle File-Specific Filtering
        if file_id:
            # DEBUG: See what IDs are actually being compared
            print(f"[DEBUG] Filtering results for target file_id: '{file_id}'")
            if results:
                sample_id = results[0].get("metadata", {}).get("file_id")
                print(f"[DEBUG] Metadata ID in first search result: '{sample_id}'")

            # Perform the filter
            results = [r for r in results if r.get("metadata", {}).get("file_id") == file_id]
            
            # If no chunks match the specific file, inform the user
            if not results:
                return JSONResponse({
                    "answer": "The requested document does not appear to contain relevant information for this question.",
                    "retrieved": [],
                    "note": f"Search found matches in other files, but 0 matches for ID: {file_id}"
                }, status_code=200)

        # 4. Final slice to the user's requested top_k
        final_results = results[:top_k]
        print(f"[DEBUG] Sending {len(final_results)} chunks to Gemini for answer generation.")

        # 5. Generate answer with Gemini
        answer_text = generator.generate_answer(question, final_results)

        return {
            "answer": answer_text, 
            "retrieved": final_results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error during RAG processing: {str(e)}"
        )

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
            "embed_dim": EMBED_DIM
        }
        try:
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