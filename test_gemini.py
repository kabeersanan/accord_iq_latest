import os
from utils.generator_gemini import GeminiGenerator

def test_rag_generation():
    print("🔵 1. Initializing Gemini Generator...")
    
    # Check if API Key exists before starting
    if not os.getenv("GEMINI_API_KEY") and not os.path.exists(".env"):
        print("❌ Error: .env file not found or empty.")
        return

    try:
        generator = GeminiGenerator()
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return

    # 2. Simulate "Retrieved" Data (Mocking the Vector Store)
    # This represents what your PDF parser + FAISS would usually output
    dummy_chunks = [
        {"text": "Project AccordIQ is a sophisticated RAG pipeline designed by Kabeer Sanan."},
        {"text": "It uses Google's Gemini 2.0 Flash model for generation."},
        {"text": "The system is built using Python, FastAPI, and FAISS for vector search."}
    ]

    query = "What model does AccordIQ use?"

    print(f"\n🔵 2. Testing Query: '{query}'")
    print("-" * 30)

    # 3. Generate Answer
    answer = generator.generate_answer(query, dummy_chunks)

    print(f"🤖 Gemini Answer:\n{answer}")
    print("-" * 30)

    # 4. Test Safety/Hallucination (Query not in context)
    fake_query = "What is the capital of France?"
    print(f"\n🔵 3. Testing Out-of-Context Query: '{fake_query}'")
    answer_fake = generator.generate_answer(fake_query, dummy_chunks)
    print(f"🤖 Gemini Answer (Should be negative):\n{answer_fake}")

if __name__ == "__main__":
    test_rag_generation()