# src/utils/generator_gemini.py

import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Dict

load_dotenv()  # loads .env variables

class GeminiGenerator:
    """
    Generates answers using Gemini 2.5 Flash with AI Studio API key.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"Gemini model initialized: {model_name}")

    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks[:5]])
        prompt = f"""
            You are a knowledgeable assistant. 
            Use only the context below to answer the question.
            If the answer is not found, say "The context does not contain that information."

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
        return prompt.strip()

    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)
        response = self.model.generate_content(prompt)
            
            # Strip whitespace from the generated text
        answer = response.text.strip()
            
            # ADDED LOGIC: If the generated answer is empty, provide the explicit fallback.
        if not answer:
                # This handles cases where the model returns an empty string 
                # (e.g., due to safety filtering) instead of the instructed fallback.
            return "The context does not contain that information."

        return answer