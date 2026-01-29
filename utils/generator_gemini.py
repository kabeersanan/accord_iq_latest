# src/utils/generator_gemini.py

import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Dict
import logging

# Set up logging to track errors without crashing the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiGenerator:
    """
    Generates answers using Gemini with RAG-specific optimizations:
    - Low temperature for factual consistency.
    - System instructions for strict guardrails.
    - Error handling for API interruptions.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        genai.configure(api_key=api_key)
        
        # 1. Configuration for RAG: Low temperature = Less creativity, more facts
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

        # 2. System Instruction: Defines the persona globally (Available in newer API versions)
        self.system_instruction = """
        You are a precise and helpful assistant.
        1. Answer the user's question explicitly using ONLY the provided Context.
        2. If the answer is not in the text, DO NOT make it up. State: "The context does not contain that information."
        3. Maintain a professional tone.
        """

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction
        )
        
        logger.info(f"Gemini model initialized: {model_name}")

    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        # 3. Enhanced Context Formatting: Clearly separates data from the query
        context_text = "\n---\n".join([chunk["text"] for chunk in retrieved_chunks[:5]])
        
        prompt = f"""
        Context Information:
        {context_text}

        User Question:
        {query}
        """
        return prompt.strip()

    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)
        
        # Debug: See exactly what text is being sent to Gemini
        print(f"--- DEBUG: CONTEXT SENT TO GEMINI ---\n{prompt[:1000]}...\n---")

        try:
            # Lower safety thresholds for technical/legal RAG
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.parts:
                # Check the "finish_reason" to see WHY it was blocked
                reason = getattr(response, 'candidates', [None])[0].finish_reason if response.candidates else "Unknown"
                logger.warning(f"⚠️ Gemini response was blocked. Reason: {reason}")
                return f"Error: The AI response was blocked by safety filters (Reason: {reason})."

            return response.text.strip()

        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"I encountered a technical error: {str(e)}"