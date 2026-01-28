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

    def __init__(self, model_name: str = "gemini-2.0-flash"):
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
        
        try:
            # 4. Error Handling: Prevents the pipeline from crashing if the API fails
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Check if response was blocked by safety filters
            if not response.parts:
                logger.warning("Gemini response was blocked by safety filters.")
                return "I cannot answer this question based on the safety guidelines."

            answer = response.text.strip()
            
            if not answer:
                return "The context does not contain that information."
                
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your request. Please try again."