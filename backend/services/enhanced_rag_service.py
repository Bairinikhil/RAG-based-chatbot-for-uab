"""
Fallback RAG service used when pgvector is unavailable.
Currently provides a minimal, retrieval-free answer via Gemini so that the
backend can operate without a local vector DB. You can later extend this to
use a local store (e.g., Chroma) if desired.
"""

from typing import Optional, Dict
import os
import time
import re

import google.generativeai as genai


class EnhancedRAGService:
    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.5-pro",
    ) -> None:
        self.chroma_path = chroma_path
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for EnhancedRAGService")

        genai.configure(api_key=self.api_key)
        
        # Ensure model name has the proper prefix
        if not model_name.startswith(('models/', 'tunedModels/')):
            model_name = f"models/{model_name}"
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def _parse_retry_delay(self, error_msg: str) -> float:
        """Extract retry delay from Gemini API error message."""
        # Look for "Please retry in X.XXXXs" pattern
        match = re.search(r'Please retry in ([\d.]+)s', error_msg)
        if match:
            return float(match.group(1))
        # Look for retry_delay.seconds in error
        match = re.search(r'retry_delay.*?seconds[:\s]+(\d+)', error_msg)
        if match:
            return float(match.group(1))
        # Default exponential backoff
        return 2.0

    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content with retry logic for rate limiting.
        Returns the generated text or raises the last exception.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                return "I couldn't generate a response."
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Calculate wait time
                        wait_time = self._parse_retry_delay(error_str)
                        # Add some jitter and exponential backoff
                        wait_time = wait_time * (1.5 ** attempt) + 0.5
                        print(f"[RATE LIMIT] Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed, raise with helpful message
                        raise Exception(
                            f"Rate limit exceeded. Please wait a moment and try again. "
                            f"Free tier allows 2 requests per minute. "
                            f"Error: {error_str}"
                        )
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        
        # If we get here, all retries failed
        raise last_error

    def generate_enhanced_response(
        self, question: str, student_context: Optional[Dict] = None
    ) -> str:
        if not question or not question.strip():
            return ""

        system_preamble = (
            "You are the UAB Programs & Fees assistant. Be concise, accurate, and helpful.\n"
            "If specific program/fee context is not provided, answer generally and suggest how the user can clarify."
        )

        context_lines = []
        if student_context:
            for key, value in student_context.items():
                context_lines.append(f"{key}: {value}")

        context_block = "\n".join(context_lines) if context_lines else "(no extra context)"

        prompt = (
            f"{system_preamble}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )

        try:
            return self._generate_with_retry(prompt, max_retries=3)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                return (
                    f"⚠️ Rate limit reached. The free tier allows 2 requests per minute. "
                    f"Please wait a moment before asking another question. "
                    f"Error: {error_str}"
                )
            else:
                return f"⚠️ Could not generate response. Error: {error_str}"


