"""
Ollama LLM Service
Free, local LLM using Ollama
"""

import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    LLM service using Ollama
    - Completely free
    - Runs locally (no API calls)
    - No rate limits
    - Good quality generation
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: int = 60
    ):
        """
        Initialize Ollama LLM service

        Args:
            base_url: Ollama server URL
            model: Model to use (e.g., "llama3.2:3b", "mistral", "llama3.1:8b")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

        logger.info(f"Initializing Ollama LLM with model: {model}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text using Ollama

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            # Add system prompt if provided
            if system:
                payload["system"] = system

            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            # Extract the generated text
            generated_text = result.get("response", "")

            logger.info(f"Generated {len(generated_text)} characters")
            return generated_text

        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.base_url}")
            raise ConnectionError(
                f"Ollama is not running. Please start Ollama:\n"
                f"  1. Install from https://ollama.ai\n"
                f"  2. Run: ollama pull {self.model}\n"
                f"  3. Ollama will start automatically"
            )
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise TimeoutError(f"Ollama generation timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the model is available

        Returns:
            True if available, False otherwise
        """
        try:
            # Try to list models
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            models_data = response.json()
            available_models = [m["name"] for m in models_data.get("models", [])]

            if self.model in available_models:
                logger.info(f"Ollama model '{self.model}' is available")
                return True
            else:
                logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Available models: {', '.join(available_models) if available_models else 'none'}"
                )
                return False

        except requests.exceptions.ConnectionError:
            logger.warning(f"Ollama is not running at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current model

        Returns:
            Model info dictionary or None if not available
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Could not get model info: {e}")
            return None


# Global instance (singleton pattern)
_ollama_service: Optional[OllamaLLM] = None


def get_ollama_service(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b"
) -> OllamaLLM:
    """
    Get or create the global Ollama service instance

    Args:
        base_url: Ollama server URL (only used if creating new instance)
        model: Model to use (only used if creating new instance)

    Returns:
        OllamaLLM instance
    """
    global _ollama_service

    if _ollama_service is None:
        logger.info("Initializing Ollama LLM service")
        _ollama_service = OllamaLLM(base_url, model)

    return _ollama_service
