"""
Ollama Service for local LLM answer generation
Provides local alternative to Gemini API
"""

import os
import logging
from typing import Optional, Tuple
import ollama

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for generating responses using local Ollama models"""

    def __init__(self, model_name: str = "llama3.2:3b", base_url: Optional[str] = None):
        """
        Initialize Ollama service

        Args:
            model_name: Name of the Ollama model to use (default: llama3.2:3b)
            base_url: Custom Ollama server URL (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.available = self._check_availability()

        if self.available:
            logger.info(f"OllamaService initialized with model: {self.model_name}")
        else:
            logger.warning("OllamaService initialized but Ollama is not available")

    def _check_availability(self) -> bool:
        """Check if Ollama is available and the model exists"""
        try:
            # List available models
            models_response = ollama.list()

            # Extract model names - handle ListResponse object
            model_names = []
            if hasattr(models_response, 'models'):
                # It's a ListResponse object
                for model in models_response.models:
                    # Get the model attribute (it's a Model object)
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Fallback for dict response
                for model in models_response['models']:
                    name = model.get('name') or model.get('model', '')
                    if name:
                        model_names.append(name)

            logger.debug(f"Available Ollama models: {model_names}")

            # Check if our model is available
            if self.model_name in model_names:
                logger.info(f"Model {self.model_name} is available")
                return True
            else:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                return False

        except Exception as e:
            logger.error(f"Failed to check Ollama availability: {e}")
            import traceback
            traceback.print_exc()
            return False

    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.available

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Tuple[bool, str, str]:
        """
        Generate response using Ollama

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Tuple of (success, response_text, error_message)
        """
        if not self.available:
            return False, "", "Ollama service not available"

        try:
            logger.debug(f"Generating response with {self.model_name}")

            # Prepare options
            options = {
                'temperature': temperature,
            }
            if max_tokens:
                options['num_predict'] = max_tokens

            # Generate response using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options
            )

            response_text = response.get('response', '').strip()

            if response_text:
                logger.info(f"Generated response ({len(response_text)} chars)")
                return True, response_text, ""
            else:
                logger.warning("Ollama returned empty response")
                return False, "", "Empty response from Ollama"

        except Exception as e:
            error_msg = f"Ollama generation failed: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg

    def generate_response_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Generate response using Ollama with streaming (for future use)

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)

        Yields:
            Response chunks as they are generated
        """
        if not self.available:
            logger.error("Ollama service not available")
            return

        try:
            options = {
                'temperature': temperature,
            }
            if max_tokens:
                options['num_predict'] = max_tokens

            # Stream response
            stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                stream=True
            )

            for chunk in stream:
                yield chunk.get('response', '')

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            return


def create_ollama_service(model_name: Optional[str] = None) -> OllamaService:
    """
    Factory function to create Ollama service

    Args:
        model_name: Optional model name override

    Returns:
        OllamaService instance
    """
    model = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
    return OllamaService(model_name=model)
