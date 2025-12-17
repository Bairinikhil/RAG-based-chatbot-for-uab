"""
Sentence Transformers Embedding Service
Free, local, unlimited embeddings using sentence-transformers
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """
    Embedding service using sentence-transformers
    - Completely free
    - Runs locally (no API calls)
    - No rate limits
    - Good quality embeddings
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the embedding model

        Args:
            model_name: Model to use. Options:
                - "all-mpnet-base-v2" (768 dims, best quality, recommended)
                - "all-MiniLM-L6-v2" (384 dims, faster, smaller)
                - "all-MiniLM-L12-v2" (384 dims, good balance)
        """
        self.model_name = model_name
        self.dimension = 768 if "mpnet" in model_name else 384

        logger.info(f"Loading sentence-transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded {model_name} (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Convert to list and return
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embeddings
        """
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100  # Show progress for large batches
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension

    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model is not None


# Global instance (singleton pattern)
_embedding_service: Optional[SentenceTransformerEmbeddings] = None


def get_embedding_service(model_name: str = "all-mpnet-base-v2") -> SentenceTransformerEmbeddings:
    """
    Get or create the global embedding service instance

    Args:
        model_name: Model to use (only used if creating new instance)

    Returns:
        SentenceTransformerEmbeddings instance
    """
    global _embedding_service

    if _embedding_service is None:
        logger.info("Initializing sentence-transformer embedding service")
        _embedding_service = SentenceTransformerEmbeddings(model_name)

    return _embedding_service
