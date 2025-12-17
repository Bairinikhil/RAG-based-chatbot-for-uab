"""
AI service for UAB Chat Bot
Handles AI interactions, RAG queries, and response generation
"""

import os
import logging
from typing import Dict, Optional, Tuple
from services.enhanced_rag_service import EnhancedRAGService
from services.rag_pgvector import PgVectorRAG
from database_models import Student

# Import Enhanced RAG V2
try:
    from services.enhanced_rag_v2 import create_enhanced_rag_service
    ENHANCED_RAG_V2_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Enhanced RAG V2 available")
except ImportError as e:
    ENHANCED_RAG_V2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced RAG V2 not available: {e}")

class AIService:
    """Service for handling AI operations"""

    def __init__(self, use_enhanced_v2: bool = True):
        """
        Initialize AI service with RAG system

        Args:
            use_enhanced_v2: Whether to use Enhanced RAG V2 (default: True)
        """
        self.rag_service = None
        self.use_enhanced_v2 = use_enhanced_v2 and ENHANCED_RAG_V2_AVAILABLE
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize the RAG service"""
        try:
            database_url = os.getenv('DATABASE_URL')
            use_sentence_transformers = os.getenv('USE_SENTENCE_TRANSFORMERS', 'true').lower() == 'true'

            logger.debug(f"DATABASE_URL present: {bool(database_url)}")
            logger.debug(f"USE_SENTENCE_TRANSFORMERS: {use_sentence_transformers}")
            if database_url:
                logger.debug(f"DATABASE_URL: {database_url[:50]}...")

            if not database_url:
                logger.warning("DATABASE_URL not found, RAG service disabled")
                return

            # Prefer Enhanced RAG V2 if available and database_url present
            if self.use_enhanced_v2 and database_url:
                try:
                    logger.info("Initializing Enhanced RAG V2...")
                    use_ollama = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
                    self.rag_service = create_enhanced_rag_service(
                        database_url,
                        use_ollama=use_ollama
                    )
                    generation_method = "Ollama" if use_ollama else "Gemini"
                    logger.info(f"AI Service with Enhanced RAG V2 initialized successfully ({generation_method})")
                    logger.info("Features: Reranking ✓, Hybrid Search ✓, Sentence Transformers ✓")
                    return
                except Exception as e:
                    logger.warning(f"Enhanced RAG V2 init failed, falling back to PgVector RAG: {e}")

            # Fallback to PgVector RAG if DATABASE_URL is present
            if database_url:
                try:
                    logger.info("Initializing PgVector RAG with Sentence Transformers...")
                    self.rag_service = PgVectorRAG(
                        database_url=database_url,
                        use_sentence_transformers=use_sentence_transformers
                    )
                    logger.info("AI Service with pgvector RAG initialized successfully")
                    return
                except Exception as e:
                    logger.error(f"PgVector RAG init failed: {e}")
                    self.rag_service = None

        except Exception as e:
            logger.error(f"Could not initialize AI Service: {e}", exc_info=True)
            self.rag_service = None
    
    def generate_response(
        self,
        question: str,
        student_context: Optional[Dict] = None,
        extracted_entities: Optional[Tuple] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True
    ) -> Tuple[bool, str, str]:
        """
        Generate AI response to user question

        Args:
            question: User's question
            student_context: Student information for personalization
            extracted_entities: Pre-extracted entities to avoid duplicate extraction
            use_hybrid: Use hybrid BM25+vector search (Enhanced V2 only)
            use_reranking: Use cross-encoder reranking (Enhanced V2 only)

        Returns:
            Tuple of (success, response, error_message)
        """
        try:
            if not question or not question.strip():
                logger.warning("Empty question provided")
                return False, "", "No question provided"

            if not self.rag_service:
                logger.error("RAG service not available")
                return False, "", "AI service not available"

            # Check if Enhanced RAG V2 is available
            if hasattr(self.rag_service, 'generate_enhanced_response_v2'):
                # Use Enhanced RAG V2 with all improvements
                logger.debug(f"Using Enhanced RAG V2 (hybrid={use_hybrid}, reranking={use_reranking})")
                response, metadata = self.rag_service.generate_enhanced_response_v2(
                    question,
                    student_context=student_context,
                    extracted_entities=extracted_entities,
                    use_hybrid=use_hybrid,
                    use_reranking=use_reranking
                )

                # Log performance metadata
                logger.info(
                    f"Enhanced V2 Response: {len(response)} chars, "
                    f"stage={metadata.get('retrieval_stage')}, "
                    f"chunks={metadata.get('chunks_used')}/{metadata.get('chunks_retrieved')}, "
                    f"tokens={metadata.get('tokens_used')}"
                )

                return True, response, ""

            else:
                # Fallback to standard RAG service
                logger.debug("Using standard RAG service")
                if extracted_entities:
                    logger.debug("Passing pre-extracted entities to RAG service")
                    response = self.rag_service.generate_enhanced_response(
                        question, student_context, extracted_entities=extracted_entities
                    )
                else:
                    response = self.rag_service.generate_enhanced_response(question, student_context)

                if response:
                    logger.info(f"Generated response for question: {question[:50]}")
                    return True, response, ""
                else:
                    logger.warning("RAG service returned empty response")
                    return False, "", "Failed to generate response"

        except Exception as e:
            logger.error(f"AI service error: {e}", exc_info=True)
            return False, "", f"AI service error: {str(e)}"
    
    def is_available(self) -> bool:
        """
        Check if AI service is available
        
        Returns:
            Boolean indicating service availability
        """
        return self.rag_service is not None
    
    def get_service_status(self) -> Dict:
        """
        Get AI service status information
        
        Returns:
            Dictionary with service status
        """
        return {
            "available": self.is_available(),
            "rag_initialized": self.rag_service is not None,
            "api_key_configured": bool(os.getenv('GEMINI_API_KEY'))
        }
