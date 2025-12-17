"""
Enhanced RAG Service V2 with Reranking, Token Counting, and Hybrid Search
Extends PgVectorRAG with advanced retrieval techniques
Supports both Gemini and Ollama for answer generation
"""

import os
import logging
import json
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from services.rag_pgvector import PgVectorRAG
from services.ollama_service import OllamaService

# Setup logging
logger = logging.getLogger(__name__)

# Try to import tiktoken, use fallback if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


class EnhancedRAGV2(PgVectorRAG):
    """
    Enhanced RAG with:
    - Cross-encoder reranking
    - Token counting and budget management
    - Hybrid BM25 + Vector search
    - Improved prompt engineering
    - Better context assembly
    """

    def __init__(self, database_url: str, use_ollama: bool = None):
        """
        Initialize enhanced RAG service

        Args:
            database_url: PostgreSQL connection URL
            use_ollama: Whether to use Ollama instead of Gemini (default: check env var USE_OLLAMA)
        """
        # Check if we should use sentence transformers
        use_sentence_transformers = os.getenv('USE_SENTENCE_TRANSFORMERS', 'false').lower() == 'true'

        super().__init__(database_url, use_sentence_transformers=use_sentence_transformers)

        # Initialize reranker
        self.reranker = None
        self.reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

        # Token counting
        self.encoder = None
        self.max_context_tokens = 8000  # Conservative limit for Gemini
        self._init_token_encoder()

        # BM25 index
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_enabled = False

        # Configuration (OPTIMIZED FOR SMALL, FOCUSED CHUNKS)
        self.use_reranking = True
        self.use_hybrid_search = False  # Will enable after building index
        self.rerank_top_k = 30  # Retrieve 30 candidates (increased for smaller chunks)
        self.final_top_k = 8  # Return top 8 after reranking (was 2, increased for smaller chunks)

        # Initialize Ollama service
        if use_ollama is None:
            use_ollama = os.getenv('USE_OLLAMA', 'false').lower() == 'true'

        self.use_ollama = use_ollama
        self.ollama_service = None

        if self.use_ollama:
            try:
                self.ollama_service = OllamaService(
                    model_name=os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
                )
                if self.ollama_service.is_available():
                    logger.info("EnhancedRAGV2 initialized with Ollama for answer generation")
                else:
                    logger.warning("Ollama not available, falling back to Gemini")
                    self.use_ollama = False
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
                self.use_ollama = False
        else:
            logger.info("EnhancedRAGV2 initialized with Gemini for answer generation")

    def _init_reranker(self):
        """Lazy initialization of cross-encoder reranker"""
        if self.reranker is None:
            try:
                logger.info(f"Loading reranker model: {self.reranker_model_name}")
                self.reranker = CrossEncoder(self.reranker_model_name)
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.use_reranking = False

    def _init_token_encoder(self):
        """Initialize token encoder for counting"""
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("Token encoder initialized (tiktoken)")
            except Exception as e:
                logger.warning(f"Failed to init tiktoken: {e}")
                self.encoder = None
        else:
            self.encoder = None
            logger.info("Using approximate token counting (4 chars per token)")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        else:
            # Approximate: ~4 characters per token
            return len(text) // 4

    def build_bm25_index(self):
        """Build BM25 index from all chunks in database"""
        try:
            logger.info("Building BM25 index...")
            conn = self._connect()

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, chunk_text, department_name, program_abbreviation, info_type
                    FROM knowledge_chunks
                """)
                rows = cur.fetchall()

                self.bm25_corpus = []
                for row in rows:
                    chunk_id, text, dept, prog, info = row
                    # Tokenize for BM25
                    tokens = text.lower().split()
                    self.bm25_corpus.append({
                        'id': chunk_id,
                        'tokens': tokens,
                        'text': text,
                        'department_name': dept,
                        'program_abbreviation': prog,
                        'info_type': info
                    })

                # Build BM25 index
                tokenized_corpus = [doc['tokens'] for doc in self.bm25_corpus]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self.bm25_enabled = True
                self.use_hybrid_search = True  # Enable hybrid search now that index is built

                logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")
                logger.info("Hybrid search enabled")

            conn.close()

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
            self.bm25_enabled = False

    def bm25_search(
        self,
        query: str,
        department_name: Optional[str] = None,
        program_abbreviation: Optional[str] = None,
        info_type: Optional[str] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """BM25 keyword search with optional metadata filtering"""
        if not self.bm25_enabled or self.bm25_index is None:
            return []

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Filter by metadata if provided
            candidates = []
            for idx, score in enumerate(scores):
                doc = self.bm25_corpus[idx]

                # Apply metadata filters
                if department_name and doc.get('department_name') != department_name:
                    continue
                if program_abbreviation and doc.get('program_abbreviation') != program_abbreviation:
                    continue
                if info_type and doc.get('info_type') != info_type:
                    continue

                candidates.append({
                    'id': doc['id'],
                    'chunk_text': doc['text'],
                    'department_name': doc.get('department_name'),
                    'program_abbreviation': doc.get('program_abbreviation'),
                    'info_type': doc.get('info_type'),
                    'bm25_score': float(score)
                })

            # Sort by BM25 score and return top-k
            candidates.sort(key=lambda x: x['bm25_score'], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}", exc_info=True)
            return []

    def hybrid_search(
        self,
        query: str,
        department_name: Optional[str] = None,
        program_abbreviation: Optional[str] = None,
        info_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid search combining vector + BM25 with Reciprocal Rank Fusion (RRF)
        """
        if not self.bm25_enabled:
            # Fallback to vector-only
            return self.query(query, department_name, program_abbreviation, info_type, top_k)

        try:
            # Get results from both methods
            vector_results = self.query(
                query, department_name, program_abbreviation, info_type, top_k=20
            )
            bm25_results = self.bm25_search(
                query, department_name, program_abbreviation, info_type, top_k=20
            )

            logger.debug(f"Vector search: {len(vector_results)} results, BM25: {len(bm25_results)} results")

            # Reciprocal Rank Fusion (RRF)
            k = 60  # RRF constant
            rrf_scores = defaultdict(float)

            # Add vector scores
            for rank, chunk in enumerate(vector_results):
                chunk_id = chunk['id']
                rrf_scores[chunk_id] += 1 / (k + rank + 1)

            # Add BM25 scores
            for rank, chunk in enumerate(bm25_results):
                chunk_id = chunk['id']
                rrf_scores[chunk_id] += 1 / (k + rank + 1)

            # Get top-k by RRF score
            top_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Fetch full chunk data
            chunk_map = {c['id']: c for c in vector_results + bm25_results}
            final_chunks = []

            for chunk_id, rrf_score in top_ids:
                if chunk_id in chunk_map:
                    chunk = chunk_map[chunk_id].copy()
                    chunk['rrf_score'] = rrf_score
                    final_chunks.append(chunk)

            logger.info(f"Hybrid search: {len(final_chunks)} chunks after RRF fusion")
            return final_chunks

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            # Fallback to vector-only
            return self.query(query, department_name, program_abbreviation, info_type, top_k)

    def rerank_chunks(self, query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Rerank chunks using cross-encoder"""
        if not self.use_reranking or len(chunks) == 0:
            return chunks

        # Lazy load reranker
        if self.reranker is None:
            self._init_reranker()

        if self.reranker is None:
            logger.warning("Reranker not available, skipping reranking")
            return chunks

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, chunk['chunk_text']] for chunk in chunks]

            # Get reranking scores
            scores = self.reranker.predict(pairs)

            # Add scores to chunks
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)

            # Sort by rerank score
            chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Limit to top K after reranking (NEW - optimized for large comprehensive chunks)
            limit = top_k if top_k is not None else self.final_top_k
            chunks = chunks[:limit]

            logger.debug(f"Reranked chunks (limited to top {limit}), top score: {chunks[0]['rerank_score']:.3f}")
            return chunks

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return chunks

    def assemble_context_with_budget(
        self,
        chunks: List[Dict],
        question: str,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, int, List[Dict]]:
        """
        Assemble context from chunks within token budget

        Returns:
            (context_text, total_tokens, chunks_used)
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        # Reserve tokens for prompt template and question
        prompt_overhead = 300  # Approximate
        question_tokens = self.count_tokens(question)
        available_tokens = max_tokens - prompt_overhead - question_tokens

        context_parts = []
        chunks_used = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            # Format chunk with metadata
            info_type = chunk.get('info_type', 'info')
            chunk_text = chunk.get('chunk_text', '')

            # Add source number for citations
            formatted_chunk = f"[Source {i+1}] [{info_type}] {chunk_text}"

            chunk_tokens = self.count_tokens(formatted_chunk)

            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_tokens > available_tokens:
                logger.warning(
                    f"Context budget exceeded at chunk {i+1}/{len(chunks)}. "
                    f"Using {total_tokens} tokens from {len(chunks_used)} chunks."
                )
                break

            context_parts.append(formatted_chunk)
            chunks_used.append(chunk)
            total_tokens += chunk_tokens

        context = "\n\n".join(context_parts)

        logger.info(
            f"Context assembled: {total_tokens} tokens from {len(chunks_used)}/{len(chunks)} chunks "
            f"(budget: {available_tokens})"
        )

        return context, total_tokens, chunks_used

    def retrieve_chunks_enhanced(
        self,
        question: str,
        dept: Optional[str] = None,
        prog: Optional[str] = None,
        info: Optional[str] = None,
        use_hybrid: bool = False,
        use_reranking: bool = True
    ) -> Tuple[List[Dict], str]:
        """
        Enhanced retrieval with reranking and optional hybrid search

        Returns:
            (chunks, stage)
        """
        # Stage 1: Program + Info Type (exact metadata)
        if prog and info:
            chunks = self._fetch_by_metadata(
                department_name=None,
                program_abbreviation=prog,
                info_type=info
            )
            if chunks:
                if use_reranking:
                    chunks = self.rerank_chunks(question, chunks)
                logger.info(f"Stage 1 (program+info): {len(chunks)} chunks, reranked={use_reranking}")
                return chunks, 'program+info'

        # Stage 2: Program ONLY (NEW - most important for accuracy!)
        # When we know the specific program, filter by it!
        if prog:
            if use_hybrid and self.bm25_enabled:
                chunks = self.hybrid_search(
                    question,
                    program_abbreviation=prog,
                    top_k=self.rerank_top_k if use_reranking else 5
                )
                stage = 'hybrid+prog'
            else:
                chunks = self.query(
                    question,
                    program_abbreviation=prog,
                    top_k=self.rerank_top_k if use_reranking else 5
                )
                stage = 'vector+prog'

            if chunks and use_reranking:
                chunks = self.rerank_chunks(question, chunks)

            logger.info(f"Stage 2 ({stage}): {len(chunks)} chunks, reranked={use_reranking}")
            return chunks[:self.final_top_k], stage

        # Stage 3: Department + Info Type (exact metadata)
        if dept and info:
            chunks = self._fetch_by_metadata(
                department_name=dept,
                program_abbreviation=None,
                info_type=info
            )
            if chunks:
                if use_reranking:
                    chunks = self.rerank_chunks(question, chunks)
                logger.info(f"Stage 3 (dept+info): {len(chunks)} chunks, reranked={use_reranking}")
                return chunks, 'dept+info'

        # Stage 4: Vector or Hybrid search with department filter
        if dept:
            if use_hybrid and self.bm25_enabled:
                chunks = self.hybrid_search(
                    question,
                    department_name=dept,
                    top_k=self.rerank_top_k if use_reranking else 5
                )
                stage = 'hybrid+dept'
            else:
                chunks = self.query(
                    question,
                    department_name=dept,
                    top_k=self.rerank_top_k if use_reranking else 5
                )
                stage = 'vector+dept'

            if chunks and use_reranking:
                chunks = self.rerank_chunks(question, chunks)

            logger.info(f"Stage 4 ({stage}): {len(chunks)} chunks, reranked={use_reranking}")
            return chunks[:self.final_top_k], stage  # Return top K after reranking (was hardcoded to 5)

        # Stage 5: Pure vector or hybrid search
        if use_hybrid and self.bm25_enabled:
            chunks = self.hybrid_search(
                question,
                top_k=self.rerank_top_k if use_reranking else 5
            )
            stage = 'hybrid'
        else:
            chunks = self.query(
                question,
                top_k=self.rerank_top_k if use_reranking else 5
            )
            stage = 'vector'

        if chunks and use_reranking:
            chunks = self.rerank_chunks(question, chunks)

        logger.info(f"Stage 5 ({stage}): {len(chunks)} chunks, reranked={use_reranking}")
        return chunks[:self.final_top_k], stage  # Return top K after reranking (was hardcoded to 5)

    def _format_source_chunks(self, chunks: List[Dict]) -> str:
        """
        Format source chunks for display in the response

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted string with source information
        """
        if not chunks:
            return ""

        source_parts = ["---", "**Sources:**", ""]

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('chunk_text', '')
            info_type = chunk.get('info_type', 'info')
            program = chunk.get('program_abbreviation', '')
            department = chunk.get('department_name', '')

            # Format metadata
            metadata_parts = []
            if program:
                metadata_parts.append(f"Program: {program}")
            if department:
                metadata_parts.append(f"Department: {department}")
            if info_type:
                metadata_parts.append(f"Type: {info_type}")

            metadata_str = " | ".join(metadata_parts) if metadata_parts else ""

            # Add source
            source_parts.append(f"**[Source {i+1}]** {metadata_str}")
            source_parts.append(f"{chunk_text}")
            source_parts.append("")  # Empty line between sources

        return "\n".join(source_parts)

    def generate_enhanced_response_v2(
        self,
        question: str,
        student_context: Optional[Dict] = None,
        extracted_entities: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None,
        use_hybrid: bool = False,
        use_reranking: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate response with enhanced retrieval and better prompting

        Returns:
            (response_text, metadata)
        """
        if not question or not question.strip():
            return "", {"error": "Empty question"}

        # Extract entities if not provided
        if extracted_entities:
            dept, prog, info = extracted_entities
            logger.debug("Using pre-extracted entities")
        else:
            dept, prog, info = self._extract_entities(question, use_cache=True)
            logger.debug(f"Extracted: dept={dept}, prog={prog}, info={info}")

        # Retrieve chunks with enhancements
        chunks, stage = self.retrieve_chunks_enhanced(
            question,
            dept=dept,
            prog=prog,
            info=info,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking
        )

        if not chunks:
            logger.info(f"No chunks found for question: {question[:50]}")
            return "I don't have specific information about that topic in my knowledge base.", {
                "retrieval_stage": "none",
                "chunks_retrieved": 0
            }

        # Assemble context with token budget
        context, token_count, chunks_used = self.assemble_context_with_budget(
            chunks, question
        )

        # Check if LLM generation should be skipped
        skip_llm = os.getenv("SKIP_LLM_GENERATION", "false").lower() == "true"

        if skip_llm:
            # Return formatted chunks without LLM generation (template-based)
            formatted_response = self._format_chunks_as_response(chunks_used, question)
            chunks_section = self._format_source_chunks(chunks_used)
            full_response = f"{formatted_response}\n\n{chunks_section}"

            logger.info("Skipping LLM generation, using template-based formatting")
            return full_response, {
                "retrieval_stage": stage,
                "chunks_retrieved": len(chunks),
                "chunks_used": len(chunks_used),
                "tokens_used": token_count,
                "reranking_used": use_reranking,
                "hybrid_search_used": use_hybrid,
                "generation_method": "template",
                "entities": {
                    "department": dept,
                    "program": prog,
                    "info_type": info
                },
                "chunks_data": chunks_used  # Include actual chunks used for generation
            }

        # Enhanced prompt with citations and adaptive length
        system_prompt = """You are the UAB Programs & Fees Assistant, an expert on UAB academic programs.

ROLE: Provide accurate, helpful information about UAB programs, admissions, requirements, and fees.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the numbered sources provided below - do not use external knowledge
2. Keep answers CONCISE and directly relevant to the question:
   - Simple fact questions (tuition, deadline, requirement): 1-3 sentences maximum
   - Comparison questions: Structured bullet list
   - Explanatory questions: 1-2 short paragraphs maximum
3. Cite sources using [Source N] format (e.g., "According to [Source 1]...")
4. Structure your answer clearly:
   - Use ## headings only if answering multiple sub-questions
   - Use bullet points (-) for lists or comparisons
   - NO unnecessary elaboration
5. Include ONLY information that directly answers the question
6. If sources lack information, state concisely: "The available information doesn't include [specific detail]."
7. AVOID repeating the question or adding filler content

NUMBERED SOURCES FROM UAB KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

CONCISE, FOCUSED ANSWER WITH CITATIONS:"""

        user_context = ""
        if student_context:
            context_lines = [f"{k}: {v}" for k, v in student_context.items()]
            user_context = f"\n\nStudent Context:\n" + "\n".join(context_lines)

        prompt = system_prompt.format(
            context=context,
            question=question
        ) + user_context

        # Generate response with Ollama or Gemini
        try:
            if self.use_ollama and self.ollama_service:
                # Use Ollama for generation
                success, response_text, error = self.ollama_service.generate_response(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024
                )

                if not success:
                    logger.error(f"Ollama generation failed: {error}")
                    return self._format_chunks_as_response(chunks_used, question), {
                        "error": error,
                        "retrieval_stage": stage,
                        "chunks_retrieved": len(chunks_used),
                        "tokens_used": token_count,
                        "generation_method": "ollama_failed"
                    }

                logger.info(
                    f"Generated response with Ollama ({len(response_text)} chars) using {len(chunks_used)} chunks, "
                    f"{token_count} tokens"
                )

                # Add source chunks to the response
                chunks_section = self._format_source_chunks(chunks_used)
                full_response = f"{response_text}\n\n{chunks_section}"

                metadata = {
                    "retrieval_stage": stage,
                    "chunks_retrieved": len(chunks),
                    "chunks_used": len(chunks_used),
                    "tokens_used": token_count,
                    "reranking_used": use_reranking,
                    "hybrid_search_used": use_hybrid,
                    "generation_method": "ollama",
                    "model": self.ollama_service.model_name,
                    "entities": {
                        "department": dept,
                        "program": prog,
                        "info_type": info
                    },
                    "chunks_data": chunks_used  # Include actual chunks used
                }

                return full_response, metadata

            else:
                # Use Gemini for generation
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    logger.error("GEMINI_API_KEY not configured")
                    return self._format_chunks_as_response(chunks_used, question), {
                        "error": "API key not configured",
                        "retrieval_stage": stage,
                        "chunks_retrieved": len(chunks_used),
                        "tokens_used": token_count
                    }

                genai.configure(api_key=api_key)
                
                # Ensure model name has the proper prefix
                model_name = getattr(self, 'generation_model_name', None)
                if not model_name:
                    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")
                
                # Ensure model name starts with 'models/' or 'tunedModels/'
                if not model_name.startswith(('models/', 'tunedModels/')):
                    model_name = f"models/{model_name}"
                
                model = genai.GenerativeModel(model_name)

                response_text = self._generate_with_retry(model, prompt)
                logger.info(
                    f"Generated response with Gemini ({len(response_text)} chars) using {len(chunks_used)} chunks, "
                    f"{token_count} tokens"
                )

                # Add source chunks to the response
                chunks_section = self._format_source_chunks(chunks_used)
                full_response = f"{response_text}\n\n{chunks_section}"

                metadata = {
                    "retrieval_stage": stage,
                    "chunks_retrieved": len(chunks),
                    "chunks_used": len(chunks_used),
                    "tokens_used": token_count,
                    "reranking_used": use_reranking,
                    "hybrid_search_used": use_hybrid,
                    "generation_method": "gemini",
                    "entities": {
                        "department": dept,
                        "program": prog,
                        "info_type": info
                    },
                    "chunks_data": chunks_used  # Include actual chunks used
                }

                return full_response, metadata

        except Exception as e:
            error_str = str(e)
            logger.error(f"Failed to generate response: {error_str}")

            # Fallback to template formatting
            if "429" in error_str or "quota" in error_str.lower():
                formatted_response = self._format_chunks_as_response(chunks_used, question)
                logger.info("Using formatted fallback response (quota exceeded)")
                return formatted_response, {
                    "retrieval_stage": stage,
                    "chunks_retrieved": len(chunks),
                    "chunks_used": len(chunks_used),
                    "fallback": "template",
                    "error": "quota_exceeded"
                }
            else:
                raise


def create_enhanced_rag_service(database_url: str, use_ollama: bool = None) -> EnhancedRAGV2:
    """
    Factory function to create enhanced RAG service

    Args:
        database_url: PostgreSQL connection URL
        use_ollama: Whether to use Ollama for generation (default: from env)

    Returns:
        EnhancedRAGV2 instance
    """
    service = EnhancedRAGV2(database_url, use_ollama=use_ollama)

    # Build BM25 index on initialization
    try:
        service.build_bm25_index()
    except Exception as e:
        logger.warning(f"Could not build BM25 index: {e}")

    return service
