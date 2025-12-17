"""
PostgreSQL pgvector-backed retrieval service.
Implements filter-then-fetch over metadata, then vector similarity.
"""

import os
import json
import psycopg2
import time
import re
import logging
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai

from services.entity_extractor import EntityExtractor, get_entity_extractor
from services.entity_cache import get_entity_cache
from services.rate_limiter import get_gemini_rate_limiter
from services.query_analyzer import get_query_analyzer
from models.entities import ExtractionResult

logger = logging.getLogger(__name__)


class PgVectorRAG:
    def __init__(
        self,
        database_url: Optional[str] = None,
        embedding_model: str = "all-mpnet-base-v2",
        use_sentence_transformers: bool = True
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for PgVectorRAG")

        self.use_sentence_transformers = use_sentence_transformers

        # Initialize embedding service
        if self.use_sentence_transformers:
            from services.sentence_transformer_embeddings import get_embedding_service
            self.embedding_service = get_embedding_service(embedding_model)
            logger.info(f"Using Sentence Transformers for embeddings: {embedding_model}")
            # Still need to configure Gemini for fallback generation
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.generation_model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")
                logger.info("Gemini configured for text generation fallback")
            else:
                self.generation_model_name = None
                logger.warning("GEMINI_API_KEY not found, will rely on Ollama for generation")
        else:
            # Fallback to Gemini (original)
            # Use Gemini embedding model instead of sentence-transformer model name
            gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
            self.embedding_model = gemini_embedding_model
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for embeddings")
            genai.configure(api_key=api_key)
            self.generation_model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")
            logger.info(f"Using Gemini for embeddings: {gemini_embedding_model}")

        # Initialize LLM service (for generation)
        use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
        if use_ollama:
            try:
                from services.ollama_llm import get_ollama_service
                self.llm_service = get_ollama_service()
                logger.info("Using Ollama for text generation")
            except Exception as e:
                logger.warning(f"Ollama not available, will try Gemini: {e}")
                self.llm_service = None
        else:
            self.llm_service = None

        # Get global services
        self.entity_extractor = get_entity_extractor()
        self.entity_cache = get_entity_cache()

        # Initialize rate limiter if using Gemini for generation
        if hasattr(self, 'generation_model_name') and self.generation_model_name:
            self.rate_limiter = get_gemini_rate_limiter()
        else:
            self.rate_limiter = None

        # Ensure extension/table exist (idempotent)
        self._ensure_schema()

        logger.info("PgVectorRAG initialized")

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def _ensure_schema(self):
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        id SERIAL PRIMARY KEY,
                        department_name VARCHAR(255),
                        program_abbreviation VARCHAR(50),
                        info_type VARCHAR(100),
                        chunk_text TEXT NOT NULL,
                        text_vector VECTOR(768)
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_meta ON knowledge_chunks (department_name, program_abbreviation, info_type);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_ivfflat ON knowledge_chunks USING ivfflat (text_vector vector_cosine_ops) WITH (lists = 100);"
                )
        finally:
            conn.close()

    def _embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * 768

        if self.use_sentence_transformers:
            # Use sentence-transformers (local, free, no rate limits)
            return self.embedding_service.embed_text(text)
        else:
            # Use Gemini (original implementation)
            res = genai.embed_content(model=self.embedding_model, content=text)
            return res.get("embedding", [0.0] * 768)

    def _embedding_to_pgvector_string(self, embedding: List[float]) -> str:
        """
        CRITICAL FIX: Convert embedding list to PostgreSQL vector format string.

        PostgreSQL pgvector requires format: "[0.1, 0.2, 0.3, ...]"
        This fixes the bug: "invalid input syntax for type vector"
        """
        if not embedding:
            return "[" + ",".join(["0.0"] * 768) + "]"

        try:
            # Handle numpy arrays
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            # Ensure all floats
            embedding = [float(x) for x in embedding]

            # Format as PostgreSQL vector: [0.1,0.2,0.3,...]
            vector_str = "[" + ",".join(str(x) for x in embedding) + "]"

            return vector_str
        except Exception as e:
            logger.error(f"Vector format conversion failed: {e}")
            return "[" + ",".join(["0.0"] * 768) + "]"

    def _fetch_by_metadata(
        self,
        department_name: Optional[str] = None,
        program_abbreviation: Optional[str] = None,
        info_type: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch chunks by precise metadata filtering only (no vector search)."""
        where_clauses = []
        params: List = []
        if department_name:
            where_clauses.append("department_name = %s")
            params.append(department_name)
        if program_abbreviation:
            where_clauses.append("program_abbreviation = %s")
            params.append(program_abbreviation)
        if info_type:
            where_clauses.append("info_type = %s")
            params.append(info_type)

        if not where_clauses:
            return []

        where_sql = "WHERE " + " AND ".join(where_clauses)
        sql = f"""
            SELECT id, department_name, program_abbreviation, info_type, chunk_text
            FROM knowledge_chunks
            {where_sql}
            ORDER BY id ASC
        """

        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append(
                        {
                            "id": r[0],
                            "department_name": r[1],
                            "program_abbreviation": r[2],
                            "info_type": r[3],
                            "chunk_text": r[4],
                            "distance": None,
                        }
                    )
                return results
        finally:
            conn.close()

    def _extract_entities(self, question: str, use_cache: bool = True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract entities using the centralized EntityExtractor service.

        Returns: (department_name, program_abbreviation, info_type)
        """
        # Check cache first
        if use_cache and self.entity_cache:
            cached = self.entity_cache.get(question, use_llm=True)
            if cached:
                logger.debug(f"Using cached entities for: {question[:50]}")
                legacy = cached.to_legacy_format()
                return legacy["department_name"], legacy["program_abbreviation"], legacy["info_type"]

        # Use centralized entity extractor if available
        if self.entity_extractor:
            try:
                result = self.entity_extractor.extract_entities(question, use_llm_fallback=True)

                # Cache the result
                if use_cache and self.entity_cache:
                    self.entity_cache.set(question, result, use_llm=True)

                # Convert to legacy format for backward compatibility
                legacy = result.to_legacy_format()
                logger.info(
                    f"Extracted entities: dept={legacy['department_name']}, "
                    f"prog={legacy['program_abbreviation']}, info={legacy['info_type']} "
                    f"(LLM used: {result.used_llm}, time: {result.extraction_time_ms:.2f}ms)"
                )
                return legacy["department_name"], legacy["program_abbreviation"], legacy["info_type"]
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}", exc_info=True)

        # Fallback to simple heuristics if extractor not available
        logger.warning("EntityExtractor not available, using basic heuristics")
        return self._extract_entities_fallback(question)

    def _extract_entities_fallback(self, question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fallback entity extraction using QueryAnalyzer for robust info_type detection
        and keyword-based department/program matching.
        Returns: (department_name, program_abbreviation, info_type)
        """
        # Use QueryAnalyzer for robust info_type detection
        try:
            analyzer = get_query_analyzer()
            analysis = analyzer.analyze(question)

            # Get info_type from analyzer
            info_type = analysis['info_type']

            # Get department from keywords
            keywords = analysis['keywords']
            dept = None

            # Map keywords to canonical department names
            department_mapping = {
                'civil engineering': 'Civil Engineering',
                'computer science': 'Computer Science',
                'electrical engineering': 'Electrical and Computer Engineering',
                'mechanical engineering': 'Mechanical Engineering',
                'biomedical engineering': 'Biomedical Engineering',
                'business': 'Business',
                'mba': 'Business',
                'nursing': 'Nursing',
                'public health': 'Public Health',
                'materials engineering': 'Materials Engineering',
                'engineering management': 'Engineering Management',
            }

            for keyword in keywords:
                if keyword in department_mapping:
                    dept = department_mapping[keyword]
                    break

            logger.info(f"QueryAnalyzer extraction: dept={dept}, info={info_type}, confidence={analysis['confidence']:.2f}")
            return dept, None, info_type  # prog=None since we don't extract it yet

        except Exception as e:
            logger.error(f"QueryAnalyzer extraction failed: {e}, using basic fallback")

            # Ultra-basic fallback
            ql = (question or "").lower()
            dept = None
            if "civil engineering" in ql:
                dept = "Civil Engineering"
            elif "computer science" in ql:
                dept = "Computer Science"

            info = None
            if "deadline" in ql:
                info = "deadlines"
            elif "admission" in ql:
                info = "admission_requirements"

            return dept, None, info

    def query(
        self,
        query_text: str,
        department_name: Optional[str] = None,
        program_abbreviation: Optional[str] = None,
        info_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        embedding = self._embed(query_text)
        # CRITICAL FIX: Convert embedding list to PostgreSQL vector format string
        embedding_str = self._embedding_to_pgvector_string(embedding)
        logger.debug(f"Vector format: {embedding_str[:60]}...")

        where_clauses = []
        params: List = []
        if department_name:
            where_clauses.append("department_name = %s")
            params.append(department_name)
        if program_abbreviation:
            where_clauses.append("program_abbreviation = %s")
            params.append(program_abbreviation)
        if info_type:
            where_clauses.append("info_type = %s")
            params.append(info_type)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # CRITICAL FIX: Build SQL with correct parameter placeholder ordering
        # The WHERE clause parameters must come BEFORE the vector parameters
        sql = f"""
            SELECT id, department_name, program_abbreviation, info_type, chunk_text,
                   (text_vector <-> %s::vector) AS distance
            FROM knowledge_chunks
            {where_sql}
            ORDER BY text_vector <-> %s::vector
            LIMIT %s;
        """

        # CRITICAL FIX: Correct parameter ordering
        # SQL placeholders order: distance calc (%s), WHERE filters (...), ORDER BY (%s), LIMIT (%s)
        # But psycopg2 processes them left-to-right in the SQL, so:
        # 1st %s = distance calculation (embedding_str)
        # 2nd-%nth %s = WHERE clause params (dept, prog, info)
        # (n+1)th %s = ORDER BY (embedding_str again)
        # last %s = LIMIT (top_k)

        # Build params in the exact order they appear in SQL
        params_full = [embedding_str] + params + [embedding_str, top_k]

        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql, params_full)
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append(
                        {
                            "id": r[0],
                            "department_name": r[1],
                            "program_abbreviation": r[2],
                            "info_type": r[3],
                            "chunk_text": r[4],
                            "distance": float(r[5]) if r[5] is not None else None,
                        }
                    )
                return results
        finally:
            conn.close()

    def _generate_with_retry(self, model, prompt: str) -> str:
        """
        Generate content using Ollama (preferred) or Gemini (fallback).
        Returns the generated text or raises the last exception.
        """
        # Try Ollama first if available
        if hasattr(self, 'llm_service') and self.llm_service:
            try:
                logger.debug("Using Ollama for generation")
                response = self.llm_service.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=2000
                )
                if response:
                    return response.strip()
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}, falling back to Gemini")

        # Fallback to Gemini
        # Use rate limiter if available
        if hasattr(self, 'rate_limiter') and self.rate_limiter:
            def _generate():
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                return "I couldn't generate a response."

            success, result, error = self.rate_limiter.execute_with_backoff(_generate)
            if success:
                return result
            else:
                logger.error(f"Generation failed after retries: {error}")
                raise Exception(f"Rate limit exceeded: {error}")
        else:
            # Fallback without rate limiter
            logger.warning("Rate limiter not available, generating without throttling")
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                return "I couldn't generate a response."
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    def _filter_chunks_by_keywords(self, chunks: List[Dict], question_lower: str) -> List[Dict]:
        """
        Filter chunks to only those that match keywords in the question.
        This helps when entity extraction fails.
        """
        # Common department and program keywords
        keywords_to_check = [
            'civil engineering', 'computer science', 'electrical engineering',
            'mechanical engineering', 'biomedical engineering', 'mba', 'business',
            'nursing', 'public health', 'biology', 'chemistry', 'physics',
            'mathematics', 'psychology', 'sociology', 'history', 'english'
        ]

        # Find keywords in question
        found_keywords = [kw for kw in keywords_to_check if kw in question_lower]

        if not found_keywords:
            return []  # No filtering needed

        # Filter chunks that match any of the found keywords
        filtered = []
        for chunk in chunks:
            dept = chunk.get('department_name', '').lower()
            prog = chunk.get('program_abbreviation', '').lower()
            text = chunk.get('chunk_text', '').lower()

            # Check if any keyword matches department, program, or text
            for keyword in found_keywords:
                if keyword in dept or keyword in prog or keyword in text[:200]:  # Check first 200 chars
                    filtered.append(chunk)
                    break  # Don't add the same chunk twice

        return filtered

    def _format_chunks_as_response(self, chunks: List[Dict], question: str) -> str:
        """
        Format retrieved chunks as a readable response without LLM
        Template-based formatting for common query types
        IMPORTANT: Chunks are already ranked by relevance, so prioritize the top ones
        """
        if not chunks:
            return "I don't have specific information about that topic in my knowledge base."

        # Detect query type from question
        question_lower = question.lower()

        # Filter chunks by program/department if mentioned in question
        filtered_chunks = self._filter_chunks_by_keywords(chunks, question_lower)
        if filtered_chunks:
            chunks = filtered_chunks  # Use filtered chunks if we found matches

        # Group chunks by info_type while preserving order (most relevant first)
        grouped = {}
        for chunk in chunks:
            info_type = chunk.get('info_type', 'general')
            if info_type not in grouped:
                grouped[info_type] = []
            grouped[info_type].append(chunk.get('chunk_text', ''))

        # Build formatted response
        response_parts = []

        # Admission requirements query
        if 'admission' in question_lower or 'requirements' in question_lower or 'apply' in question_lower:
            if 'admission_requirements' in grouped:
                response_parts.append("## Admission Requirements\n")
                # Show top 2 most relevant chunks
                for text in grouped['admission_requirements'][:2]:
                    response_parts.append(f"{text}\n")

            if 'degree_requirements' in grouped and 'admission' not in question_lower:
                response_parts.append("\n## Degree Requirements\n")
                # Show top 2 most relevant chunks
                for text in grouped['degree_requirements'][:2]:
                    response_parts.append(f"{text}\n")

        # Deadline query
        elif 'deadline' in question_lower:
            if 'deadlines' in grouped:
                response_parts.append("## Application Deadlines\n")
                # Show only the most relevant deadline chunk (top 1)
                for text in grouped['deadlines'][:1]:
                    response_parts.append(f"{text}\n")

        # Contact query
        elif 'contact' in question_lower or 'email' in question_lower or 'phone' in question_lower:
            if 'contact_info' in grouped:
                response_parts.append("## Contact Information\n")
                # Show top 1 most relevant contact chunk
                for text in grouped['contact_info'][:1]:
                    response_parts.append(f"{text}\n")

        # Overview/general query
        elif 'what is' in question_lower or 'about' in question_lower or 'tell me' in question_lower:
            if 'program_overview' in grouped:
                response_parts.append("## Program Overview\n")
                # Show top 2 most relevant overview chunks
                for text in grouped['program_overview'][:2]:
                    response_parts.append(f"{text}\n")

            if 'admission_requirements' in grouped:
                response_parts.append("\n## Admission Requirements\n")
                # Show top 2 most relevant admission chunks
                for text in grouped['admission_requirements'][:2]:
                    response_parts.append(f"{text}\n")

        # Default: show top 3 most relevant chunks grouped by type
        else:
            for info_type, texts in grouped.items():
                # Clean up info_type name for display
                display_name = info_type.replace('_', ' ').title()
                response_parts.append(f"## {display_name}\n")
                # Limit to top 3 chunks per type
                for text in texts[:3]:
                    response_parts.append(f"{text}\n")

        response = '\n'.join(response_parts).strip()

        # Add helpful note at the end
        if not response:
            # Fallback if formatting didn't work
            response = '\n\n'.join([chunk.get('chunk_text', '') for chunk in chunks])

        return response

    def generate_enhanced_response(
        self, question: str, student_context: Optional[Dict] = None,
        extracted_entities: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None
    ) -> str:
        """
        Generate AI response using RAG retrieval + optional LLM generation

        Args:
            question: The user's question
            student_context: Optional student context information
            extracted_entities: Pre-extracted entities (dept, prog, info) to avoid duplicate extraction

        Returns:
            Generated response text
        """
        if not question or not question.strip():
            return ""

        # Use pre-extracted entities if provided, otherwise extract now
        if extracted_entities:
            dept, prog, info = extracted_entities
            logger.debug("Using pre-extracted entities (avoiding duplicate extraction)")
        else:
            dept, prog, info = self._extract_entities(question, use_cache=True)

        # Pass extracted entities to avoid duplicate extraction
        chunks, stage = self.retrieve_chunks(question, dept=dept, prog=prog, info=info)

        if not chunks:
            logger.info(f"No chunks found for question: {question[:50]}")
            return "I don't have specific information about that topic in my knowledge base."

        logger.info(f"Retrieved {len(chunks)} chunks at stage: {stage}")

        # Check if LLM generation should be skipped
        skip_llm = os.getenv("SKIP_LLM_GENERATION", "false").lower() == "true"

        if skip_llm:
            # Return formatted chunks without LLM generation (no API calls, no errors)
            formatted_response = self._format_chunks_as_response(chunks, question)
            logger.info("Skipping LLM generation, returning formatted chunks")
            return formatted_response

        # Build context from retrieved chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk.get('info_type', 'info')}] {chunk.get('chunk_text', '')}")

        context = "\n\n".join(context_parts)

        # Try LLM generation (Ollama or Gemini)
        try:
            # Try Ollama first if available
            if hasattr(self, 'llm_service') and self.llm_service:
                try:
                    logger.info("Attempting generation with Ollama")
                    system_prompt = (
                        "You are the UAB Programs & Fees assistant. Answer questions about academic programs "
                        "using the provided context. Be helpful, accurate, and concise."
                    )
                    prompt = f"""{system_prompt}

Context from knowledge base:
{context}

Question: {question}

Answer:"""
                    response_text = self.llm_service.generate(prompt, temperature=0.1, max_tokens=2000)
                    logger.info(f"Generated response with Ollama ({len(response_text)} chars)")
                    return response_text
                except Exception as e:
                    logger.warning(f"Ollama generation failed: {e}, trying Gemini")

            # Fallback to Gemini if available
            if hasattr(self, 'generation_model_name') and self.generation_model_name:
                logger.info("Attempting generation with Gemini")
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("GEMINI_API_KEY not configured, using formatted chunks")
                    return self._format_chunks_as_response(chunks, question)

                genai.configure(api_key=api_key)
                
                # Ensure model name has the proper prefix
                model_name = self.generation_model_name
                if not model_name.startswith(('models/', 'tunedModels/')):
                    model_name = f"models/{model_name}"
                
                model = genai.GenerativeModel(model_name)

                system_prompt = (
                    "You are the UAB Programs & Fees assistant. Answer questions about academic programs "
                    "using the provided context. Be helpful, accurate, and concise."
                )

                prompt = f"""{system_prompt}

Context from knowledge base:
{context}

Question: {question}

Answer:"""

                response_text = self._generate_with_retry(model, prompt)
                logger.info(f"Generated response with Gemini ({len(response_text)} chars)")
                return response_text
            else:
                # No LLM available, return formatted chunks
                logger.info("No LLM available, returning formatted chunks")
                return self._format_chunks_as_response(chunks, question)

        except Exception as e:
            error_str = str(e)
            # Silently fall back to formatted chunks (no error logging for quota issues)
            if "429" in error_str or "quota" in error_str.lower():
                logger.info("LLM quota exceeded, using formatted chunks")
            else:
                logger.warning(f"LLM generation failed: {error_str[:100]}, using formatted chunks")
            return self._format_chunks_as_response(chunks, question)

    def retrieve_chunks(
        self,
        question: str,
        dept: Optional[str] = None,
        prog: Optional[str] = None,
        info: Optional[str] = None
    ) -> Tuple[List[Dict], str]:
        """
        Run strict cascade and return (chunks, stage):
        - stage values: 'program+info', 'dept+info', 'vector+dept', 'vector'

        Args:
            question: The question text
            dept: Pre-extracted department name (optional, will extract if not provided)
            prog: Pre-extracted program abbreviation (optional, will extract if not provided)
            info: Pre-extracted info type (optional, will extract if not provided)
        """
        # Extract entities only if not provided (avoids duplicate API calls)
        if dept is None and prog is None and info is None:
            dept, prog, info = self._extract_entities(question, use_cache=True)
            logger.debug(f"Extracted entities in retrieve_chunks: dept={dept}, prog={prog}, info={info}")
        else:
            logger.debug("Using pre-extracted entities (no duplicate extraction)")

        # Stage 1: Program + Info Type
        if prog and info:
            chunks = self._fetch_by_metadata(
                department_name=None,
                program_abbreviation=prog,
                info_type=info,
            )
            if chunks:
                logger.info(f"Stage 1 (program+info) found {len(chunks)} chunks")
                return chunks, 'program+info'

        # Stage 1.5: Program only (no info_type) - default to overview/admission
        if prog and not info:
            # Try to get overview_admission first, then program_overview, then admission_requirements
            for default_info_type in ['overview_admission', 'program_overview', 'admission_requirements']:
                chunks = self._fetch_by_metadata(
                    department_name=None,
                    program_abbreviation=prog,
                    info_type=default_info_type,
                )
                if chunks:
                    logger.info(f"Stage 1.5 (program+default_overview) found {len(chunks)} chunks with info_type={default_info_type}")
                    return chunks[:5], 'program+overview'  # Return up to 5 chunks

        # Stage 2: Department + Info Type
        if dept and info:
            chunks = self._fetch_by_metadata(
                department_name=dept,
                program_abbreviation=None,
                info_type=info,
            )
            if chunks:
                logger.info(f"Stage 2 (dept+info) found {len(chunks)} chunks")
                return chunks, 'dept+info'

        # Stage 3: Vector search with department filter
        if dept:
            chunks = self.query(question, department_name=dept, top_k=5)

            # Apply degree-level filtering
            question_lower = question.lower()
            if any(word in question_lower for word in ['master', 'masters', 'ms', 'mba', 'mph', 'ma', 'msha', 'mspas']):
                chunks = [c for c in chunks if 'phd' not in c.get('program_abbreviation', '').lower()]
            elif any(word in question_lower for word in ['phd', 'doctorate', 'doctoral']):
                chunks = [c for c in chunks if 'phd' in c.get('program_abbreviation', '').lower()]

            # Prioritize info_type based on query intent
            # For general queries, prioritize overview over detailed requirements
            has_specific_info_request = any(word in question_lower for word in
                ['requirement', 'requirements', 'course', 'courses', 'curriculum',
                 'tuition', 'cost', 'fee', 'fees', 'deadline', 'apply', 'application'])

            if not has_specific_info_request:
                # General query - prioritize overview_admission over degree_requirements
                overview_chunks = [c for c in chunks if c.get('info_type') == 'overview_admission']
                other_chunks = [c for c in chunks if c.get('info_type') != 'overview_admission']
                chunks = overview_chunks + other_chunks

            # Limit to top 2 most relevant chunks
            chunks = chunks[:2]

            logger.info(f"Stage 3 (vector+dept) found {len(chunks)} chunks (filtered to 2 most relevant)")
            return chunks, 'vector+dept'

        # Stage 4: Pure vector search
        chunks = self.query(question, top_k=5)

        # Apply degree-level filtering
        question_lower = question.lower()
        if any(word in question_lower for word in ['master', 'masters', 'ms', 'mba', 'mph', 'ma', 'msha', 'mspas']):
            chunks = [c for c in chunks if 'phd' not in c.get('program_abbreviation', '').lower()]
        elif any(word in question_lower for word in ['phd', 'doctorate', 'doctoral']):
            chunks = [c for c in chunks if 'phd' in c.get('program_abbreviation', '').lower()]

        # Prioritize info_type based on query intent
        # For general queries, prioritize overview over detailed requirements
        has_specific_info_request = any(word in question_lower for word in
            ['requirement', 'requirements', 'course', 'courses', 'curriculum',
             'tuition', 'cost', 'fee', 'fees', 'deadline', 'apply', 'application'])

        if not has_specific_info_request:
            # General query - prioritize overview_admission over degree_requirements
            overview_chunks = [c for c in chunks if c.get('info_type') == 'overview_admission']
            other_chunks = [c for c in chunks if c.get('info_type') != 'overview_admission']
            chunks = overview_chunks + other_chunks

        # Limit to top 2 most relevant chunks
        chunks = chunks[:2]

        logger.info(f"Stage 4 (vector) found {len(chunks)} chunks (filtered to 2 most relevant)")
        return chunks, 'vector'


