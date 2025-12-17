"""
Centralized Entity Extraction Service
Handles all entity extraction with caching, error handling, and multiple extraction methods
"""

import re
import json
import logging
from typing import Optional, List, Tuple, Dict, Any
from difflib import SequenceMatcher
from datetime import datetime
import time

from models.entities import (
    Entity, EntityType, ExtractionMethod, ExtractionResult,
    ProgramEntity, DepartmentEntity, FeeEntity, CourseEntity,
    DateEntity, InfoTypeEntity,
    PROGRAM_ABBREVIATIONS, DEPARTMENT_ABBREVIATIONS, INFO_TYPE_MAPPINGS
)

# Setup logging
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Unified entity extraction service with multiple extraction methods
    """

    def __init__(self, config_path: Optional[str] = None, genai_model=None):
        """
        Initialize entity extractor

        Args:
            config_path: Path to entity_config.json
            genai_model: Optional Google Generative AI model for LLM extraction
        """
        self.genai_model = genai_model
        self.config = self._load_config(config_path)

        # Compile regex patterns for performance
        self.regex_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in self.config.get("regex_patterns", {}).items()
        }

        logger.info("EntityExtractor initialized with config")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if config_path is None:
            import os
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "entity_config.json"
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded entity config from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            "extraction": {
                "use_heuristic_first": True,
                "heuristic_confidence_threshold": 0.85,
                "llm_fallback_enabled": True
            },
            "similarity": {
                "fuzzy_match_threshold": 0.78
            },
            "stop_words": ["a", "an", "the", "is", "are", "was", "were"]
        }

    def extract_entities(
        self,
        query: str,
        use_llm_fallback: bool = True
    ) -> ExtractionResult:
        """
        Main entry point for entity extraction

        Args:
            query: User query string
            use_llm_fallback: Whether to use LLM if heuristics don't find entities

        Returns:
            ExtractionResult with all extracted entities
        """
        start_time = time.time()
        result = ExtractionResult(query=query)

        try:
            # Step 1: Extract using heuristics and regex
            heuristic_entities = self._extract_with_heuristics(query)
            result.entities.extend(heuristic_entities)

            # Step 2: Extract using regex patterns
            regex_entities = self._extract_with_regex(query)
            result.entities.extend(regex_entities)

            # Step 3: Calculate overall confidence
            avg_confidence = self._calculate_average_confidence(result.entities)

            # Step 4: Use QueryAnalyzer fallback if no entities or low confidence
            threshold = self.config["extraction"]["heuristic_confidence_threshold"]
            if avg_confidence < threshold:
                logger.info(f"Heuristic confidence {avg_confidence:.2f} below threshold {threshold}, using QueryAnalyzer fallback")
                query_analyzer_entities = self._extract_with_query_analyzer(query)
                if query_analyzer_entities:
                    result.entities.extend(query_analyzer_entities)
                    logger.info(f"QueryAnalyzer added {len(query_analyzer_entities)} entities")
                    # Recalculate confidence after adding QueryAnalyzer entities
                    avg_confidence = self._calculate_average_confidence(result.entities)

            # Step 5: Use LLM fallback if confidence is still low
            if (use_llm_fallback and
                self.config["extraction"]["llm_fallback_enabled"] and
                avg_confidence < threshold and
                self.genai_model is not None):

                logger.info(f"Confidence {avg_confidence:.2f} still below threshold {threshold}, using LLM fallback")
                llm_entities = self._extract_with_llm(query)
                result.entities.extend(llm_entities)
                result.used_llm = True

            # Step 6: Deduplicate and normalize
            result.entities = self._deduplicate_entities(result.entities)

            # Calculate extraction time
            result.extraction_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Extracted {len(result.entities)} entities in {result.extraction_time_ms:.2f}ms "
                f"(LLM: {result.used_llm})"
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            result.error = str(e)

        return result

    def _extract_with_heuristics(self, query: str) -> List[Entity]:
        """Extract entities using keyword matching and heuristics"""
        entities = []
        query_lower = query.lower()

        # Extract department
        dept_entity = self._extract_department(query, query_lower)
        if dept_entity:
            entities.append(dept_entity)

        # Extract program
        prog_entity = self._extract_program(query, query_lower)
        if prog_entity:
            entities.append(prog_entity)

        # Extract info type
        info_entity = self._extract_info_type(query, query_lower)
        if info_entity:
            entities.append(info_entity)

        # Extract school
        school_entity = self._extract_school(query, query_lower)
        if school_entity:
            entities.append(school_entity)

        # Extract degree level
        level_entity = self._extract_degree_level(query, query_lower)
        if level_entity:
            entities.append(level_entity)

        # Extract degree type
        degree_entity = self._extract_degree_type(query, query_lower)
        if degree_entity:
            entities.append(degree_entity)

        return entities

    def _extract_department(self, query: str, query_lower: str) -> Optional[DepartmentEntity]:
        """Extract department entity"""
        departments = self.config.get("departments", {}).get("keywords", {})

        for dept_name, keywords in departments.items():
            # Check full department name (search in lowercase but return proper case from config)
            if dept_name.lower() in query_lower:
                return DepartmentEntity(
                    value=dept_name,  # Use exact name from config (already proper case)
                    confidence=0.95,
                    method=ExtractionMethod.HEURISTIC,
                    department_abbreviation=DEPARTMENT_ABBREVIATIONS.get(dept_name)
                )

            # Check abbreviations
            for keyword in keywords:
                if self._is_word_in_query(keyword, query_lower):
                    confidence = 0.90 if len(keyword) <= 3 else 0.85
                    return DepartmentEntity(
                        value=dept_name,  # Use exact name from config (already proper case)
                        confidence=confidence,
                        method=ExtractionMethod.HEURISTIC,
                        department_abbreviation=DEPARTMENT_ABBREVIATIONS.get(dept_name)
                    )

        return None

    def _extract_program(self, query: str, query_lower: str) -> Optional[ProgramEntity]:
        """Extract program entity"""
        programs = self.config.get("programs", {})

        best_match = None
        best_confidence = 0.0

        for prog_id, prog_info in programs.items():
            keywords = prog_info.get("keywords", [])

            for keyword in keywords:
                if keyword in query_lower:
                    # Exact match
                    confidence = 0.95
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = ProgramEntity(
                            value=prog_info["full_name"],
                            confidence=confidence,
                            method=ExtractionMethod.HEURISTIC,
                            program_abbreviation=prog_info["abbreviation"],
                            program_full_name=prog_info["full_name"]
                        )
                else:
                    # Fuzzy match
                    similarity = self._fuzzy_similarity(keyword, query_lower)
                    threshold = self.config["similarity"]["fuzzy_match_threshold"]
                    if similarity >= threshold and similarity > best_confidence:
                        best_confidence = similarity
                        best_match = ProgramEntity(
                            value=prog_info["full_name"],
                            confidence=similarity,
                            method=ExtractionMethod.HEURISTIC,
                            program_abbreviation=prog_info["abbreviation"],
                            program_full_name=prog_info["full_name"]
                        )

        return best_match

    def _extract_info_type(self, query: str, query_lower: str) -> Optional[InfoTypeEntity]:
        """Extract information type entity"""
        info_types = self.config.get("info_types", {})

        # Generic filler words that shouldn't boost confidence
        generic_phrases = {"tell me about", "what is", "about", "information", "details", "description"}

        best_match = None
        best_confidence = 0.0
        best_keyword_length = 0
        best_is_specific = False

        for info_type, info_data in info_types.items():
            keywords = info_data.get("keywords", [])

            for keyword in keywords:
                if keyword in query_lower:
                    # Check if this is a domain-specific keyword (not generic)
                    is_specific = keyword not in generic_phrases

                    # Longer, more specific keywords get higher confidence
                    keyword_length = len(keyword)
                    base_confidence = 0.85 + (min(keyword_length, 20) / 100)

                    # Penalize generic phrases
                    if not is_specific:
                        base_confidence *= 0.7  # 30% penalty for generic terms

                    # Prioritize: 1) specific over generic, 2) higher confidence, 3) longer keywords
                    is_better = (
                        (is_specific and not best_is_specific) or  # Specific beats generic
                        (is_specific == best_is_specific and base_confidence > best_confidence) or  # Same specificity, higher confidence
                        (base_confidence == best_confidence and keyword_length > best_keyword_length)  # Same confidence, longer
                    )

                    if is_better:
                        best_confidence = base_confidence
                        best_keyword_length = keyword_length
                        best_is_specific = is_specific
                        # Normalize using mappings - map the keyword, then fall back to info_type
                        normalized = INFO_TYPE_MAPPINGS.get(keyword, INFO_TYPE_MAPPINGS.get(info_type, info_type))
                        best_match = InfoTypeEntity(
                            value=keyword,
                            normalized_value=normalized,
                            confidence=min(base_confidence, 0.95),  # Cap at 0.95
                            method=ExtractionMethod.HEURISTIC,
                            category=info_type
                        )

        return best_match

    def _extract_school(self, query: str, query_lower: str) -> Optional[Entity]:
        """Extract school entity"""
        schools = self.config.get("schools", {}).get("keywords", [])

        for school in schools:
            if school in query_lower:
                return Entity(
                    type=EntityType.SCHOOL,
                    value=school.title(),
                    confidence=0.90,
                    method=ExtractionMethod.HEURISTIC
                )

        return None

    def _extract_degree_level(self, query: str, query_lower: str) -> Optional[Entity]:
        """Extract degree level (undergraduate/graduate)"""
        levels = self.config.get("degree_levels", {}).get("keywords", {})

        for level, keywords in levels.items():
            for keyword in keywords:
                if self._is_word_in_query(keyword, query_lower):
                    return Entity(
                        type=EntityType.DEGREE_LEVEL,
                        value=level.capitalize(),
                        confidence=0.85,
                        method=ExtractionMethod.HEURISTIC
                    )

        return None

    def _extract_degree_type(self, query: str, query_lower: str) -> Optional[Entity]:
        """Extract degree type (Bachelor, Master, PhD, Certificate)"""
        types = self.config.get("degree_types", {}).get("keywords", {})

        for deg_type, keywords in types.items():
            for keyword in keywords:
                if self._is_word_in_query(keyword, query_lower):
                    return Entity(
                        type=EntityType.DEGREE_TYPE,
                        value=deg_type.capitalize(),
                        confidence=0.85,
                        method=ExtractionMethod.HEURISTIC
                    )

        return None

    def _extract_with_regex(self, query: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []

        # Extract course codes
        if "course_code" in self.regex_patterns:
            for match in self.regex_patterns["course_code"].finditer(query):
                dept_code = match.group(1)
                course_num = match.group(2)
                course_code = f"{dept_code} {course_num}"
                entities.append(CourseEntity(
                    value=course_code,
                    confidence=0.95,
                    method=ExtractionMethod.REGEX,
                    course_code=course_code,
                    course_department=dept_code,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        # Extract fee amounts
        if "fee_amount" in self.regex_patterns:
            for match in self.regex_patterns["fee_amount"].finditer(query):
                amount_str = match.group(1).replace(",", "")
                try:
                    amount = float(amount_str)
                    entities.append(FeeEntity(
                        value=match.group(0),
                        confidence=0.95,
                        method=ExtractionMethod.REGEX,
                        amount=amount,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                except ValueError:
                    pass

        # Extract student IDs
        if "student_id" in self.regex_patterns:
            for match in self.regex_patterns["student_id"].finditer(query):
                entities.append(Entity(
                    type=EntityType.STUDENT_ID,
                    value=match.group(0),
                    confidence=0.80,
                    method=ExtractionMethod.REGEX,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        # Extract dates (semesters)
        if "semester" in self.regex_patterns:
            for match in self.regex_patterns["semester"].finditer(query):
                entities.append(DateEntity(
                    value=match.group(0),
                    confidence=0.90,
                    method=ExtractionMethod.REGEX,
                    date_type="semester",
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return entities

    def _extract_with_query_analyzer(self, query: str) -> List[Entity]:
        """Extract entities using QueryAnalyzer pattern-based detection"""
        entities = []

        try:
            from services.query_analyzer import get_query_analyzer
            analyzer = get_query_analyzer()
            analysis = analyzer.analyze(query)

            # Extract department from keywords
            if analysis['keywords']:
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

                for keyword in analysis['keywords']:
                    if keyword in department_mapping:
                        dept_name = department_mapping[keyword]
                        entities.append(DepartmentEntity(
                            value=dept_name,
                            confidence=analysis['confidence'],
                            method=ExtractionMethod.HEURISTIC,
                            department_abbreviation=DEPARTMENT_ABBREVIATIONS.get(dept_name)
                        ))
                        logger.info(f"QueryAnalyzer extracted department: {dept_name}")
                        break

            # Extract info_type
            if analysis['info_type']:
                entities.append(InfoTypeEntity(
                    value=analysis['info_type'],
                    confidence=analysis['confidence'],
                    method=ExtractionMethod.HEURISTIC
                ))
                logger.info(f"QueryAnalyzer extracted info_type: {analysis['info_type']}")

        except Exception as e:
            logger.error(f"QueryAnalyzer extraction failed: {e}", exc_info=True)

        return entities

    def _extract_with_llm(self, query: str) -> List[Entity]:
        """Extract entities using LLM (Gemini) as fallback"""
        if self.genai_model is None:
            logger.warning("LLM model not available for entity extraction")
            return []

        try:
            prompt = f"""Extract entities from this query about UAB programs.
Return ONLY valid JSON with this exact structure:
{{
  "department": "department name or null",
  "program": "program abbreviation like MSECE or null",
  "info_type": "one of: admission_requirements, degree_requirements, deadlines, contact_info, overview, tuition, or null",
  "school": "school name or null",
  "degree_level": "undergraduate or graduate or null",
  "degree_type": "bachelor, master, phd, certificate, or null"
}}

Query: {query}

JSON:"""

            llm_config = self.config.get("llm", {})
            response = self.genai_model.generate_content(
                prompt,
                generation_config={
                    "temperature": llm_config.get("temperature", 0.1),
                    "max_output_tokens": 500
                }
            )

            # Parse response with robust error handling
            entities = self._parse_llm_response(response.text, query)
            logger.info(f"LLM extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}", exc_info=True)
            return []

    def _parse_llm_response(self, response_text: str, original_query: str) -> List[Entity]:
        """Parse LLM response with robust error handling"""
        entities = []

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []

            data = json.loads(json_match.group(0))

            # Validate and extract entities
            if data.get("department"):
                entities.append(DepartmentEntity(
                    value=data["department"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM,
                    department_abbreviation=DEPARTMENT_ABBREVIATIONS.get(data["department"].lower())
                ))

            if data.get("program"):
                entities.append(ProgramEntity(
                    value=data["program"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM,
                    program_abbreviation=data["program"]
                ))

            if data.get("info_type"):
                entities.append(InfoTypeEntity(
                    value=data["info_type"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM,
                    category=data["info_type"]
                ))

            if data.get("school"):
                entities.append(Entity(
                    type=EntityType.SCHOOL,
                    value=data["school"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM
                ))

            if data.get("degree_level"):
                entities.append(Entity(
                    type=EntityType.DEGREE_LEVEL,
                    value=data["degree_level"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM
                ))

            if data.get("degree_type"):
                entities.append(Entity(
                    type=EntityType.DEGREE_TYPE,
                    value=data["degree_type"],
                    confidence=0.80,
                    method=ExtractionMethod.LLM
                ))

            return entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}. Response: {response_text[:200]}")
            # Try regex fallback
            return self._parse_llm_response_with_regex(response_text)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return []

    def _parse_llm_response_with_regex(self, response_text: str) -> List[Entity]:
        """Fallback parser using regex if JSON parsing fails"""
        entities = []

        # Extract department
        dept_match = re.search(r'"department"\s*:\s*"([^"]+)"', response_text)
        if dept_match and dept_match.group(1).lower() != "null":
            entities.append(DepartmentEntity(
                value=dept_match.group(1),
                confidence=0.70,
                method=ExtractionMethod.LLM
            ))

        # Extract program
        prog_match = re.search(r'"program"\s*:\s*"([^"]+)"', response_text)
        if prog_match and prog_match.group(1).lower() != "null":
            entities.append(ProgramEntity(
                value=prog_match.group(1),
                confidence=0.70,
                method=ExtractionMethod.LLM,
                program_abbreviation=prog_match.group(1)
            ))

        # Extract info_type
        info_match = re.search(r'"info_type"\s*:\s*"([^"]+)"', response_text)
        if info_match and info_match.group(1).lower() != "null":
            entities.append(InfoTypeEntity(
                value=info_match.group(1),
                confidence=0.70,
                method=ExtractionMethod.LLM,
                category=info_match.group(1)
            ))

        logger.info(f"Regex fallback extracted {len(entities)} entities from malformed JSON")
        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence"""
        if not entities:
            return []

        # Group by type and normalized value
        entity_map: Dict[Tuple[EntityType, str], Entity] = {}

        for entity in entities:
            key = (entity.type, entity.normalized_value)
            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity

        return list(entity_map.values())

    def _calculate_average_confidence(self, entities: List[Entity]) -> float:
        """Calculate average confidence of extracted entities"""
        if not entities:
            return 0.0
        return sum(e.confidence for e in entities) / len(entities)

    def _fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Calculate fuzzy string similarity"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _is_word_in_query(self, word: str, query_lower: str) -> bool:
        """Check if word exists as whole word in query (not substring)"""
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, query_lower))

    def extract_search_terms(self, query: str) -> List[str]:
        """
        Extract search terms from query by removing stop words
        Used for program search
        """
        stop_words = set(self.config.get("stop_words", []))
        tokens = query.lower().split()
        search_terms = [t for t in tokens if t not in stop_words and len(t) > 2]
        return search_terms


# Global instance (initialized by app.py)
_entity_extractor_instance: Optional[EntityExtractor] = None


def get_entity_extractor() -> Optional[EntityExtractor]:
    """Get global entity extractor instance"""
    return _entity_extractor_instance


def set_entity_extractor(extractor: EntityExtractor):
    """Set global entity extractor instance"""
    global _entity_extractor_instance
    _entity_extractor_instance = extractor
