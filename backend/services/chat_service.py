"""
Chat Service for UAB Program Finder
Handles intent recognition, entity extraction, and program search
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
import difflib
from program_loader import ProgramLoader
from services.entity_extractor import get_entity_extractor
from models.entities import EntityType

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling natural language chat queries about programs"""

    def __init__(self, program_loader: ProgramLoader):
        self.program_loader = program_loader
        self.entity_extractor = get_entity_extractor()
        self.school_keywords = {
            'engineering': ['School of Engineering'],
            'business': ['Collat School of Business'],
            'arts': ['College of Arts and Sciences'],
            'nursing': ['School of Nursing'],
            'education': ['School of Education', 'School of Education and Human Sciences'],
            'public health': ['School of Public Health'],
            'graduate': ['Graduate School'],
            'health': ['School of Health Professions']
        }
        
        self.level_keywords = {
            'undergraduate': ['undergraduate', 'undergrad', 'bachelor', 'bs', 'ba', 'bachelor\'s'],
            'graduate': ['graduate', 'grad', 'master', 'phd', 'doctorate', 'ms', 'ma', 'mba', 'master\'s', 'doctoral']
        }
        
        self.degree_type_keywords = {
            'bachelor': ['bachelor', 'bs', 'ba', 'bachelor\'s'],
            'master': ['master', 'ms', 'ma', 'mba', 'master\'s'],
            'phd': ['phd', 'doctorate', 'doctoral', 'ph.d'],
            'certificate': ['certificate', 'cert', 'certification']
        }
    
    def classify_intent(self, question: str) -> str:
        """
        Classify user intent as either 'program_search' or 'general_question'
        """
        question_lower = question.lower()
        
        # Keywords that indicate program search intent
        program_search_keywords = [
            'program', 'programs', 'degree', 'degrees', 'major', 'majors',
            'study', 'studies', 'course', 'courses', 'undergraduate', 'graduate',
            'bachelor', 'master', 'phd', 'doctorate', 'engineering', 'business',
            'arts', 'science', 'medicine', 'nursing', 'education', 'school',
            'college', 'department', 'field', 'discipline', 'subject'
        ]
        
        # Check if question contains program search keywords
        for keyword in program_search_keywords:
            if keyword in question_lower:
                return 'program_search'
        
        # Check for specific question patterns
        program_patterns = [
            r'what.*programs?',
            r'what.*degrees?',
            r'what.*majors?',
            r'what.*can.*study',
            r'what.*engineering',
            r'what.*business',
            r'what.*arts',
            r'what.*science',
            r'list.*programs?',
            r'show.*programs?',
            r'find.*programs?',
            r'search.*programs?',
            r'available.*programs?',
            r'offered.*programs?'
        ]
        
        for pattern in program_patterns:
            if re.search(pattern, question_lower):
                return 'program_search'

        # Heuristic: if the question contains known degree acronyms or looks like a program code
        acronym_hits = [
            'ms', 'ma', 'mba', 'mph', 'msece', 'msce', 'msme', 'msmte', 'msbme',
            'bs', 'ba', 'bfa',
            'phd', 'ph.d'
        ]
        for a in acronym_hits:
            if a in question_lower:
                return 'program_search'

        # Fallback: if any token appears in any program/school/department name, treat as program search
        try:
            tokens = [t for t in re.split(r"\W+", question_lower) if t]
            programs = self.program_loader.get_all_programs()
            program_texts = [
                f"{p.get('program_name','')} {p.get('department','')}".lower()
                for p in programs
            ]
            school_texts = [s.lower() for s in self.program_loader.get_schools()]
            corpus = program_texts + school_texts
            if any(any(tok in text for text in corpus) for tok in tokens if len(tok) >= 3):
                return 'program_search'
        except Exception:
            pass

        # Fuzzy fallback: tokenize corpus and allow small typos (e.g., "nursuing porgram")
        try:
            def corpus_words():
                words = set()
                for text in corpus:
                    for w in re.split(r"\W+", text):
                        if len(w) >= 4:
                            words.add(w)
                return words
            words_set = corpus_words()
            for tok in tokens:
                if len(tok) < 4:
                    continue
                for w in words_set:
                    if difflib.SequenceMatcher(None, tok, w).ratio() >= 0.80:
                        return 'program_search'
            # near-miss for the word "program"
            for tok in tokens:
                if difflib.SequenceMatcher(None, tok, 'program').ratio() >= 0.78:
                    return 'program_search'
        except Exception:
            pass
        
        return 'general_question'
    
    def extract_entities(self, question: str) -> Dict[str, str]:
        """
        Extract entities (school, level, degree_type, search_term) from user question
        Uses centralized entity extractor when available
        """
        entities = {
            'school': '',
            'level': '',
            'degree_type': '',
            'search_term': ''
        }

        # Try using centralized entity extractor first
        if self.entity_extractor:
            try:
                result = self.entity_extractor.extract_entities(question, use_llm_fallback=False)

                # Extract entities by type
                school_entity = result.get_best_entity(EntityType.SCHOOL)
                if school_entity:
                    entities['school'] = school_entity.value

                level_entity = result.get_best_entity(EntityType.DEGREE_LEVEL)
                if level_entity:
                    entities['level'] = level_entity.value

                degree_entity = result.get_best_entity(EntityType.DEGREE_TYPE)
                if degree_entity:
                    entities['degree_type'] = degree_entity.value

                # Extract search terms
                search_terms = self.entity_extractor.extract_search_terms(question)
                entities['search_term'] = ' '.join(search_terms[:2])  # Limit to 2 terms

                logger.info(f"Extracted entities using centralized extractor: {entities}")

                # If we got useful entities, return them
                if any(entities.values()):
                    return entities
            except Exception as e:
                logger.warning(f"Centralized entity extraction failed, using fallback: {e}")

        # Fallback to original logic if extractor not available or failed
        return self._extract_entities_fallback(question)

    def _extract_entities_fallback(self, question: str) -> Dict[str, str]:
        """
        Fallback entity extraction using original heuristic logic
        """
        question_lower = question.lower()
        entities = {
            'school': '',
            'level': '',
            'degree_type': '',
            'search_term': ''
        }
        
        # Extract school
        for keyword, schools in self.school_keywords.items():
            if keyword in question_lower:
                entities['school'] = schools[0]  # Use the first (most common) school name
                break

        # Fuzzy school detection if none matched
        if not entities['school']:
            try:
                question_tokens = [t for t in re.split(r"\W+", question_lower) if len(t) >= 4]
                available_schools = self.program_loader.get_schools()
                best_match: Tuple[float, Optional[str]] = (0.0, None)
                for school in available_schools:
                    school_tokens = [t for t in re.split(r"\W+", school.lower()) if t]
                    for qt in question_tokens:
                        for st in school_tokens:
                            score = difflib.SequenceMatcher(None, qt, st).ratio()
                            if score > best_match[0]:
                                best_match = (score, school)
                if best_match[0] >= 0.78 and best_match[1]:
                    entities['school'] = best_match[1]
            except Exception:
                pass
        
        # Extract level
        for level, keywords in self.level_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    entities['level'] = level.title()
                    break
            if entities['level']:
                break
        
        # Extract degree type from explicit keywords first
        if not entities['degree_type']:
            for degree_type, keywords in self.degree_type_keywords.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        entities['degree_type'] = degree_type.title()
                        break
                if entities['degree_type']:
                    break
        
        # Then infer degree type from level if still not specified
        if not entities['degree_type'] and entities['level']:
            if entities['level'].lower() == 'undergraduate':
                entities['degree_type'] = 'Bachelor'
            elif entities['level'].lower() == 'graduate':
                entities['degree_type'] = 'Master'
        
        # Extract search term (remove common words and extract meaningful terms)
        search_terms = []
        words = question_lower.split()
        
        # Remove common stop words
        stop_words = {
            'what', 'are', 'there', 'is', 'the', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'can', 'i', 'you', 'we', 'they',
            'this', 'that', 'these', 'those', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'have', 'has',
            'had', 'be', 'been', 'being', 'am', 'is', 'are', 'was', 'were',
            'show', 'me', 'list', 'find', 'search', 'available', 'programs',
            'degrees', 'majors', 'study', 'studies', 'courses', 'undergraduate',
            'graduate', 'bachelor', 'master', 'phd', 'doctorate', 'school',
            'college', 'department', 'field', 'discipline', 'subject', 'undergrads',
            'grads', 'there', 'here', 'where', 'when', 'how', 'why', 'which'
        }
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip('.,!?;:')
            if clean_word not in stop_words and len(clean_word) > 2:
                search_terms.append(clean_word)
        
        # If we have too many search terms, prioritize the most relevant ones
        if len(search_terms) > 2:
            # Keep only the first few meaningful terms
            search_terms = search_terms[:2]
        
        entities['search_term'] = ' '.join(search_terms)
        
        return entities
    
    def search_programs(self, entities: Dict[str, str]) -> List[Dict]:
        """
        Search programs based on extracted entities
        """
        programs = self.program_loader.get_all_programs()
        
        # Apply filters
        if entities['school']:
            programs = [p for p in programs if p.get('school', '').lower() == entities['school'].lower()]
        
        if entities['level']:
            programs = [p for p in programs if p.get('level', '').lower() == entities['level'].lower()]
        
        if entities['degree_type']:
            desired = entities['degree_type'].lower()
            def classify_degree_type(dt: str) -> str:
                s = (dt or '').lower()
                if ('doctor of philosophy' in s) or ('phd' in s) or ('ph.d' in s) or ('doctoral' in s):
                    return 'phd'
                if ('master' in s) or ('m.s' in s) or ('m.a' in s) or ('mba' in s):
                    return 'master'
                if ('bachelor' in s) or ('b.s' in s) or ('b.a' in s):
                    return 'bachelor'
                if 'certificate' in s or 'cert' in s:
                    return 'certificate'
                return s.strip()
            programs = [p for p in programs if classify_degree_type(p.get('degree_type', '')) == classify_degree_type(desired)]
        
        if entities['search_term']:
            search_lower = entities['search_term'].lower()
            search_tokens = [t for t in re.split(r"\W+", search_lower) if t]
            def token_match(token: str, text: str) -> bool:
                if token in text:
                    return True
                # fuzzy against each word in text
                for word in re.split(r"\W+", text):
                    if not word:
                        continue
                    if difflib.SequenceMatcher(None, token, word).ratio() >= 0.80:
                        return True
                return False
            def matches(p):
                name = p.get('program_name', '').lower()
                dept = p.get('department', '').lower()
                text = f"{name} {dept}".strip()
                return all(token_match(tok, text) for tok in search_tokens)
            filtered = [p for p in programs if matches(p)]
            # If fuzzy search yields nothing, relax search term filter
            programs = filtered if filtered else programs
        
        return programs
    
    def format_program_response(self, programs: List[Dict], entities: Dict[str, str]) -> str:
        """
        Format the list of programs into a user-friendly response
        """
        if not programs:
            return "I couldn't find any programs matching your criteria. Try adjusting your search terms or being more specific about the school or level you're interested in."
        
        response_parts = []
        
        # Add context about the search
        search_context = []
        if entities['school']:
            search_context.append(f"in {entities['school']}")
        if entities['level']:
            search_context.append(f"at {entities['level']} level")
        if entities['search_term']:
            search_context.append(f"related to '{entities['search_term']}'")
        
        if search_context:
            response_parts.append(f"Here are the programs {', '.join(search_context)}:")
        else:
            response_parts.append("Here are the programs I found:")
        
        # Add program count
        response_parts.append(f"\n**Found {len(programs)} program{'s' if len(programs) != 1 else ''}:**\n")
        
        # Add programs list (limit to first 20 for readability)
        display_programs = programs[:20]
        for i, program in enumerate(display_programs, 1):
            program_name = program.get('program_name', 'Unknown Program')
            school = program.get('school', 'Unknown School')
            level = program.get('level', 'Unknown Level')
            degree_type = program.get('degree_type', 'Unknown Degree')
            url = program.get('url', '')
            
            program_line = f"{i}. **{program_name}**"
            if school:
                program_line += f" - {school}"
            if level:
                program_line += f" ({level})"
            if degree_type:
                program_line += f" - {degree_type}"
            
            response_parts.append(program_line)
            
            if url:
                response_parts.append(f"   📎 [View Details]({url})")
            
            response_parts.append("")  # Empty line for readability
        
        if len(programs) > 20:
            response_parts.append(f"\n*... and {len(programs) - 20} more programs. Try being more specific with your search to narrow down the results.*")
        
        return '\n'.join(response_parts)
    
    def process_question(self, question: str) -> Tuple[bool, str, str]:
        """
        Process a user question and return response
        Returns: (success, response, error)
        """
        try:
            # Classify intent
            intent = self.classify_intent(question)
            
            if intent == 'general_question':
                return False, "", "This appears to be a general question. The Program Finder is designed to help you search for specific academic programs. Try asking something like 'What engineering programs are available?' or 'Show me undergraduate business programs.'"
            
            # Extract entities
            entities = self.extract_entities(question)
            
            # Search programs
            programs = self.search_programs(entities)
            
            # Format response
            response = self.format_program_response(programs, entities)
            
            return True, response, ""
            
        except Exception as e:
            return False, "", f"Error processing question: {str(e)}"
