"""
Query Analyzer - Detects info_type and extracts keywords from queries
Production-ready query understanding for better retrieval
"""

import re
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes user queries to extract:
    - Info type (deadlines, admission_requirements, etc.)
    - Keywords for filtering
    - Query intent
    """

    # Info type patterns (ordered by specificity)
    INFO_TYPE_PATTERNS = {
        'deadlines': [
            r'\bdeadline\b',
            r'\bdue\s+date\b',
            r'\bapplication\s+deadline\b',
            r'\bwhen.*apply\b',
            r'\bwhen.*submit\b',
            r'\bapplication\s+date\b',
            r'\bsubmission\s+date\b',
            r'\bdue\s+by\b',
        ],
        'admission_requirements': [
            r'\badmission\s+requirement',
            r'\brequirement.*admission\b',
            r'\bgpa\s+requirement\b',
            r'\btest\s+score',
            r'\bprerequisite',
            r'\bhow\s+to\s+apply\b',
            r'\bapplication\s+requirement',
            r'\bwhat.*need.*apply\b',
        ],
        'tuition_and_fees': [
            r'\btuition\b',
            r'\bcost\b',
            r'\bfee[s]?\b',
            r'\bprice\b',
            r'\bhow\s+much\b',
            r'\bexpensive\b',
            r'\baffordable\b',
        ],
        'financial_aid': [
            r'\bfinancial\s+aid\b',
            r'\bscholarship',
            r'\bgrant',
            r'\bfunding\b',
            r'\bassistantship',
            r'\blow.*pay\b',
        ],
        'program_overview': [
            r'\bwhat\s+is\b',
            r'\btell.*about\b',
            r'\bdescribe\b',
            r'\boverview\b',
            r'\bexplain\b',
        ],
        'contact_info': [
            r'\bcontact\b',
            r'\bemail\b',
            r'\bphone\b',
            r'\bcall\b',
            r'\boffice\b',
        ],
    }

    # Department/Program keywords for extraction
    DEPARTMENT_KEYWORDS = [
        'civil engineering', 'computer science', 'electrical engineering',
        'mechanical engineering', 'biomedical engineering',
        'business', 'mba', 'nursing', 'public health',
        'biology', 'chemistry', 'physics', 'mathematics',
        'psychology', 'sociology', 'history', 'english',
        'materials engineering', 'engineering management',
    ]

    def __init__(self):
        # Compile regex patterns for performance
        self.compiled_patterns = {
            info_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for info_type, patterns in self.INFO_TYPE_PATTERNS.items()
        }

        logger.info("QueryAnalyzer initialized")

    def analyze(self, query: str) -> dict:
        """
        Comprehensive query analysis

        Returns:
            dict with keys:
            - info_type: detected info type or None
            - keywords: list of extracted keywords
            - confidence: confidence score (0-1)
        """
        query_lower = query.lower().strip()

        # Detect info type
        info_type, info_confidence = self.detect_info_type(query_lower)

        # Extract keywords
        keywords = self.extract_keywords(query_lower)

        result = {
            'info_type': info_type,
            'keywords': keywords,
            'confidence': info_confidence,
            'original_query': query
        }

        logger.debug(f"Query analysis: {result}")
        return result

    def detect_info_type(self, query_lower: str) -> Tuple[Optional[str], float]:
        """
        Detect info_type from query patterns

        Returns:
            (info_type, confidence) tuple
        """
        # Check each info type's patterns
        matches = []

        for info_type, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(query_lower):
                    match_count += 1

            if match_count > 0:
                # Confidence based on number of patterns matched
                confidence = min(0.5 + (match_count * 0.2), 1.0)
                matches.append((info_type, confidence, match_count))

        if not matches:
            return None, 0.0

        # Return info_type with highest confidence
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        best_match = matches[0]

        logger.info(f"Detected info_type: {best_match[0]} (confidence: {best_match[1]:.2f})")
        return best_match[0], best_match[1]

    def extract_keywords(self, query_lower: str) -> List[str]:
        """
        Extract department/program keywords from query

        Returns:
            List of matched keywords
        """
        found_keywords = []

        for keyword in self.DEPARTMENT_KEYWORDS:
            if keyword in query_lower:
                found_keywords.append(keyword)
                logger.debug(f"Found keyword: {keyword}")

        return found_keywords


# Singleton instance
_query_analyzer_instance: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get or create the global QueryAnalyzer instance"""
    global _query_analyzer_instance

    if _query_analyzer_instance is None:
        _query_analyzer_instance = QueryAnalyzer()

    return _query_analyzer_instance
