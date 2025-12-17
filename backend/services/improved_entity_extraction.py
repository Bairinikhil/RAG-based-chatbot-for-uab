#!/usr/bin/env python3
"""
Improved Entity Extraction with Program Name Matching and Degree Level Detection
"""

import re
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImprovedEntityExtractor:
    """Enhanced entity extraction specifically for program name and degree level detection"""

    # Comprehensive program mapping
    PROGRAM_PATTERNS = {
        # Computer Science
        ('computer_science_ms', 'Computer Science'): [
            r'computer science\s+(ms|master)',
            r'ms\s+computer science',
            r'master.*computer science',
            r'mscs',
            r'\bcs\s+ms\b',
            r'\bcs\s+master',
        ],
        ('computer_science_phd', 'Computer Science'): [
            r'computer science\s+(phd|doctoral|doctorate)',
            r'phd.*computer science',
            r'doctoral.*computer science',
        ],

        # Electrical and Computer Engineering (must check BEFORE Computer Science!)
        ('electrical_and_computer_engineering_msece', 'Electrical and Computer Engineering'): [
            r'electrical\s+and\s+computer\s+engineering\s+(ms|master)',
            r'msece',
            r'\bece\s+ms\b',
            r'\bece\s+master',
            r'master.*electrical.*computer.*engineering',
        ],
        ('electrical_and_computer_engineering_phd', 'Electrical and Computer Engineering'): [
            r'electrical\s+and\s+computer\s+engineering\s+(phd|doctoral)',
            r'ece\s+phd',
            r'phd.*electrical.*computer.*engineering',
        ],

        # MBA
        ('mba', 'Business'): [
            r'\bmba\b',
            r'master.*business\s+administration',
            r'business\s+administration\s+master',
        ],

        # Civil Engineering
        ('civil_engineering_msce', 'Civil Engineering'): [
            r'civil\s+engineering\s+(ms|master)',
            r'msce',
            r'master.*civil\s+engineering',
        ],
        ('civil_engineering_phd', 'Civil Engineering'): [
            r'civil\s+engineering\s+(phd|doctoral)',
            r'phd.*civil\s+engineering',
        ],

        # Mechanical Engineering
        ('mechanical_engineering_msme', 'Mechanical Engineering'): [
            r'mechanical\s+engineering\s+(ms|master)',
            r'msme',
            r'master.*mechanical\s+engineering',
        ],
        ('mechanical_engineering_phd', 'Mechanical Engineering'): [
            r'mechanical\s+engineering\s+(phd|doctoral)',
            r'phd.*mechanical\s+engineering',
        ],

        # Biomedical Engineering
        ('biomedical_engineering_msbme', 'Biomedical Engineering'): [
            r'biomedical\s+engineering\s+(ms|master)',
            r'msbme',
            r'\bbme\s+ms\b',
            r'master.*biomedical\s+engineering',
        ],
        ('biomedical_engineering_phd', 'Biomedical Engineering'): [
            r'biomedical\s+engineering\s+(phd|doctoral)',
            r'bme\s+phd',
            r'phd.*biomedical\s+engineering',
        ],
    }

    # Department-only patterns (no specific program)
    DEPARTMENT_PATTERNS = {
        'Computer Science': [
            r'\bcomputer science\b',
            r'\bcs\s+department\b',
        ],
        'Electrical and Computer Engineering': [
            r'electrical\s+and\s+computer\s+engineering',
            r'\bece\s+department\b',
        ],
        'Business': [
            r'business\s+school',
            r'collat',
        ],
        'Civil Engineering': [
            r'civil\s+engineering',
        ],
        'Mechanical Engineering': [
            r'mechanical\s+engineering',
        ],
        'Biomedical Engineering': [
            r'biomedical\s+engineering',
            r'\bbme\b',
        ],
    }

    # Degree level patterns
    DEGREE_LEVEL_PATTERNS = {
        'ms': [r'\bms\b', r'\bm\.s\.\b', r'master', r'masters'],
        'phd': [r'\bphd\b', r'\bph\.d\.\b', r'doctoral', r'doctorate'],
    }

    # Info type patterns
    INFO_TYPE_PATTERNS = {
        'admission_requirements': [r'admission', r'apply', r'application'],
        'degree_requirements': [r'requirement', r'requirements', r'plan of study', r'credit', r'course'],
        'tuition_and_fees': [r'tuition', r'fee', r'cost', r'price', r'how much'],
        'deadlines': [r'deadline', r'due date'],
        'contact_info': [r'contact', r'email', r'phone'],
        'overview': [r'overview', r'about', r'description', r'what is', r'tell me about'],
    }

    @classmethod
    def extract_program_and_degree(cls, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract program, department, and degree level from query

        Returns:
            (program_abbreviation, department_name, degree_level)
        """
        query_lower = query.lower()

        # Step 1: Try to match specific program + degree patterns (most specific)
        for (prog_abbr, dept_name), patterns in cls.PROGRAM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Extract degree level from program_abbreviation
                    if '_ms' in prog_abbr or prog_abbr == 'mba' or 'msce' in prog_abbr or 'msme' in prog_abbr or 'msbme' in prog_abbr:
                        degree_level = 'ms'
                    elif '_phd' in prog_abbr:
                        degree_level = 'phd'
                    else:
                        degree_level = None

                    logger.info(f"Matched program: {prog_abbr}, dept: {dept_name}, degree: {degree_level}")
                    return prog_abbr, dept_name, degree_level

        # Step 2: If no specific program, try department + degree level
        matched_dept = None
        for dept_name, patterns in cls.DEPARTMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matched_dept = dept_name
                    break
            if matched_dept:
                break

        # Extract degree level separately
        matched_degree = None
        for degree, patterns in cls.DEGREE_LEVEL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matched_degree = degree
                    break
            if matched_degree:
                break

        if matched_dept:
            logger.info(f"Matched department: {matched_dept}, degree: {matched_degree}")
            return None, matched_dept, matched_degree

        return None, None, None

    @classmethod
    def extract_info_type(cls, query: str) -> Optional[str]:
        """Extract info type from query"""
        query_lower = query.lower()

        for info_type, patterns in cls.INFO_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return info_type

        return None

    @classmethod
    def extract_all(cls, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Complete extraction: program, department, and info type

        Returns:
            (department_name, program_abbreviation, info_type)
        """
        prog_abbr, dept_name, degree_level = cls.extract_program_and_degree(query)
        info_type = cls.extract_info_type(query)

        return dept_name, prog_abbr, info_type


def enhanced_extract_entities(question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Enhanced entity extraction function

    Returns:
        (department_name, program_abbreviation, info_type)
    """
    return ImprovedEntityExtractor.extract_all(question)


# Test cases
if __name__ == '__main__':
    test_cases = [
        "What are the requirements for Computer Science MS?",
        "Masters in computer science",
        "Computer Science PhD requirements",
        "Electrical and Computer Engineering masters program",
        "How much does MBA cost?",
        "Tell me about Civil Engineering MS",
        "Mechanical Engineering MS program",
        "What is Biomedical Engineering MS?",
    ]

    print("="*80)
    print("TESTING IMPROVED ENTITY EXTRACTION")
    print("="*80)

    for query in test_cases:
        dept, prog, info = enhanced_extract_entities(query)
        print(f"\nQuery: {query}")
        print(f"  Department: {dept}")
        print(f"  Program: {prog}")
        print(f"  Info Type: {info}")
