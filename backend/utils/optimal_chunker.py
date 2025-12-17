#!/usr/bin/env python3
"""
Optimal Chunking Strategy for RAG System
Creates small, focused chunks optimized for precise retrieval and concise answers
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimalChunker:
    """
    Creates optimal chunks for RAG retrieval

    Design Principles:
    - Small chunks (300-500 chars, ~75-125 tokens)
    - One semantic unit per chunk
    - Granular info types
    - Preserves context with metadata
    """

    def __init__(
        self,
        target_chunk_size: int = 400,  # characters
        max_chunk_size: int = 800,     # characters
        min_chunk_size: int = 100,     # characters
        overlap: int = 50               # character overlap between chunks
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)"""
        return len(text) // 4

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = text.replace("Ph.D.", "PhD")
        text = text.replace("M.S.", "MS")
        text = text.replace("B.S.", "BS")
        text = text.replace("e.g.", "eg")
        text = text.replace("i.e.", "ie")

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into sections based on markdown headers and common patterns
        Returns: List of (section_type, content) tuples
        """
        sections = []

        # Split by markdown headers
        header_pattern = r'^#+\s+(.+)$'
        lines = text.split('\n')

        current_section_title = "general"
        current_content = []

        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((
                            self._classify_section(current_section_title),
                            content
                        ))

                # Start new section
                current_section_title = header_match.group(1)
                current_content = [line]
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((
                    self._classify_section(current_section_title),
                    content
                ))

        return sections

    def _classify_section(self, title: str) -> str:
        """Classify section into granular info type"""
        title_lower = title.lower()

        # Admission-related
        if any(kw in title_lower for kw in ['admission', 'apply', 'application']):
            if 'requirement' in title_lower:
                return 'admission_requirements'
            elif 'deadline' in title_lower:
                return 'admission_deadlines'
            elif 'process' in title_lower or 'how to' in title_lower:
                return 'application_process'
            else:
                return 'admission_general'

        # Cost-related
        if any(kw in title_lower for kw in ['tuition', 'fee', 'cost', 'financ', 'scholarship']):
            if 'scholarship' in title_lower or 'aid' in title_lower:
                return 'financial_aid'
            else:
                return 'tuition_and_fees'

        # Requirements
        if 'requirement' in title_lower:
            if 'degree' in title_lower or 'graduation' in title_lower:
                return 'degree_requirements'
            elif 'course' in title_lower:
                return 'course_requirements'
            else:
                return 'general_requirements'

        # Courses
        if any(kw in title_lower for kw in ['course', 'curriculum', 'class']):
            if 'description' in title_lower or 'catalog' in title_lower:
                return 'course_descriptions'
            elif 'required' in title_lower:
                return 'required_courses'
            elif 'elective' in title_lower:
                return 'elective_courses'
            else:
                return 'curriculum_general'

        # Program info
        if any(kw in title_lower for kw in ['overview', 'about', 'description', 'program']):
            return 'program_overview'

        # Career
        if any(kw in title_lower for kw in ['career', 'employment', 'outcome', 'job']):
            return 'career_outcomes'

        # Faculty
        if any(kw in title_lower for kw in ['faculty', 'professor', 'instructor', 'staff']):
            return 'faculty_info'

        # Contact
        if any(kw in title_lower for kw in ['contact', 'office', 'phone', 'email']):
            return 'contact_info'

        # Default
        return 'general_info'

    def chunk_section(self, content: str, info_type: str) -> List[Dict]:
        """
        Chunk a section into optimal-sized pieces
        Returns list of chunk dictionaries
        """
        chunks = []

        # If content is already small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            return [{
                'text': content,
                'info_type': info_type,
                'tokens': self.estimate_tokens(content)
            }]

        # Split into sentences
        sentences = self.split_into_sentences(content)

        # Group sentences into chunks
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed max size, save current chunk
            if current_chunk and (current_size + sentence_size > self.max_chunk_size):
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'info_type': info_type,
                    'tokens': self.estimate_tokens(chunk_text)
                })

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    # Keep last sentence for context
                    current_chunk = [current_chunk[-1], sentence]
                    current_size = len(current_chunk[-1]) + sentence_size
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

                # If we've reached target size, save chunk
                if current_size >= self.target_chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'info_type': info_type,
                        'tokens': self.estimate_tokens(chunk_text)
                    })

                    # Start new chunk with overlap
                    if len(current_chunk) > 1:
                        current_chunk = [current_chunk[-1]]
                        current_size = len(current_chunk[-1])
                    else:
                        current_chunk = []
                        current_size = 0

        # Save remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            # Only save if it meets minimum size or is the only chunk
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append({
                    'text': chunk_text,
                    'info_type': info_type,
                    'tokens': self.estimate_tokens(chunk_text)
                })

        return chunks

    def chunk_document(
        self,
        text: str,
        program_abbreviation: str,
        department_name: str,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk entire document into optimal pieces

        Args:
            text: Full document text
            program_abbreviation: Program code (e.g., 'computer_science_ms')
            department_name: Department name
            metadata: Additional metadata to include

        Returns:
            List of chunk dictionaries ready for ingestion
        """
        all_chunks = []

        # Split into sections
        sections = self.split_by_sections(text)

        if not sections:
            # No clear sections, treat as one section
            sections = [('general_info', text)]

        # Chunk each section
        for section_type, content in sections:
            section_chunks = self.chunk_section(content, section_type)

            # Add metadata to each chunk
            for chunk in section_chunks:
                chunk_dict = {
                    'chunk_text': chunk['text'],
                    'info_type': chunk['info_type'],
                    'program_abbreviation': program_abbreviation,
                    'department_name': department_name,
                    'tokens': chunk['tokens'],
                    'char_count': len(chunk['text'])
                }

                # Add additional metadata if provided
                if metadata:
                    chunk_dict.update(metadata)

                all_chunks.append(chunk_dict)

        logger.info(
            f"Chunked document for {program_abbreviation}: "
            f"{len(all_chunks)} chunks, "
            f"avg {sum(c['tokens'] for c in all_chunks) // len(all_chunks) if all_chunks else 0} tokens/chunk"
        )

        return all_chunks

    def merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge consecutive chunks that are too small"""
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk.copy()
            elif (
                len(current['chunk_text']) < self.min_chunk_size and
                current['info_type'] == chunk['info_type']
            ):
                # Merge with current
                current['chunk_text'] += ' ' + chunk['chunk_text']
                current['tokens'] = self.estimate_tokens(current['chunk_text'])
                current['char_count'] = len(current['chunk_text'])
            else:
                # Save current and start new
                merged.append(current)
                current = chunk.copy()

        if current:
            merged.append(current)

        return merged


def test_chunker():
    """Test the chunker with sample text"""
    chunker = OptimalChunker(
        target_chunk_size=400,
        max_chunk_size=800,
        min_chunk_size=100
    )

    sample_text = """
# Computer Science MS

## Program Overview

The Master of Science in Computer Science provides advanced education in computing.
The program prepares students for careers in software development, research, and technology leadership.

## Admission Requirements

Applicants must have a bachelor's degree in Computer Science or related field.
A minimum GPA of 3.0 is required.
GRE scores are optional but recommended for international students.

## Tuition and Fees

Tuition is $450 per credit hour for Alabama residents.
Out-of-state students pay $950 per credit hour.
The program requires 30 credit hours total.

## Degree Requirements

Students must complete 30 credit hours including:
- 18 hours of core courses
- 12 hours of electives
- A thesis or comprehensive exam

Core courses include:
- CS 601 Advanced Algorithms
- CS 602 Database Systems
- CS 603 Software Engineering
"""

    chunks = chunker.chunk_document(
        text=sample_text,
        program_abbreviation='computer_science_ms',
        department_name='Computer Science'
    )

    print(f"\n{'='*80}")
    print(f"CHUNKING TEST RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Average tokens per chunk: {sum(c['tokens'] for c in chunks) // len(chunks)}")
    print(f"Average chars per chunk: {sum(c['char_count'] for c in chunks) // len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Info Type: {chunk['info_type']}")
        print(f"Tokens: {chunk['tokens']}")
        print(f"Chars: {chunk['char_count']}")
        print(f"Text: {chunk['chunk_text'][:200]}...")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_chunker()
