"""
Smart Chunking Utility for RAG
Splits text into optimal-sized chunks with overlap and sentence awareness
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


class SmartChunker:
    """
    Smart text chunker with:
    - Target chunk size (tokens)
    - Sentence boundary awareness
    - Configurable overlap
    - Metadata preservation
    """

    def __init__(
        self,
        target_chunk_size: int = 250,
        overlap_size: int = 50,
        min_chunk_size: int = 50,
        max_chunk_size: int = 500
    ):
        """
        Initialize chunker

        Args:
            target_chunk_size: Target tokens per chunk
            overlap_size: Overlap tokens between chunks
            min_chunk_size: Minimum chunk size (merge small chunks)
            max_chunk_size: Maximum chunk size (split large chunks)
        """
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Initialize token encoder
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("SmartChunker using tiktoken for token counting")
            except Exception as e:
                logger.warning(f"Failed to init tiktoken: {e}")
                self.encoder = None
        else:
            self.encoder = None
            logger.info("SmartChunker using approximate token counting")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        else:
            # Approximate: ~4 characters per token
            return max(1, len(text) // 4)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with awareness of common abbreviations

        Handles:
        - Mr., Mrs., Dr., etc.
        - Ph.D., M.S., B.S., etc.
        - U.A.B., U.S.A., etc.
        """
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Ph\.D|M\.S|B\.S|B\.A|M\.A|U\.A\.B|U\.S|etc|vs|e\.g|i\.e)'

        # Temporarily replace abbreviations
        protected_text = text
        abbrev_map = {}
        for match in re.finditer(abbreviations, text, re.IGNORECASE):
            placeholder = f"__ABBREV_{len(abbrev_map)}__"
            abbrev_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder, 1)

        # Split on sentence boundaries
        # Match: period/exclamation/question followed by space and capital letter or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected_text)

        # Restore abbreviations
        restored_sentences = []
        for sentence in sentences:
            for placeholder, original in abbrev_map.items():
                sentence = sentence.replace(placeholder, original)
            restored_sentences.append(sentence.strip())

        # Filter empty sentences
        sentences = [s for s in restored_sentences if s]

        return sentences

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dicts with text, metadata, and stats
        """
        if not text or not text.strip():
            return []

        # Default metadata
        if metadata is None:
            metadata = {}

        # Split into sentences
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        logger.debug(f"Split text into {len(sentences)} sentences")

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds max size, split it further
            if sentence_tokens > self.max_chunk_size:
                # If we have accumulated sentences, save them first
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        current_tokens,
                        chunk_id,
                        metadata
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by clauses or phrases
                sub_chunks = self._split_long_sentence(sentence, metadata, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                continue

            # Check if adding this sentence would exceed target
            if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(
                    current_chunk,
                    current_tokens,
                    chunk_id,
                    metadata
                ))
                chunk_id += 1

                # Start new chunk with overlap
                # Include last few sentences for context
                overlap_sentences, overlap_tokens = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            # Don't create tiny final chunks - merge with previous if too small
            if current_tokens < self.min_chunk_size and chunks:
                # Merge with previous chunk
                last_chunk = chunks[-1]
                last_chunk['text'] += ' ' + ' '.join(current_chunk)
                last_chunk['token_count'] = self.count_tokens(last_chunk['text'])
                logger.debug(f"Merged small final chunk with previous (tokens: {last_chunk['token_count']})")
            else:
                chunks.append(self._create_chunk(
                    current_chunk,
                    current_tokens,
                    chunk_id,
                    metadata
                ))

        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> tuple[List[str], int]:
        """Get last few sentences for overlap"""
        overlap_sentences = []
        overlap_tokens = 0

        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.overlap_size:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens

        return overlap_sentences, overlap_tokens

    def _split_long_sentence(
        self,
        sentence: str,
        metadata: Dict,
        start_chunk_id: int
    ) -> List[Dict]:
        """Split very long sentence by clauses or fixed size"""
        # Try to split by clauses (commas, semicolons)
        parts = re.split(r'[;,]\s+', sentence)

        chunks = []
        current_part = []
        current_tokens = 0

        for part in parts:
            part_tokens = self.count_tokens(part)

            if current_tokens + part_tokens > self.target_chunk_size and current_part:
                # Save current chunk
                chunk_text = ', '.join(current_part)
                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata.copy(),
                    'token_count': current_tokens,
                    'chunk_id': start_chunk_id + len(chunks),
                    'is_partial': True
                })
                current_part = [part]
                current_tokens = part_tokens
            else:
                current_part.append(part)
                current_tokens += part_tokens

        # Add final part
        if current_part:
            chunk_text = ', '.join(current_part)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata.copy(),
                'token_count': current_tokens,
                'chunk_id': start_chunk_id + len(chunks),
                'is_partial': True
            })

        logger.debug(f"Split long sentence ({self.count_tokens(sentence)} tokens) into {len(chunks)} chunks")
        return chunks

    def _create_chunk(
        self,
        sentences: List[str],
        token_count: int,
        chunk_id: int,
        metadata: Dict
    ) -> Dict:
        """Create chunk dict from sentences"""
        chunk_text = ' '.join(sentences)

        return {
            'text': chunk_text,
            'metadata': metadata.copy(),
            'token_count': token_count,
            'chunk_id': chunk_id,
            'sentence_count': len(sentences),
            'is_partial': False
        }

    def chunk_document(
        self,
        document: Dict,
        text_fields: List[str],
        metadata_fields: List[str]
    ) -> List[Dict]:
        """
        Chunk a document with multiple text fields

        Args:
            document: Document dict
            text_fields: List of fields containing text to chunk
            metadata_fields: List of fields to include as metadata

        Returns:
            List of chunks with metadata
        """
        all_chunks = []

        # Extract metadata
        metadata = {
            field: document.get(field)
            for field in metadata_fields
            if field in document
        }

        # Process each text field
        for field in text_fields:
            if field not in document:
                continue

            value = document[field]

            # Handle different value types
            if isinstance(value, str):
                text = value
            elif isinstance(value, list):
                text = ' '.join(str(item) for item in value if item)
            elif isinstance(value, dict):
                text = ' '.join(str(v) for v in value.values() if v)
            else:
                continue

            if not text or not text.strip():
                continue

            # Add field info to metadata
            field_metadata = metadata.copy()
            field_metadata['source_field'] = field

            # Chunk this field
            chunks = self.chunk_text(text, field_metadata)
            all_chunks.extend(chunks)

        logger.info(f"Chunked document: {len(all_chunks)} chunks from {len(text_fields)} fields")
        return all_chunks


def test_chunker():
    """Test the smart chunker"""
    chunker = SmartChunker(target_chunk_size=100, overlap_size=20)

    test_text = """
    The Master of Science in Electrical and Computer Engineering (MSECE) program
    prepares students for professional careers in industry, government, or academia.
    The program offers advanced coursework in signal processing, communications,
    VLSI design, computer architecture, and related areas. Students can choose
    between thesis and non-thesis options. The thesis option requires 24 credit
    hours of coursework plus 6 hours of thesis research. The non-thesis option
    requires 33 credit hours of coursework including a capstone project.
    """

    chunks = chunker.chunk_text(test_text.strip())

    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({chunk['token_count']} tokens):")
        print(f"  {chunk['text'][:100]}...")
        print()


if __name__ == "__main__":
    test_chunker()
