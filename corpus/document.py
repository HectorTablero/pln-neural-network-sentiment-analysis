"""
Document: Central data structure representing a single review.

Provides a clean interface to access raw content and its various
processed and annotated versions with intelligent caching.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class Document:
    """
    Represents a single review with its raw content and processed versions.

    This class acts as a central hub for all document-level data, providing
    lazy loading and caching of expensive operations.
    """

    def __init__(
        self,
        doc_id: str,
        raw_text: str,
        rating: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a document.

        Args:
            doc_id: Unique identifier (e.g., "game123_user456")
            raw_text: Original, unprocessed review text
            rating: Numerical rating (1-10)
            metadata: Additional data (user, timestamp, game_id, etc.)
        """
        self.doc_id = doc_id
        self.raw_text = raw_text
        self.rating = rating
        self.metadata = metadata or {}

        # Sentiment label (assigned by Corpus)
        self.label: Optional[str] = None

        # Internal cache for expensive operations
        self._cache: Dict[str, Any] = {}

        # References to pipeline components (set by Corpus)
        self._preprocessing_pipeline = None
        self._linguistic_analyzer = None

    @property
    def game_id(self) -> Optional[str]:
        """Get the game ID from metadata."""
        return self.metadata.get('game_id')

    @property
    def user(self) -> Optional[str]:
        """Get the username from metadata."""
        return self.metadata.get('user') or self.metadata.get('username')

    @property
    def timestamp(self) -> Optional[datetime]:
        """Get the timestamp from metadata."""
        ts = self.metadata.get('timestamp')
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return None
        return ts

    def set_preprocessing_pipeline(self, pipeline):
        """Set the preprocessing pipeline for this document."""
        self._preprocessing_pipeline = pipeline

    def set_linguistic_analyzer(self, analyzer):
        """Set the linguistic analyzer for this document."""
        self._linguistic_analyzer = analyzer

    def get_processed_text(self) -> str:
        """
        Get the preprocessed text.

        Returns:
            Processed text after applying preprocessing pipeline
        """
        cache_key = 'processed_text'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._preprocessing_pipeline is None:
            raise RuntimeError(
                "Preprocessing pipeline not set for this document")

        processed = self._preprocessing_pipeline.process(self.raw_text)
        self._cache[cache_key] = processed
        return processed

    def get_sentences(self) -> List[str]:
        """
        Get a list of sentences from the text.

        Returns:
            List of sentence strings
        """
        cache_key = 'sentences'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._linguistic_analyzer is None:
            raise RuntimeError("Linguistic analyzer not set for this document")

        sentences = self._linguistic_analyzer.segment_sentences(self.raw_text)
        self._cache[cache_key] = sentences
        return sentences

    def get_tokens(self, level: str = 'document') -> List[Any]:
        """
        Get tokens (words, punctuation).

        Args:
            level: Either 'document' (flat list) or 'sentence' (list of lists)

        Returns:
            List of tokens or list of lists of tokens
        """
        cache_key = f'tokens_{level}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._linguistic_analyzer is None:
            raise RuntimeError("Linguistic analyzer not set for this document")

        if level == 'document':
            tokens = self._linguistic_analyzer.tokenize(self.raw_text)
        elif level == 'sentence':
            sentences = self.get_sentences()
            tokens = [self._linguistic_analyzer.tokenize(
                sent) for sent in sentences]
        else:
            raise ValueError(
                f"Invalid level: {level}. Use 'document' or 'sentence'")

        self._cache[cache_key] = tokens
        return tokens

    def get_pos_tags(self) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for tokens.

        Returns:
            List of (token, pos_tag) tuples
        """
        cache_key = 'pos_tags'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._linguistic_analyzer is None:
            raise RuntimeError("Linguistic analyzer not set for this document")

        tokens = self.get_tokens(level='document')
        pos_tags = self._linguistic_analyzer.pos_tag(tokens)
        self._cache[cache_key] = pos_tags
        return pos_tags

    def get_dependencies(self) -> List[List[Tuple]]:
        """
        Get dependency parse trees for each sentence.

        Returns:
            List of dependency parses (one per sentence)
        """
        cache_key = 'dependencies'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._linguistic_analyzer is None:
            raise RuntimeError("Linguistic analyzer not set for this document")

        sentences = self.get_sentences()
        dependencies = [
            self._linguistic_analyzer.dependency_parse(sent)
            for sent in sentences
        ]
        self._cache[cache_key] = dependencies
        return dependencies

    def clear_cache(self):
        """Clear all cached processed data."""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, rating={self.rating}, label={self.label})"

    def __len__(self) -> int:
        """Return the length of the raw text."""
        return len(self.raw_text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to a dictionary for serialization.

        Returns:
            Dictionary representation of the document
        """
        return {
            'doc_id': self.doc_id,
            'raw_text': self.raw_text,
            'rating': self.rating,
            'label': self.label,
            'metadata': self.metadata,
            'text_length': len(self.raw_text)
        }
