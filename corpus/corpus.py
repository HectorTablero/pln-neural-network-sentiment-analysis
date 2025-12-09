"""
Corpus: Main entry point for managing the collection of documents.

Handles loading, filtering, partitioning, balancing, and statistical analysis
of the entire corpus.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import random
import numpy as np
from tqdm import autonotebook

from .document import Document
from .reader import CorpusReader
from .preprocessing import PreprocessingPipeline
from .linguistic_analyzer import LinguisticAnalyzer


class Corpus:
    """
    Manages the entire collection of documents.

    Provides functionality for loading, filtering, balancing, splitting,
    and analyzing the corpus.
    """

    def __init__(
        self,
        label_map: Optional[Dict[str, List[float]]] = None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        linguistic_analyzer: Optional[LinguisticAnalyzer] = None
    ):
        """
        Initialize the corpus.

        Args:
            label_map: Mapping from label names to [min_rating, max_rating].
                      Defaults to binary positive/negative classification.
            preprocessing_pipeline: Pipeline for text preprocessing
            linguistic_analyzer: Analyzer for linguistic features
        """
        # Default binary sentiment labels
        if label_map is None:
            label_map = {
                'positive': [7, 10],
                'negative': [1, 4]
            }

        self.label_map = label_map
        self.documents: Dict[str, Document] = {}

        # Initialize components
        self._preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline()
        self._linguistic_analyzer = linguistic_analyzer or LinguisticAnalyzer()

    def load(
        self,
        reader: CorpusReader,
        game_ids: Optional[List[str]] = None,
        max_documents: Optional[int] = None,
        min_text_length: int = 10,
        show_progress: bool = True
    ):
        """
        Load documents from a CorpusReader.

        Args:
            reader: CorpusReader instance
            game_ids: Optional list of game IDs to filter
            max_documents: Maximum number of documents to load
            min_text_length: Minimum text length to include document
            show_progress: Whether to show progress bar
        """
        # Get total count for progress bar
        total = reader.count_reviews(game_ids) if show_progress else None
        if max_documents and total:
            total = min(total, max_documents)

        review_stream = reader.stream_reviews(game_ids)
        if show_progress:
            review_stream = autonotebook.tqdm(review_stream, total=total,
                                              desc="Loading documents")

        loaded_count = 0

        for review_data in review_stream:
            # Check if we've reached max_documents
            if max_documents and loaded_count >= max_documents:
                break

            # Extract fields
            text = review_data.get('text', '').strip()
            rating = review_data.get('rating')

            # Skip if text is too short or rating is missing
            if not text or len(text) < min_text_length or rating is None:
                continue

            # Create document ID
            game_id = review_data.get('game_id', 'unknown')
            user = review_data.get('user') or review_data.get(
                'username', 'anonymous')
            doc_id = f"{game_id}_{user}_{loaded_count}"

            # Create metadata dict
            metadata = {
                'game_id': game_id,
                'user': user,
                'timestamp': review_data.get('timestamp'),
            }

            # Add any other fields from review_data to metadata
            for key, value in review_data.items():
                if key not in ['text', 'rating', 'game_id', 'user', 'username']:
                    metadata[key] = value

            # Create document
            doc = Document(
                doc_id=doc_id,
                raw_text=text,
                rating=float(rating),
                metadata=metadata
            )

            # Set pipeline and analyzer references
            doc.set_preprocessing_pipeline(self._preprocessing_pipeline)
            doc.set_linguistic_analyzer(self._linguistic_analyzer)

            self.documents[doc_id] = doc
            loaded_count += 1

        print(f"Loaded {len(self.documents)} documents")

    def assign_labels(self):
        """
        Assign sentiment labels to documents based on ratings and label_map.

        Documents that don't fall into any label category are left unlabeled.
        """
        for doc in self.documents.values():
            for label, (min_rating, max_rating) in self.label_map.items():
                if min_rating <= doc.rating <= max_rating:
                    doc.label = label
                    break

    def get_documents_by_label(self, label: str) -> List[Document]:
        """
        Get all documents with a specific label.

        Args:
            label: The sentiment label

        Returns:
            List of Document objects
        """
        return [doc for doc in self.documents.values() if doc.label == label]

    def balance_dataset(
        self,
        strategy: str = 'subsample',
        random_seed: int = 42
    ) -> List[Document]:
        """
        Create a balanced dataset with equal representation of each label.

        Args:
            strategy: Either 'subsample' (undersample) or 'oversample'
            random_seed: Random seed for reproducibility

        Returns:
            List of Document objects (balanced)
        """
        random.seed(random_seed)

        # Group documents by label
        label_groups = defaultdict(list)
        for doc in self.documents.values():
            if doc.label:  # Only include labeled documents
                label_groups[doc.label].append(doc)

        if not label_groups:
            raise ValueError(
                "No labeled documents found. Call assign_labels() first.")

        # Find the target size
        if strategy == 'subsample':
            # Use size of smallest class
            target_size = min(len(docs) for docs in label_groups.values())
        elif strategy == 'oversample':
            # Use size of largest class
            target_size = max(len(docs) for docs in label_groups.values())
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")

        # Balance each class
        balanced_docs = []
        for label, docs in label_groups.items():
            if len(docs) >= target_size:
                # Subsample
                balanced_docs.extend(random.sample(docs, target_size))
            else:
                # Oversample (sample with replacement)
                balanced_docs.extend(random.choices(docs, k=target_size))

        # Shuffle the result
        random.shuffle(balanced_docs)

        print(f"Balanced dataset created: {len(balanced_docs)} documents "
              f"({target_size} per class)")

        return balanced_docs

    def create_stratified_split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_seed: int = 42,
        documents: Optional[List[Document]] = None
    ) -> Tuple[List[Document], ...]:
        """
        Split corpus into train, test, and optional validation sets.

        Maintains the proportion of sentiment labels in each split.

        Args:
            test_size: Proportion of data for testing (0-1)
            validation_size: Proportion of data for validation (0-1)
            random_seed: Random seed for reproducibility
            documents: Optional list of documents to split (uses all if None)

        Returns:
            Tuple of (train_docs, test_docs) or (train_docs, val_docs, test_docs)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Use provided documents or all corpus documents
        if documents is None:
            documents = [doc for doc in self.documents.values() if doc.label]

        # Group by label
        label_groups = defaultdict(list)
        for doc in documents:
            label_groups[doc.label].append(doc)

        train_docs = []
        val_docs = []
        test_docs = []

        # Split each label group
        for label, docs in label_groups.items():
            # Shuffle
            random.shuffle(docs)

            n_docs = len(docs)
            n_test = int(n_docs * test_size)
            n_val = int(n_docs * validation_size)
            n_train = n_docs - n_test - n_val

            # Split
            test_docs.extend(docs[:n_test])
            val_docs.extend(docs[n_test:n_test + n_val])
            train_docs.extend(docs[n_test + n_val:])

        # Shuffle the splits
        random.shuffle(train_docs)
        random.shuffle(test_docs)
        random.shuffle(val_docs)

        print(
            f"Split created: {len(train_docs)} train, {len(test_docs)} test", end="")

        if validation_size > 0:
            print(f", {len(val_docs)} validation")
            return train_docs, val_docs, test_docs
        else:
            print()
            return train_docs, test_docs

    def get_statistics(self) -> Dict:
        """
        Compute comprehensive statistics about the corpus.

        Returns:
            Dictionary with various corpus statistics
        """
        stats = {
            'total_documents': len(self.documents),
            'labeled_documents': sum(1 for doc in self.documents.values() if doc.label),
        }

        # Label distribution
        label_counts = Counter(
            doc.label for doc in self.documents.values() if doc.label)
        stats['label_distribution'] = dict(label_counts)

        # Rating distribution
        rating_counts = Counter(doc.rating for doc in self.documents.values())
        stats['rating_distribution'] = dict(sorted(rating_counts.items()))

        # Text length statistics
        text_lengths = [len(doc.raw_text) for doc in self.documents.values()]
        if text_lengths:
            stats['text_length'] = {
                'mean': np.mean(text_lengths),
                'median': np.median(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'std': np.std(text_lengths),
            }

        # Average rating per label
        label_ratings = defaultdict(list)
        for doc in self.documents.values():
            if doc.label:
                label_ratings[doc.label].append(doc.rating)

        stats['avg_rating_per_label'] = {
            label: np.mean(ratings)
            for label, ratings in label_ratings.items()
        }

        # Game distribution
        game_counts = Counter(doc.game_id for doc in self.documents.values())
        stats['unique_games'] = len(game_counts)
        stats['reviews_per_game'] = {
            'mean': np.mean(list(game_counts.values())),
            'median': np.median(list(game_counts.values())),
            'min': np.min(list(game_counts.values())),
            'max': np.max(list(game_counts.values())),
        }

        return stats

    def filter_by_game_ids(self, game_ids: List[str]) -> 'Corpus':
        """
        Create a new corpus containing only documents from specified games.

        Args:
            game_ids: List of game IDs to include

        Returns:
            New Corpus instance
        """
        new_corpus = Corpus(
            label_map=self.label_map,
            preprocessing_pipeline=self._preprocessing_pipeline,
            linguistic_analyzer=self._linguistic_analyzer
        )

        for doc in self.documents.values():
            if doc.game_id in game_ids:
                new_corpus.documents[doc.doc_id] = doc

        return new_corpus

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.documents)

    def __repr__(self) -> str:
        return f"Corpus(documents={len(self.documents)}, labels={list(self.label_map.keys())})"
