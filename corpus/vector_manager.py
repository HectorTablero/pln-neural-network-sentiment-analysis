"""
VectorManager: Handles vectorized representations of the corpus.

Creates, stores, and retrieves different vectorized representations including
TF-IDF and linguistic feature vectors.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

from .document import Document


class VectorManager:
    """
    Manages vectorized representations of documents.

    Handles TF-IDF vectorization, linguistic feature vectorization,
    and combination of different feature types.
    """

    def __init__(self):
        """Initialize the vector manager."""
        self.vectorizers: Dict[str,
                               Union[TfidfVectorizer, CountVectorizer]] = {}
        self._feature_names: Dict[str, List[str]] = {}

    def fit_tfidf_vectorizer(
        self,
        documents: List[Document],
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        use_processed_text: bool = True,
        vectorizer_name: Optional[str] = None
    ) -> TfidfVectorizer:
        """
        Fit a TF-IDF vectorizer on training documents.

        Args:
            documents: List of training documents
            ngram_range: Range of n-grams (e.g., (1,1) for unigrams, (1,2) for uni+bigrams)
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_processed_text: Whether to use preprocessed text
            vectorizer_name: Name to store vectorizer under (auto-generated if None)

        Returns:
            Fitted TfidfVectorizer
        """
        if vectorizer_name is None:
            vectorizer_name = f"tfidf_{ngram_range[0]}to{ngram_range[1]}gram"

        # Extract texts
        if use_processed_text:
            texts = [doc.get_processed_text() for doc in documents]
        else:
            texts = [doc.raw_text for doc in documents]

        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Use log scaling
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )

        vectorizer.fit(texts)

        # Store vectorizer
        self.vectorizers[vectorizer_name] = vectorizer
        self._feature_names[vectorizer_name] = vectorizer.get_feature_names_out(
        ).tolist()

        print(f"Fitted TF-IDF vectorizer '{vectorizer_name}': "
              f"{len(self._feature_names[vectorizer_name])} features")

        return vectorizer

    def transform_tfidf(
        self,
        documents: List[Document],
        ngram_range: Tuple[int, int] = (1, 1),
        use_processed_text: bool = True,
        vectorizer_name: Optional[str] = None
    ) -> csr_matrix:
        """
        Transform documents using a fitted TF-IDF vectorizer.

        Args:
            documents: List of documents to transform
            ngram_range: Range of n-grams (must match fitted vectorizer)
            use_processed_text: Whether to use preprocessed text
            vectorizer_name: Name of vectorizer to use

        Returns:
            Sparse matrix of TF-IDF features
        """
        if vectorizer_name is None:
            vectorizer_name = f"tfidf_{ngram_range[0]}to{ngram_range[1]}gram"

        if vectorizer_name not in self.vectorizers:
            raise ValueError(
                f"Vectorizer '{vectorizer_name}' not found. "
                f"Call fit_tfidf_vectorizer() first."
            )

        vectorizer = self.vectorizers[vectorizer_name]

        # Extract texts
        if use_processed_text:
            texts = [doc.get_processed_text() for doc in documents]
        else:
            texts = [doc.raw_text for doc in documents]

        # Transform
        return vectorizer.transform(texts)

    def fit_count_vectorizer(
        self,
        documents: List[Document],
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        use_processed_text: bool = True,
        vectorizer_name: Optional[str] = None
    ) -> CountVectorizer:
        """
        Fit a count vectorizer on training documents.

        Args:
            documents: List of training documents
            ngram_range: Range of n-grams
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_processed_text: Whether to use preprocessed text
            vectorizer_name: Name to store vectorizer under

        Returns:
            Fitted CountVectorizer
        """
        if vectorizer_name is None:
            vectorizer_name = f"count_{ngram_range[0]}to{ngram_range[1]}gram"

        # Extract texts
        if use_processed_text:
            texts = [doc.get_processed_text() for doc in documents]
        else:
            texts = [doc.raw_text for doc in documents]

        # Create and fit vectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )

        vectorizer.fit(texts)

        # Store vectorizer
        self.vectorizers[vectorizer_name] = vectorizer
        self._feature_names[vectorizer_name] = vectorizer.get_feature_names_out(
        ).tolist()

        print(f"Fitted count vectorizer '{vectorizer_name}': "
              f"{len(self._feature_names[vectorizer_name])} features")

        return vectorizer

    def vectorize_linguistic_features(
        self,
        feature_dicts: List[Dict],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert linguistic feature dictionaries to a numerical matrix.

        Args:
            feature_dicts: List of feature dictionaries from FeatureExtractor
            feature_names: Optional list of feature names to include (uses all if None)

        Returns:
            Tuple of (feature_matrix, feature_names_used)
        """
        if not feature_dicts:
            raise ValueError("feature_dicts cannot be empty")

        # Determine which features to use
        if feature_names is None:
            # Use all numeric features (exclude doc_id, label, etc.)
            sample_dict = feature_dicts[0]
            feature_names = [
                key for key, value in sample_dict.items()
                if isinstance(value, (int, float, bool)) and key not in ['doc_id']
            ]

        # Build feature matrix
        feature_matrix = np.zeros((len(feature_dicts), len(feature_names)))

        for i, feat_dict in enumerate(feature_dicts):
            for j, feat_name in enumerate(feature_names):
                value = feat_dict.get(feat_name, 0)
                # Convert booleans to int
                if isinstance(value, bool):
                    value = int(value)
                feature_matrix[i, j] = value

        return feature_matrix, feature_names

    def combine_features(
        self,
        *feature_matrices: Union[np.ndarray, csr_matrix]
    ) -> csr_matrix:
        """
        Horizontally stack multiple feature matrices.

        Args:
            *feature_matrices: Variable number of feature matrices to combine

        Returns:
            Combined sparse matrix
        """
        # Convert dense matrices to sparse if needed
        sparse_matrices = []
        for matrix in feature_matrices:
            if isinstance(matrix, np.ndarray):
                sparse_matrices.append(csr_matrix(matrix))
            else:
                sparse_matrices.append(matrix)

        # Horizontally stack
        combined = hstack(sparse_matrices, format='csr')

        print(
            f"Combined features: {combined.shape[0]} samples, {combined.shape[1]} features")

        return combined

    def get_feature_names(self, vectorizer_name: str) -> List[str]:
        """
        Get feature names for a specific vectorizer.

        Args:
            vectorizer_name: Name of the vectorizer

        Returns:
            List of feature names
        """
        if vectorizer_name not in self._feature_names:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")

        return self._feature_names[vectorizer_name]

    def save_vectorizer(self, vectorizer_name: str, path: str):
        """
        Save a vectorizer to disk.

        Args:
            vectorizer_name: Name of the vectorizer
            path: File path to save to
        """
        if vectorizer_name not in self.vectorizers:
            raise ValueError(f"Vectorizer '{vectorizer_name}' not found")

        with open(path, 'wb') as f:
            pickle.dump(self.vectorizers[vectorizer_name], f)

        print(f"Saved vectorizer '{vectorizer_name}' to {path}")

    def load_vectorizer(self, path: str, vectorizer_name: str):
        """
        Load a vectorizer from disk.

        Args:
            path: File path to load from
            vectorizer_name: Name to store the vectorizer under
        """
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)

        self.vectorizers[vectorizer_name] = vectorizer

        # Extract feature names
        if hasattr(vectorizer, 'get_feature_names_out'):
            self._feature_names[vectorizer_name] = vectorizer.get_feature_names_out(
            ).tolist()

        print(f"Loaded vectorizer '{vectorizer_name}' from {path}")

    def get_top_features(
        self,
        vectorizer_name: str,
        feature_vector: Union[np.ndarray, csr_matrix],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get the top-N features with highest values for a given vector.

        Args:
            vectorizer_name: Name of the vectorizer
            feature_vector: Feature vector (single document)
            top_n: Number of top features to return

        Returns:
            List of (feature_name, value) tuples
        """
        feature_names = self.get_feature_names(vectorizer_name)

        # Convert to dense if sparse
        if isinstance(feature_vector, csr_matrix):
            feature_vector = feature_vector.toarray().flatten()

        # Get top indices
        top_indices = np.argsort(feature_vector)[-top_n:][::-1]

        return [
            (feature_names[idx], feature_vector[idx])
            for idx in top_indices
        ]
