"""
PersistenceManager: Manages saving and loading of processed data artifacts.

Handles serialization of documents, vectors, and linguistic annotations
with optimized formats (CSV, Parquet) for Polars integration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pickle
import json
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

from .document import Document


class PersistenceManager:
    """
    Manages storage and retrieval of corpus artifacts.

    Uses efficient formats:
    - CSV/Parquet for tabular data (Polars-optimized)
    - NPZ for sparse matrices
    - Pickle for Python objects
    """

    def __init__(self, base_path: str):
        """
        Initialize persistence manager.

        Args:
            base_path: Root directory for data storage
        """
        self.base_path = Path(base_path)

        # Create directory structure
        self.raw_data_dir = self.base_path / "raw_data"
        self.processed_dir = self.base_path / "processed_data"
        self.vectors_dir = self.base_path / "vector_representations"
        self.splits_dir = self.base_path / "data_splits"

        # Create directories
        for dir_path in [self.raw_data_dir, self.processed_dir,
                         self.vectors_dir, self.splits_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_documents_to_csv(
        self,
        documents: List[Document],
        filename: str = "corpus_documents.csv"
    ):
        """
        Save documents to a single CSV file.

        NOTE: Saves to processed_dir, NOT raw_data_dir, to protect original data.

        Args:
            documents: List of documents to save
            filename: Output filename (default: corpus_documents.csv)
        """
        # Convert documents to list of dicts
        data = []
        for doc in documents:
            row = {
                'doc_id': doc.doc_id,
                'raw_text': doc.raw_text,
                'rating': doc.rating,
                'label': doc.label,
                'game_id': doc.game_id,
                'user': doc.user,
                'text_length': len(doc.raw_text)
            }
            # Add metadata fields
            for key, value in doc.metadata.items():
                if key not in row:
                    row[key] = value
            data.append(row)

        # Create Polars DataFrame and save to PROCESSED directory (never overwrite raw data!)
        df = pl.DataFrame(data)
        output_path = self.processed_dir / filename
        df.write_csv(output_path)

        print(f"Saved {len(documents)} documents to {output_path}")

    def load_documents_from_csv(
        self,
        filename: str = "corpus_documents.csv"
    ) -> pl.DataFrame:
        """
        Load processed documents from CSV file.

        Args:
            filename: CSV filename to load (default: corpus_documents.csv)

        Returns:
            Polars DataFrame
        """
        # Load from processed_dir (where save_documents_to_csv saves)
        file_path = self.processed_dir / filename
        df = pl.read_csv(file_path)

        print(f"Loaded {len(df)} documents from {file_path}")
        return df

    def save_sentences(
        self,
        documents: List[Document],
        filename: str = "sentences.csv"
    ):
        """
        Save sentence segmentation results to CSV.

        Args:
            documents: List of documents
            filename: Output filename
        """
        data = []

        for doc in documents:
            sentences = doc.get_sentences()
            for idx, sentence in enumerate(sentences):
                data.append({
                    'doc_id': doc.doc_id,
                    'sentence_idx': idx,
                    'text': sentence
                })

        df = pl.DataFrame(data)
        output_path = self.processed_dir / filename
        df.write_csv(output_path)

        print(f"Saved sentences to {output_path}")

    def save_pos_tags(
        self,
        documents: List[Document],
        filename: str = "pos_tags.csv"
    ):
        """
        Save POS tags to CSV.

        Args:
            documents: List of documents
            filename: Output filename
        """
        data = []

        for doc in documents:
            sentences = doc.get_sentences()
            pos_tags = doc.get_pos_tags()

            # We need to map tokens back to sentences
            # This is a simplified version
            token_idx = 0
            for sent_idx, sentence in enumerate(sentences):
                sent_tokens = doc._linguistic_analyzer.tokenize(sentence)
                for t_idx, token in enumerate(sent_tokens):
                    if token_idx < len(pos_tags):
                        _, pos = pos_tags[token_idx]
                        data.append({
                            'doc_id': doc.doc_id,
                            'sentence_idx': sent_idx,
                            'token_idx': t_idx,
                            'token': token,
                            'pos_tag': pos
                        })
                        token_idx += 1

        df = pl.DataFrame(data)
        output_path = self.processed_dir / filename
        df.write_csv(output_path)

        print(f"Saved POS tags to {output_path}")

    def save_dependencies(
        self,
        documents: List[Document],
        filename: str = "dependencies.csv"
    ):
        """
        Save dependency parses to CSV.

        Args:
            documents: List of documents
            filename: Output filename
        """
        data = []

        for doc in documents:
            dependencies = doc.get_dependencies()

            for sent_idx, sent_deps in enumerate(dependencies):
                for dep_info in sent_deps:
                    data.append({
                        'doc_id': doc.doc_id,
                        'sentence_idx': sent_idx,
                        'token_idx': dep_info['token_idx'],
                        'token': dep_info['token'],
                        'pos': dep_info.get('pos', ''),
                        'dep_label': dep_info.get('dep', ''),
                        'head_idx': dep_info.get('head_idx', -1),
                        'head': dep_info.get('head', '')
                    })

        df = pl.DataFrame(data)
        output_path = self.processed_dir / filename
        df.write_csv(output_path)

        print(f"Saved dependencies to {output_path}")

    def save_linguistic_features(
        self,
        feature_dicts: List[Dict[str, Any]],
        filename: str = "linguistic_features.csv"
    ):
        """
        Save extracted linguistic features to CSV.

        Args:
            feature_dicts: List of feature dictionaries
            filename: Output filename
        """
        df = pl.DataFrame(feature_dicts)
        output_path = self.processed_dir / filename
        df.write_csv(output_path)

        print(f"Saved linguistic features to {output_path}")

    def load_linguistic_features(
        self,
        filename: str = "linguistic_features.csv"
    ) -> pl.DataFrame:
        """
        Load linguistic features from CSV.

        Args:
            filename: CSV filename

        Returns:
            Polars DataFrame
        """
        file_path = self.processed_dir / filename
        return pl.read_csv(file_path)

    def save_sparse_matrix(
        self,
        matrix: csr_matrix,
        filename: str
    ):
        """
        Save a sparse matrix in NPZ format.

        Args:
            matrix: Sparse matrix to save
            filename: Output filename
        """
        output_path = self.vectors_dir / filename
        save_npz(output_path, matrix)

        print(f"Saved sparse matrix {matrix.shape} to {output_path}")

    def load_sparse_matrix(self, filename: str) -> csr_matrix:
        """
        Load a sparse matrix from NPZ format.

        Args:
            filename: NPZ filename

        Returns:
            Sparse matrix
        """
        file_path = self.vectors_dir / filename
        matrix = load_npz(file_path)

        print(f"Loaded sparse matrix {matrix.shape} from {file_path}")
        return matrix

    def save_dense_matrix(
        self,
        matrix: np.ndarray,
        filename: str,
        format: str = 'npy'
    ):
        """
        Save a dense matrix.

        Args:
            matrix: Dense numpy array
            filename: Output filename
            format: Either 'npy' or 'parquet'
        """
        output_path = self.vectors_dir / filename

        if format == 'npy':
            np.save(output_path, matrix)
        elif format == 'parquet':
            # Convert to DataFrame for Parquet
            df = pl.DataFrame(matrix)
            output_path = output_path.with_suffix('.parquet')
            df.write_parquet(output_path)

        print(f"Saved dense matrix {matrix.shape} to {output_path}")

    def load_dense_matrix(
        self,
        filename: str,
        format: str = 'npy'
    ) -> np.ndarray:
        """
        Load a dense matrix.

        Args:
            filename: Filename to load
            format: Either 'npy' or 'parquet'

        Returns:
            Numpy array
        """
        file_path = self.vectors_dir / filename

        if format == 'npy':
            matrix = np.load(file_path)
        elif format == 'parquet':
            file_path = file_path.with_suffix('.parquet')
            df = pl.read_parquet(file_path)
            matrix = df.to_numpy()

        print(f"Loaded dense matrix {matrix.shape} from {file_path}")
        return matrix

    def save_data_split(
        self,
        train_docs: List[Document],
        test_docs: List[Document],
        validation_docs: Optional[List[Document]] = None,
        filename: str = "splits.csv"
    ):
        """
        Save data split information to CSV.

        Args:
            train_docs: Training documents
            test_docs: Test documents
            validation_docs: Optional validation documents
            filename: Output filename
        """
        data = []

        for doc in train_docs:
            data.append(
                {'doc_id': doc.doc_id, 'set': 'train', 'label': doc.label})

        for doc in test_docs:
            data.append(
                {'doc_id': doc.doc_id, 'set': 'test', 'label': doc.label})

        if validation_docs:
            for doc in validation_docs:
                data.append(
                    {'doc_id': doc.doc_id, 'set': 'validation', 'label': doc.label})

        df = pl.DataFrame(data)
        output_path = self.splits_dir / filename
        df.write_csv(output_path)

        print(f"Saved data split to {output_path}")

    def load_data_split(self, filename: str = "splits.csv") -> pl.DataFrame:
        """
        Load data split information.

        Args:
            filename: CSV filename

        Returns:
            Polars DataFrame
        """
        file_path = self.splits_dir / filename
        return pl.read_csv(file_path)

    def save(self, data: Any, path: Union[str, Path]):
        """
        Generic save method using pickle.

        Args:
            data: Any Python object
            path: File path (can be relative to base_path or absolute)
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.base_path / path

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved data to {path}")

    def load(self, path: Union[str, Path]) -> Any:
        """
        Generic load method using pickle.

        Args:
            path: File path (can be relative to base_path or absolute)

        Returns:
            Loaded Python object
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.base_path / path

        with open(path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded data from {path}")
        return data

    def get_path_for_artifact(
        self,
        doc_id: str,
        analysis_type: str
    ) -> Path:
        """
        Generate a standardized file path for a document artifact.

        Args:
            doc_id: Document identifier
            analysis_type: Type of analysis (e.g., 'pos_tags', 'dependencies')

        Returns:
            Path object
        """
        safe_doc_id = doc_id.replace('/', '_').replace('\\', '_')
        subdir = self.processed_dir / analysis_type
        subdir.mkdir(exist_ok=True)

        return subdir / f"{safe_doc_id}.json"
