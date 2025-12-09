"""
CorpusReader: Handles loading of raw data from the file system.

This module abstracts away file format details and provides a clean interface
for streaming review data and accessing game metadata.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
import polars as pl


class CorpusReader:
    """
    Handles the initial loading of raw data from the file system.

    Supports both CSV and JSON formats, with automatic format detection.
    Provides memory-efficient streaming of reviews via generators.
    """

    def __init__(self, corpus_path: str):
        """
        Initialize the corpus reader.

        Args:
            corpus_path: Path to the root directory containing raw BGG data
        """
        self.corpus_path = Path(corpus_path)
        self._reviews_file = None
        self._metadata_cache = {}

        # Detect data format
        self._detect_format()

    def _detect_format(self):
        """Detect whether data is in CSV or JSON format."""
        # Look for reviews.csv first (preferred format)
        csv_path = self.corpus_path / "reviews.csv"
        if csv_path.exists():
            self._reviews_file = csv_path
            self._format = "csv"
            return

        # Look for JSON files
        json_files = list(self.corpus_path.glob("*.json"))
        if json_files:
            self._format = "json"
            return

        # Check for nested structure (game_id folders)
        subdirs = [d for d in self.corpus_path.iterdir() if d.is_dir()]
        if subdirs:
            self._format = "nested_json"
            return

        raise ValueError(f"No valid data found in {self.corpus_path}")

    def stream_reviews(self, game_ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Generator function that yields raw review data one by one.

        This is memory-efficient for large corpora.

        Args:
            game_ids: Optional list of game IDs to filter by

        Yields:
            Dictionary containing review data with keys like:
            - text: The review text
            - rating: Numerical rating (1-10)
            - user: Username
            - timestamp: Review timestamp
            - game_id: Board game ID
        """
        if self._format == "csv":
            yield from self._stream_csv(game_ids)
        elif self._format == "json":
            yield from self._stream_json(game_ids)
        elif self._format == "nested_json":
            yield from self._stream_nested_json(game_ids)

    def _stream_csv(self, game_ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Stream reviews from a single CSV file using Polars for efficiency."""
        # Use Polars for efficient reading
        try:
            df = pl.read_csv(self._reviews_file, schema_overrides={
                             'user_id': pl.Utf8})
        except Exception as e:
            df = pl.read_csv(self._reviews_file)

        # Standardize column names (handle different CSV formats)
        column_mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}

        # Map text column (boardgames_reviews_clean.csv uses 'comment')
        for text_col in ['text', 'comment', 'review', 'content']:
            if text_col in columns_lower:
                column_mapping[columns_lower[text_col]] = 'text'
                break

        # Map rating column (boardgames_reviews_clean.csv uses 'rating')
        for rating_col in ['rating', 'value', 'score']:
            if rating_col in columns_lower:
                column_mapping[columns_lower[rating_col]] = 'rating'
                break

        # Map game_id column (boardgames_reviews_clean.csv uses 'game_id')
        for game_col in ['game_id', 'gameid', 'id', 'bgg_id']:
            if game_col in columns_lower:
                column_mapping[columns_lower[game_col]] = 'game_id'
                break

        # Map user column (boardgames_reviews_clean.csv uses 'user_id')
        for user_col in ['user', 'username', 'user_name', 'name', 'user_id', 'userid']:
            if user_col in columns_lower:
                column_mapping[columns_lower[user_col]] = 'user'
                break

        # Rename columns if mapping exists
        if column_mapping:
            df = df.rename(column_mapping)

        # Filter by game_ids if specified
        if game_ids and 'game_id' in df.columns:
            df = df.filter(pl.col("game_id").is_in(game_ids))

        # Convert to dictionaries and yield
        for row in df.iter_rows(named=True):
            yield row

    def _stream_json(self, game_ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Stream reviews from JSON files in the root directory."""
        json_files = self.corpus_path.glob("*.json")

        for json_file in json_files:
            # Try to extract game_id from filename
            game_id = json_file.stem

            if game_ids and game_id not in game_ids:
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    for review in data:
                        review['game_id'] = game_id
                        yield review
                elif isinstance(data, dict):
                    if 'reviews' in data:
                        for review in data['reviews']:
                            review['game_id'] = game_id
                            yield review
                    else:
                        data['game_id'] = game_id
                        yield data

    def _stream_nested_json(self, game_ids: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Stream reviews from nested directory structure (game_id/reviews.json)."""
        subdirs = [d for d in self.corpus_path.iterdir() if d.is_dir()]

        for game_dir in subdirs:
            game_id = game_dir.name

            if game_ids and game_id not in game_ids:
                continue

            reviews_file = game_dir / "reviews.json"
            if not reviews_file.exists():
                continue

            with open(reviews_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    for review in data:
                        review['game_id'] = game_id
                        yield review
                elif isinstance(data, dict) and 'reviews' in data:
                    for review in data['reviews']:
                        review['game_id'] = game_id
                        yield review

    def get_game_metadata(self, game_id: str) -> Dict[str, Any]:
        """
        Returns metadata for a specific game.

        Args:
            game_id: The board game ID

        Returns:
            Dictionary containing game metadata (categories, mechanics, etc.)
        """
        # Check cache first
        if game_id in self._metadata_cache:
            return self._metadata_cache[game_id]

        # Look for metadata file
        metadata_file = self.corpus_path / f"{game_id}_metadata.json"
        if not metadata_file.exists():
            # Try nested structure
            metadata_file = self.corpus_path / game_id / "metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self._metadata_cache[game_id] = metadata
                return metadata

        # Return empty metadata if not found
        return {
            'game_id': game_id,
            'categories': [],
            'mechanics': [],
            'name': None
        }

    def get_all_game_ids(self) -> List[str]:
        """
        Returns a list of all game IDs available in the raw data.

        Returns:
            List of game ID strings
        """
        if self._format == "csv":
            df = pl.read_csv(self._reviews_file, columns=["game_id"])
            return df["game_id"].unique().to_list()

        elif self._format == "json":
            json_files = self.corpus_path.glob("*.json")
            return [f.stem for f in json_files if not f.stem.endswith("_metadata")]

        elif self._format == "nested_json":
            subdirs = [d for d in self.corpus_path.iterdir() if d.is_dir()]
            return [d.name for d in subdirs]

        return []

    def count_reviews(self, game_ids: Optional[List[str]] = None) -> int:
        """
        Count total number of reviews, optionally filtered by game IDs.

        Args:
            game_ids: Optional list of game IDs to filter by

        Returns:
            Total count of reviews
        """
        if self._format == "csv":
            df = pl.read_csv(self._reviews_file, columns=["game_id"])
            if game_ids:
                df = df.filter(pl.col("game_id").is_in(game_ids))
            return len(df)

        # For JSON formats, count by streaming
        count = 0
        for _ in self.stream_reviews(game_ids):
            count += 1
        return count
