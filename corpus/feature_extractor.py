"""
FeatureExtractor: Implements logic for extracting linguistic features.

Extracts high-level features for sentiment analysis including opinion words,
negations, intensifiers, and domain-specific vocabulary.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import re

from .document import Document


class FeatureExtractor:
    """
    Extracts high-level linguistic features from documents.

    Implements feature extraction for sentiment analysis including:
    - Opinion word counts (positive/negative)
    - Negation detection
    - Intensifier/mitigator detection
    - Domain-specific vocabulary
    """

    def __init__(
        self,
        opinion_lexicon: Optional[str] = 'vader',
        domain_vocabulary: Optional[List[str]] = None
    ):
        """
        Initialize feature extractor.

        Args:
            opinion_lexicon: Which lexicon to use ('vader', 'sentiwordnet', or None)
            domain_vocabulary: List of domain-specific terms to track
        """
        self.opinion_lexicon_name = opinion_lexicon
        self._opinion_lexicon = None
        self._positive_words: Set[str] = set()
        self._negative_words: Set[str] = set()

        # Load opinion lexicon
        self._load_opinion_lexicon(opinion_lexicon)

        # Domain vocabulary (board game terms)
        if domain_vocabulary is None:
            domain_vocabulary = [
                'mechanics', 'mechanic', 'strategy', 'tactical', 'theme', 'themed',
                'components', 'miniatures', 'cards', 'dice', 'gameplay', 'playthrough',
                'replayability', 'replay', 'complexity', 'luck', 'skill',
                'balance', 'balanced', 'player', 'players', 'turns', 'rounds'
            ]

        self.domain_vocabulary = set(word.lower()
                                     for word in domain_vocabulary)

    def _load_opinion_lexicon(self, lexicon_name: Optional[str]):
        """Load the specified opinion lexicon."""
        if lexicon_name == 'vader':
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                import nltk

                # Download VADER lexicon if needed
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)

                self._opinion_lexicon = SentimentIntensityAnalyzer()

                # Extract positive and negative words from VADER lexicon
                lexicon = self._opinion_lexicon.lexicon
                self._positive_words = {word for word,
                                        score in lexicon.items() if score > 0}
                self._negative_words = {word for word,
                                        score in lexicon.items() if score < 0}

            except ImportError:
                print("Warning: VADER not available, using basic lexicon")
                self._load_basic_lexicon()

        elif lexicon_name == 'sentiwordnet':
            try:
                from nltk.corpus import sentiwordnet as swn
                import nltk

                # Download SentiWordNet if needed
                try:
                    nltk.data.find('corpora/sentiwordnet')
                except LookupError:
                    nltk.download('sentiwordnet', quiet=True)

                # This is simplified - full SentiWordNet usage requires POS tags
                print("Note: SentiWordNet support is basic in this implementation")
                self._load_basic_lexicon()

            except ImportError:
                print("Warning: SentiWordNet not available, using basic lexicon")
                self._load_basic_lexicon()

        else:
            # Use basic lexicon
            self._load_basic_lexicon()

    def _load_basic_lexicon(self):
        """Load a basic positive/negative word lexicon."""
        self._positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'terrific',
            'love', 'enjoy', 'fun', 'entertaining', 'engaging', 'exciting',
            'beautiful', 'perfect', 'best', 'favorite', 'recommend', 'recommended'
        }

        self._negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'boring', 'dull', 'tedious', 'disappointing', 'disappointed',
            'hate', 'dislike', 'avoid', 'waste', 'broken', 'flawed',
            'frustrating', 'annoying', 'confusing', 'complicated', 'difficult'
        }

    def extract_opinion_words(self, doc: Document) -> Dict[str, int]:
        """
        Count positive and negative opinion words.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with 'positive_count' and 'negative_count'
        """
        tokens = [token.lower() for token in doc.get_tokens(level='document')]

        positive_count = sum(
            1 for token in tokens if token in self._positive_words)
        negative_count = sum(
            1 for token in tokens if token in self._negative_words)

        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'opinion_word_ratio': (positive_count - negative_count) / max(len(tokens), 1)
        }

    def extract_negations(self, doc: Document) -> Dict[str, int]:
        """
        Identify negation patterns using dependency parse.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with negation features
        """
        dependencies = doc.get_dependencies()

        negation_count = 0
        negated_opinion_words = 0

        for sent_deps in dependencies:
            # Find negations
            negated_indices = doc._linguistic_analyzer.find_negations(
                sent_deps)
            negation_count += len(negated_indices)

            # Check if negated words are opinion words
            for idx in negated_indices:
                if idx < len(sent_deps):
                    token = sent_deps[idx]['token'].lower()
                    if token in self._positive_words or token in self._negative_words:
                        negated_opinion_words += 1

        return {
            'negation_count': negation_count,
            'has_negation': negation_count > 0,
            'negated_opinion_words': negated_opinion_words
        }

    def extract_intensifiers(self, doc: Document) -> Dict[str, int]:
        """
        Identify intensifiers and mitigators using dependency parse.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with intensifier features
        """
        dependencies = doc.get_dependencies()

        intensifier_count = 0
        mitigator_count = 0

        for sent_deps in dependencies:
            results = doc._linguistic_analyzer.find_intensifiers(sent_deps)

            for _, mod_type in results:
                if mod_type == 'intensifier':
                    intensifier_count += 1
                elif mod_type == 'mitigator':
                    mitigator_count += 1

        return {
            'intensifier_count': intensifier_count,
            'mitigator_count': mitigator_count,
            'has_intensifier': intensifier_count > 0,
            'has_mitigator': mitigator_count > 0
        }

    def extract_domain_vocabulary(self, doc: Document) -> Dict[str, int]:
        """
        Count occurrences of domain-specific vocabulary.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with domain vocabulary features
        """
        tokens = [token.lower() for token in doc.get_tokens(level='document')]

        domain_word_count = sum(
            1 for token in tokens if token in self.domain_vocabulary)

        # Get specific domain word counts
        domain_word_freq = Counter(
            token for token in tokens if token in self.domain_vocabulary)

        # Calculate proximity to opinion words (simplified)
        # This could be enhanced to look at actual word distances
        opinion_tokens = [
            token for token in tokens
            if token in self._positive_words or token in self._negative_words
        ]

        return {
            'domain_word_count': domain_word_count,
            'domain_word_ratio': domain_word_count / max(len(tokens), 1),
            'has_domain_words': domain_word_count > 0,
            'domain_opinion_overlap': len(opinion_tokens) > 0 and domain_word_count > 0
        }

    def extract_structural_features(self, doc: Document) -> Dict[str, any]:
        """
        Extract structural features of the text.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with structural features
        """
        sentences = doc.get_sentences()
        tokens = doc.get_tokens(level='document')

        return {
            'text_length': len(doc.raw_text),
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'avg_sentence_length': len(tokens) / max(len(sentences), 1),
            'rating': doc.rating
        }

    def extract_vader_scores(self, doc: Document) -> Dict[str, float]:
        """
        Get VADER compound sentiment scores (if VADER is loaded).

        Args:
            doc: Document to analyze

        Returns:
            Dictionary with VADER scores
        """
        if self._opinion_lexicon is None or self.opinion_lexicon_name != 'vader':
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0
            }

        scores = self._opinion_lexicon.polarity_scores(doc.raw_text)

        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu']
        }

    def generate_feature_dict(self, doc: Document) -> Dict[str, any]:
        """
        Extract all features for a document.

        Args:
            doc: Document to analyze

        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'doc_id': doc.doc_id,
            'label': doc.label,
        }

        # Extract all feature types
        features.update(self.extract_opinion_words(doc))
        features.update(self.extract_negations(doc))
        features.update(self.extract_intensifiers(doc))
        features.update(self.extract_domain_vocabulary(doc))
        features.update(self.extract_structural_features(doc))
        features.update(self.extract_vader_scores(doc))

        return features

    def extract_features_batch(
        self,
        documents: List[Document],
        show_progress: bool = True,
        n_jobs: int = 8
    ) -> List[Dict[str, any]]:
        """
        Extract features for multiple documents.

        Args:
            documents: List of documents to process
            show_progress: Whether to show progress bar
            n_jobs: Number of parallel jobs to run (default: 8, use 1 for sequential)

        Returns:
            List of feature dictionaries
        """
        # Sequential processing
        if n_jobs == 1:
            if show_progress:
                try:
                    from tqdm import autonotebook
                    documents = autonotebook.tqdm(
                        documents, desc="Extracting features")
                except ImportError:
                    pass
            return [self.generate_feature_dict(doc) for doc in documents]
        
        # Parallel processing
        from multiprocessing import Pool, Manager
        from functools import partial
        import threading
        
        # Shared counter for progress tracking
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        # Create a partial function with counter
        extract_func = partial(self._extract_features_wrapper, counter=counter, lock=lock)
        
        # Progress bar thread (only runs in main thread)
        pbar = None
        stop_progress = threading.Event()
        
        def update_progress():
            """Main thread function to update progress bar"""
            if show_progress:
                try:
                    from tqdm import autonotebook
                    nonlocal pbar
                    pbar = autonotebook.tqdm(total=len(documents), desc="Extracting features")
                    
                    last_value = 0
                    while not stop_progress.is_set():
                        current_value = counter.value
                        if current_value > last_value:
                            pbar.update(current_value - last_value)
                            last_value = current_value
                        stop_progress.wait(0.1)  # Check every 100ms
                    
                    # Final update
                    current_value = counter.value
                    if current_value > last_value:
                        pbar.update(current_value - last_value)
                    pbar.close()
                except ImportError:
                    pass
        
        # Start progress tracking thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        # Process documents in parallel
        with Pool(processes=n_jobs) as pool:
            chunksize = max(1, len(documents) // (n_jobs * 4))
            results = pool.map(extract_func, documents, chunksize=chunksize)
        
        # Stop progress tracking
        stop_progress.set()
        progress_thread.join()
        
        return results
    
    def _extract_features_wrapper(self, doc: Document, counter=None, lock=None) -> Dict[str, any]:
        """
        Wrapper method for parallel feature extraction.
        
        Args:
            doc: Document to process
            counter: Shared counter for progress tracking
            lock: Lock for thread-safe counter updates
            
        Returns:
            Feature dictionary
        """
        result = self.generate_feature_dict(doc)
        
        # Update counter in thread-safe manner (no tqdm in worker threads)
        if counter is not None and lock is not None:
            with lock:
                counter.value += 1
        
        return result
