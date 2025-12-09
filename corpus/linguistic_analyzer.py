"""
LinguisticAnalyzer: Encapsulates advanced linguistic analysis.

Wraps NLP libraries like NLTK and spaCy for POS tagging, dependency parsing,
and sentence segmentation.
"""

from typing import Dict, List, Tuple, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# Try to import spaCy (optional for enhanced dependency parsing)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class LinguisticAnalyzer:
    """
    Handles advanced linguistic analysis tasks.

    Provides POS tagging, dependency parsing, and sentence segmentation
    with support for multiple backends (NLTK, spaCy).
    """

    def __init__(self, backend: str = 'spacy', spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the linguistic analyzer.

        Args:
            backend: Either 'nltk' or 'spacy'
            spacy_model: Name of spaCy model to load (if using spaCy)
        """
        self.backend = backend
        self._nlp = None

        if backend == 'spacy':
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is not installed. Install with: pip install spacy && "
                    f"python -m spacy download {spacy_model}"
                )
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Download with: python -m spacy download {spacy_model}"
                )

        # Ensure NLTK data is available
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        required_data = [
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ]

        for data_path, package_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                nltk.download(package_name, quiet=True)

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentence strings
        """
        if self.backend == 'spacy' and self._nlp:
            doc = self._nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            return sent_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words and punctuation.

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        if self.backend == 'spacy' and self._nlp:
            doc = self._nlp(text)
            return [token.text for token in doc]
        else:
            return word_tokenize(text)

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Perform part-of-speech tagging on tokens.

        Args:
            tokens: List of token strings

        Returns:
            List of (token, pos_tag) tuples
        """
        if self.backend == 'spacy' and self._nlp:
            # Join tokens and process with spaCy
            text = ' '.join(tokens)
            doc = self._nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            return pos_tag(tokens)

    def dependency_parse(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Parse the dependency structure of a sentence.

        Args:
            sentence: Input sentence string

        Returns:
            List of dictionaries with dependency information:
            [{'token': str, 'pos': str, 'dep': str, 'head': int}, ...]
        """
        if self.backend == 'spacy' and self._nlp:
            doc = self._nlp(sentence)
            dependencies = []

            for token in doc:
                dependencies.append({
                    'token': token.text,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'head': token.head.text,
                    'head_idx': token.head.i,
                    'token_idx': token.i,
                    'lemma': token.lemma_,
                })

            return dependencies
        else:
            # NLTK doesn't have built-in dependency parsing
            # Fall back to basic structure
            tokens = self.tokenize(sentence)
            pos_tags = self.pos_tag(tokens)

            # Create a simple structure (without true dependency relations)
            dependencies = []
            for idx, (token, pos) in enumerate(pos_tags):
                dependencies.append({
                    'token': token,
                    'pos': pos,
                    'dep': 'unknown',  # NLTK doesn't provide dependency labels
                    'head': None,
                    'head_idx': None,
                    'token_idx': idx,
                    'lemma': token.lower(),
                })

            return dependencies

    def get_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text.

        Args:
            text: Input text

        Returns:
            List of noun phrase strings
        """
        if self.backend == 'spacy' and self._nlp:
            doc = self._nlp(text)
            return [chunk.text for chunk in doc.noun_chunks]
        else:
            # Simple chunking with NLTK
            tokens = self.tokenize(text)
            pos_tags = self.pos_tag(tokens)

            # Define a simple noun phrase grammar
            grammar = r"""
                NP: {<DT>?<JJ>*<NN.*>+}  # Noun phrase
            """
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(pos_tags)

            noun_phrases = []
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    noun_phrases.append(
                        ' '.join(word for word, tag in subtree.leaves()))

            return noun_phrases

    def find_negations(self, dependencies: List[Dict[str, Any]]) -> List[int]:
        """
        Find negation patterns in dependency parse.

        Args:
            dependencies: Output from dependency_parse()

        Returns:
            List of token indices that are negated
        """
        negated_indices = []

        for dep in dependencies:
            # Look for negation dependency relation
            if dep['dep'] == 'neg':
                # The head of the negation is the negated token
                negated_indices.append(dep['head_idx'])

        return negated_indices

    def find_intensifiers(self, dependencies: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """
        Find intensifiers and mitigators in dependency parse.

        Args:
            dependencies: Output from dependency_parse()

        Returns:
            List of (token_idx, type) tuples where type is 'intensifier' or 'mitigator'
        """
        intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'highly', 'exceptionally', 'remarkably'
        }

        mitigators = {
            'somewhat', 'fairly', 'rather', 'quite', 'pretty',
            'sort of', 'kind of', 'a bit', 'slightly', 'moderately'
        }

        results = []

        for dep in dependencies:
            token_lower = dep['token'].lower()

            # Check if token is an intensifier/mitigator
            if dep['dep'] in ['advmod', 'amod']:  # Adverbial or adjectival modifier
                if token_lower in intensifiers:
                    results.append((dep['token_idx'], 'intensifier'))
                elif token_lower in mitigators:
                    results.append((dep['token_idx'], 'mitigator'))

        return results

    def __repr__(self) -> str:
        return f"LinguisticAnalyzer(backend='{self.backend}')"
