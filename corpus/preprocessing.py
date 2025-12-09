"""
PreprocessingPipeline: Manages text cleaning and normalization operations.

Provides a configurable pipeline with pluggable components for various
preprocessing steps.
"""

import re
import string
from typing import List, Dict, Callable, Optional
from abc import ABC, abstractmethod
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Process the text and return the result."""
        pass


class HtmlTagRemover(PreprocessingStep):
    """Removes HTML tags from text."""

    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')

    def process(self, text: str) -> str:
        """Remove HTML tags."""
        return self.html_pattern.sub('', text)


class LowercaseConverter(PreprocessingStep):
    """Converts text to lowercase."""

    def process(self, text: str) -> str:
        """Convert to lowercase."""
        return text.lower()


class UrlRemover(PreprocessingStep):
    """Removes URLs from text."""

    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

    def process(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub('', text)


class ExtraWhitespaceRemover(PreprocessingStep):
    """Removes extra whitespace from text."""

    def process(self, text: str) -> str:
        """Remove extra whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()


class NumberRemover(PreprocessingStep):
    """Removes or replaces numbers in text."""

    def __init__(self, replace_with: str = ''):
        self.replace_with = replace_with
        self.number_pattern = re.compile(r'\d+')

    def process(self, text: str) -> str:
        """Remove or replace numbers."""
        return self.number_pattern.sub(self.replace_with, text)


class PunctuationHandler(PreprocessingStep):
    """Handles punctuation - either removes or separates it."""

    def __init__(self, mode: str = 'remove', keep_chars: str = ''):
        """
        Initialize punctuation handler.

        Args:
            mode: Either 'remove' or 'separate'
            keep_chars: String of punctuation characters to keep
        """
        self.mode = mode
        self.punctuation = ''.join(
            [c for c in string.punctuation if c not in keep_chars])

    def process(self, text: str) -> str:
        """Handle punctuation."""
        if self.mode == 'remove':
            # Remove punctuation
            translator = str.maketrans('', '', self.punctuation)
            return text.translate(translator)
        elif self.mode == 'separate':
            # Add spaces around punctuation
            for char in self.punctuation:
                text = text.replace(char, f' {char} ')
            return text
        return text


class StopwordRemover(PreprocessingStep):
    """Removes stopwords from text."""

    def __init__(self, language: str = 'english', custom_stopwords: Optional[List[str]] = None):
        """
        Initialize stopword remover.

        Args:
            language: Language for NLTK stopwords
            custom_stopwords: Additional stopwords to remove
        """
        self.stopwords = set(stopwords.words(
            language)) - {'not', 'no', 'nor', 'never', 'neither', 'hardly', 'scarcely', 'barely'}
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def process(self, text: str) -> str:
        """Remove stopwords."""
        tokens = word_tokenize(text)
        filtered_tokens = [
            word for word in tokens if word.lower() not in self.stopwords]
        return ' '.join(filtered_tokens)


class Stemmer(PreprocessingStep):
    """Applies stemming to reduce words to their root form."""

    def __init__(self, algorithm: str = 'porter'):
        """
        Initialize stemmer.

        Args:
            algorithm: Either 'porter' or 'snowball'
        """
        if algorithm == 'porter':
            self.stemmer = PorterStemmer()
        elif algorithm == 'snowball':
            self.stemmer = SnowballStemmer('english')
        else:
            raise ValueError(f"Unknown stemming algorithm: {algorithm}")

    def process(self, text: str) -> str:
        """Apply stemming."""
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)


class ContractionExpander(PreprocessingStep):
    """Expands contractions in English text (e.g., "don't" -> "do not")."""

    def __init__(self):
        """Initialize contraction expander with a dictionary focused on common terminations."""
        self.contractions = {
            "n't": " not",
            "'ve": " have",
            "'re": " are",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is",
            # Special cases for better accuracy
            "can't": "cannot",
            "won't": "will not",
            "let's": "let us",
        }

        # Create regex pattern for special cases first (more specific)
        special_cases = ['can\'t', 'won\'t', 'let\'s']
        special_pattern = r'\b(' + '|'.join(re.escape(c)
                                            for c in special_cases) + r')\b'

        # Pattern for suffix contractions (applied after special cases)
        suffix_contractions = ['n\'t', '\'ve',
                               '\'re', '\'ll', '\'d', '\'m', '\'s']
        suffix_pattern = '(' + '|'.join(re.escape(c)
                                        for c in suffix_contractions) + ')'

        self.special_pattern = re.compile(special_pattern, re.IGNORECASE)
        self.suffix_pattern = re.compile(suffix_pattern, re.IGNORECASE)

    def process(self, text: str) -> str:
        """Expand contractions in the text."""
        def replace_contraction(match):
            contraction = match.group(0)
            expanded = self.contractions.get(contraction.lower(), contraction)

            # Preserve original casing for first letter
            if contraction[0].isupper():
                expanded = expanded[0].upper() + expanded[1:]

            return expanded

        # First, replace special cases
        text = self.special_pattern.sub(replace_contraction, text)

        # Then, replace suffix contractions
        text = self.suffix_pattern.sub(replace_contraction, text)

        return text


class Lemmatizer(PreprocessingStep):
    """Applies lemmatization with optional POS tagging for accuracy."""

    def __init__(self, use_pos: bool = True):
        """
        Initialize lemmatizer.

        Args:
            use_pos: Whether to use POS tags for more accurate lemmatization
        """
        self.lemmatizer = WordNetLemmatizer()
        self.use_pos = use_pos

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert Treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return 'a'  # Adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # Verb
        elif treebank_tag.startswith('N'):
            return 'n'  # Noun
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'  # Default to noun

    def process(self, text: str) -> str:
        """Apply lemmatization."""
        tokens = word_tokenize(text)

        if self.use_pos:
            # Use POS tags for better lemmatization
            pos_tags = nltk.pos_tag(tokens)
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(word, self._get_wordnet_pos(pos))
                for word, pos in pos_tags
            ]
        else:
            # Simple lemmatization without POS
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(lemmatized_tokens)


class PreprocessingPipeline:
    """
    Manages and applies a series of text preprocessing operations.

    The pipeline is configurable and allows easy addition, removal,
    or reordering of preprocessing steps.
    """

    # Registry of available preprocessing steps
    STEP_REGISTRY: Dict[str, Callable] = {
        'remove_html': lambda: HtmlTagRemover(),
        'lowercase': lambda: LowercaseConverter(),
        'remove_urls': lambda: UrlRemover(),
        'remove_numbers': lambda: NumberRemover(),
        'remove_extra_whitespace': lambda: ExtraWhitespaceRemover(),
        'remove_punctuation': lambda: PunctuationHandler(mode='remove'),
        'separate_punctuation': lambda: PunctuationHandler(mode='separate'),
        'remove_stopwords': lambda: StopwordRemover(),
        'expand_contractions': lambda: ContractionExpander(),
        'stem': lambda: Stemmer(algorithm='porter'),
        'stem_snowball': lambda: Stemmer(algorithm='snowball'),
        'lemmatize': lambda: Lemmatizer(use_pos=True),
        'lemmatize_simple': lambda: Lemmatizer(use_pos=False),
    }

    def __init__(self, steps: Optional[List[str]] = None):
        """
        Initialize preprocessing pipeline.

        Args:
            steps: List of step names to execute in order.
                   If None, uses a default pipeline.
        """
        if steps is None:
            # Default pipeline
            steps = [
                'lowercase',
                'remove_html',
                'remove_urls',
                'remove_numbers',
                'remove_extra_whitespace',
                'expand_contractions',
                'stem',
                'lemmatize_simple',
                'remove_stopwords'
            ]

        self.step_names = steps
        self.steps: List[PreprocessingStep] = []

        # Build the pipeline
        for step_name in steps:
            if step_name not in self.STEP_REGISTRY:
                raise ValueError(f"Unknown preprocessing step: {step_name}")
            self.steps.append(self.STEP_REGISTRY[step_name]())

    def process(self, text: str) -> str:
        """
        Execute the configured pipeline on the given text.

        Args:
            text: Raw input text

        Returns:
            Processed text after applying all steps
        """
        for step in self.steps:
            text = step.process(text)
        return text

    def add_step(self, step_name: str, position: Optional[int] = None):
        """
        Add a preprocessing step to the pipeline.

        Args:
            step_name: Name of the step to add
            position: Position to insert (None = append to end)
        """
        if step_name not in self.STEP_REGISTRY:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

        step = self.STEP_REGISTRY[step_name]()

        if position is None:
            self.steps.append(step)
            self.step_names.append(step_name)
        else:
            self.steps.insert(position, step)
            self.step_names.insert(position, step_name)

    def remove_step(self, step_name: str):
        """Remove a preprocessing step from the pipeline."""
        if step_name in self.step_names:
            idx = self.step_names.index(step_name)
            del self.steps[idx]
            del self.step_names[idx]

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={self.step_names})"
