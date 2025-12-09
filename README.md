**Project Overview**

This repository contains a set of experiments and a small corpus management package for sentiment/opinion analysis on BoardGameGeek (BGG) reviews. The project includes three main experiments:

-   `pln_p3_7461_01_e1.py`: Classical neural approaches using pre-trained word embeddings (FNN and Bi-LSTM RNN).
-   `pln_p3_7461_01_e2.py`: Fine-tuning a BERT model for sequence classification (Hugging Face / PyTorch).
-   `pln_p3_7461_01_e3.py`: Opinion summarization / prompting pipeline (supports Ollama local LLM).

The `corpus/` package implements utilities for reading data, preprocessing pipelines, document abstractions, linguistic analysis and vector management.

**Repository Layout**

-   `pln_p3_7461_01_e1.py`, `pln_p3_7461_01_e2.py`, `pln_p3_7461_01_e3.py`: Experiment entrypoints.
-   `corpus/`: Python package with modules:
    -   `document.py`, `corpus.py`, `reader.py`, `preprocessing.py`, `linguistic_analyzer.py`, `feature_extractor.py`, `vector_manager.py`, `persistence.py`
-   `data/processed_data/`: CSV files for `train_set.csv`, `validation_set.csv`, `test_set.csv` used by the experiments.
-   `vector_representations/`: Saved numpy compressed arrays (.npz) used for baseline/vector experiments.
-   `results/`: Text reports and saved models (e.g., `bert_model/`).

**Key Features**

-   Configurable preprocessing pipeline (`PreprocessingPipeline` in `corpus/preprocessing.py`).
-   Document-level abstraction with caching and helper methods (`corpus/document.py`).
-   Support for both classical (gensim embeddings + Keras/TensorFlow) and transformer-based (Hugging Face BERT + PyTorch) approaches.
-   Opinion summarization using prompt generation with optional local LLM via Ollama.

**Requirements (suggested)**

The project uses Python and several packages. Below is a conservative list to create a working environment. Pin versions as needed for reproducibility.

Recommended (example):

```
python>=3.8
numpy
polars
scikit-learn
gensim
tqdm
nltk
tensorflow
torch
transformers
sentencepiece  # if using some HF models
ollama  # optional, only for local LLM generation in e3
```

Install in a virtual environment (example):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1  ; # PowerShell
pip install -U pip
pip install numpy polars scikit-learn gensim tqdm nltk tensorflow torch transformers sentencepiece
# optional: pip install ollama
```

Notes:

-   `corpus/preprocessing.py` will attempt to download required NLTK data automatically (stopwords, wordnet, punkt, tagger). If downloads fail, run:

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

**Data**

-   Place or confirm processed CSV files exist in `data/processed_data/` named `train_set.csv`, `validation_set.csv`, and `test_set.csv`. Each script expects those files.
-   Precomputed vector representations (npz) are in `vector_representations/` and may be used by custom code.

**How to run each experiment**

-   FNN / Bi-LSTM (word-embeddings, TensorFlow/Keras):

    -   `python pln_p3_7461_01_e1.py`
    -   This script downloads/loads the `glove-wiki-gigaword-100` embeddings via `gensim.downloader` if not present.

-   BERT fine-tuning (Hugging Face, PyTorch):

    -   `python pln_p3_7461_01_e2.py`
    -   Uses `bert-base-uncased` by default; the first run downloads the model weights automatically unless a local `results/bert_model/` exists.
    -   Uses GPU if available; otherwise runs on CPU (slower).

-   Opinion summarization (prompting, optional Ollama):
    -   `python pln_p3_7461_01_e3.py`
    -   Attempts to use Ollama locally (if installed) to generate summaries. If Ollama is absent, it prints a warning and still writes the prompts and metadata to `results/e3_opinion_summary_results.txt`.

**Outputs**

-   Each script writes a results file into the `results/` folder, e.g. `e1_neural_networks_results.txt`, `e2_bert_results.txt`, `e3_opinion_summary_results.txt`.
-   The BERT experiment saves the fine-tuned model and tokenizer to `results/bert_model/`.
