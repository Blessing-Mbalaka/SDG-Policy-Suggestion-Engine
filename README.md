# Policy Recommendation Engine

A modular prototype for turning unstructured public feedback into policy intelligence.

The project is intentionally dependency-light so it can run in a fresh Python 3.12
virtual environment. The default pipeline uses deterministic standard-library
implementations that can later be swapped for transformer embeddings, BERTopic,
spaCy, vector databases, or a web backend.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .
```

## Run The Demo

```powershell
.\.venv\Scripts\python -m policy_recommendation_engine.cli --demo
```

## Run The Upload GUI

```powershell
.\scripts\run_gui.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

The GUI supports pasted text plus `.txt`, `.md`, and `.csv` uploads. CSV files
should include a `text` column and can optionally include `source`, `author`,
and `timestamp` columns.

The GUI includes four analysis modes:

- Fast local NLP: dependency-light tokenizer, hashed embeddings, and emotion lexicons.
- spaCy NLP: real spaCy tokenization, lemmatization, sentence splitting, and named entities.
- BERT semantic analysis: real SentenceTransformer embeddings for semantic theme grouping.
- spaCy + BERT: spaCy preprocessing with transformer embeddings.

Install the real NLP dependencies before using spaCy or BERT modes:

```powershell
.\.venv\Scripts\python -m pip install -e ".[nlp]"
.\.venv\Scripts\python -m spacy download en_core_web_sm
```

To verify the real NLP modes:

```powershell
.\.venv\Scripts\python scripts\smoke_nlp.py
```

## Run Tests

```powershell
.\scripts\run_tests.ps1
```

or:

```cmd
scripts\run_tests.bat
```

## Project Layout

- `src/policy_recommendation_engine/ingestion.py` normalizes supported text-like inputs.
- `src/policy_recommendation_engine/preprocessing.py` cleans and segments documents.
- `src/policy_recommendation_engine/embeddings.py` creates deterministic hashed vectors.
- `src/policy_recommendation_engine/themes.py` groups semantically similar documents.
- `src/policy_recommendation_engine/emotions.py` detects emotion signals and intensity.
- `src/policy_recommendation_engine/policy_gap.py` compares public themes with policy priorities.
- `src/policy_recommendation_engine/trends.py` summarizes theme movement over time.
- `src/policy_recommendation_engine/insights.py` creates human-readable intelligence notes.
- `src/policy_recommendation_engine/pipeline.py` wires the modules together.
