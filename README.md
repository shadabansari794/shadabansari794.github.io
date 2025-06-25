# Quran Verse Semantic Search

This project provides a semantic search engine for finding similar verses in the Quran based on meaning rather than just keywords.

## Features

- Extract verses from a Quran PDF file
- Generate semantic embeddings using an open-source model
- Find similar verses using cosine similarity
- Interactive command-line search interface

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Extract verses from the PDF:

```bash
python extract_verses.py
```

3. Run the semantic search:

```bash
python quran_search.py
```

## Usage

1. When you run `quran_search.py`, you'll be prompted to enter your query.
2. Type any text or verse fragment and press Enter.
3. The system will display the top 5 most semantically similar verses.
4. Type 'exit' to quit the program.

## How It Works

The system uses the `sentence-transformers` library with the `all-MiniLM-L6-v2` model, which is a lightweight but effective open-source embedding model. This model converts text into high-dimensional vectors such that semantically similar texts have vectors close to each other.

The search process works as follows:

1. Each verse in the Quran is converted to an embedding vector
2. Your query is also converted to an embedding vector
3. Cosine similarity is calculated between your query and all verses
4. The verses with the highest similarity scores are returned

## Files

- `extract_verses.py`: Script to extract verses from the Quran PDF
- `quran_search.py`: Main script for semantic search
- `quran_verses.json`: JSON file containing extracted verses (created by extract_verses.py)
- `quran_embeddings.npy`: NumPy file containing pre-computed embeddings (created by quran_search.py)
