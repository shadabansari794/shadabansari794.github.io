# Quran Verse Semantic Search

A modern semantic search engine for exploring the Quran based on meaning rather than just keywords. This application allows users to find verses with similar semantic content using natural language queries.

## 🌟 Features

- **Semantic Search**: Find verses based on meaning using embedding vectors
- **Web Interface**: User-friendly frontend for easy interaction
- **API Backend**: FastAPI-powered backend with efficient search capabilities
- **Pre-processed Data**: Includes extracted verses and pre-computed embeddings

## 🔧 Technologies Used

- **Backend**: Python, FastAPI
- **Embeddings**: Sentence Transformers with all-MiniLM-L6-v2 model
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: PDF extraction, NumPy for vector operations

## 📋 Requirements

- Python 3.7+
- Node.js (for serving the frontend)
- Dependencies listed in `requirements.txt`

## 🚀 Getting Started

### Backend Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Process the Quran data and generate embeddings:

```bash
python quran_index.py
```

4. Start the FastAPI server:

```bash
uvicorn app:app --reload
```

### Frontend Setup

1. Install the serve package (if not already installed):

```bash
npm install -g serve
```

2. Serve the frontend files:

```bash
serve .
```

3. Open your browser and navigate to the displayed URL (typically http://localhost:3000)

## 💡 How It Works

The search system operates through these steps:

1. **Data Processing**: Each verse in the Quran is converted to an embedding vector using the Sentence Transformers model
2. **Query Processing**: User search queries are converted to the same vector space
3. **Similarity Calculation**: Cosine similarity is calculated between the query and all verses
4. **Result Ranking**: Verses with the highest similarity scores are returned as search results

## 📁 Project Structure

- `extract_verses.py`: Script to extract verses from the Quran PDF
- `quran_index.py`: Script for generating and storing verse embeddings
- `app.py`: FastAPI backend application
- `quran_verses.json`: JSON file containing extracted verses
- `quran_embeddings.npy`: NumPy file containing pre-computed embeddings
- `index.html`: Frontend application interface

## 📝 Usage

1. Enter your search query in the text box
2. View the matching verses ranked by semantic similarity
3. Explore different phrasings to discover related concepts in the Quran

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
