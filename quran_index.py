# FILE: build_quran_index.py
import requests
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load Quran JSON
url = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran_en.json"
response = requests.get(url)
quran_data = response.json()

# Step 2: Prepare data and metadata
translations = []
metadata = []

for surah in quran_data:
    for verse in surah["verses"]:
        translations.append(verse["translation"])
        metadata.append({
            "surah_id": surah["id"],
            "surah_name": surah["transliteration"]+"("+surah['translation']+")",
            "verse_id": verse["id"],
            "text": verse["text"],
            "translation": verse["translation"]
        })

# Step 3: Compute embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(translations, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

# Step 4: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Step 5: Save index and metadata
faiss.write_index(index, "data/quran_faiss_index.index")
df_meta = pd.DataFrame(metadata)
df_meta.to_pickle("data/quran_verse_metadata.pkl")
print("✅ Index and metadata saved.")
