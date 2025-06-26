# FILE: app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List, Optional

# Load index and metadata only (no model needed)
index = faiss.read_index("data/quran_faiss_index.index")
df_meta = pd.read_pickle("data/quran_verse_metadata.pkl")

# Create a mapping of verse IDs to their index in the dataframe for quick lookup
verse_id_map = {}
for i, row in df_meta.iterrows():
    verse_id = f"{row['surah_id']}:{row['verse_id']}"
    verse_id_map[verse_id] = i

# Extract unique surah names
unique_surahs = df_meta[['surah_id', 'surah_name']].drop_duplicates().sort_values('surah_id').to_dict('records')

app = FastAPI()

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    verse_id: str  # Format like "2:255" for Surah 2, Verse 255
    top_k: int = 5
    same_surah_only: bool = False
    different_surah_only: bool = False

@app.post("/search")
def search(request: SearchRequest):
    # Find the verse by ID
    verse_id = request.verse_id
    if verse_id not in verse_id_map:
        raise HTTPException(status_code=404, detail=f"Verse ID {verse_id} not found")
    
    # Get the index of the verse
    verse_idx = verse_id_map[verse_id]
    
    # Get the verse data
    verse_data = df_meta.iloc[verse_idx]
    verse_surah_id = verse_data['surah_id']
    
    # Get the pre-computed embedding
    query_vector = np.array([index.reconstruct(verse_idx)])
    
    # Search for similar verses - get more than needed to allow for filtering
    search_k = request.top_k * 3 if request.same_surah_only or request.different_surah_only else request.top_k + 1
    D, I = index.search(query_vector, search_k)
    
    results = []
    for i, idx in enumerate(I[0]):
        # Skip the verse itself (it will have perfect similarity)
        if idx == verse_idx:
            continue
            
        result_data = df_meta.iloc[idx]
        result_surah_id = result_data['surah_id']
        
        # Apply the surah filters
        if request.same_surah_only and result_surah_id != verse_surah_id:
            continue
        if request.different_surah_only and result_surah_id == verse_surah_id:
            continue
            
        result = result_data.to_dict()
        result['score'] = float(D[0][i])
        results.append(result)
        
        # Stop once we have top_k results
        if len(results) >= request.top_k:
            break
            
    return results

@app.get("/surahs")
def list_surahs():
    """Return a list of all surahs"""
    return unique_surahs

@app.get("/verses")
def list_verses(surah_id: Optional[int] = None):
    """Return a list of verses, optionally filtered by surah_id"""
    # Filter by surah if requested
    filtered_df = df_meta[df_meta['surah_id'] == surah_id] if surah_id is not None else df_meta
    
    verses = []
    for i, row in filtered_df.iterrows():
        verses.append({
            "id": f"{row['surah_id']}:{row['verse_id']}",
            "surah_id": int(row['surah_id']),
            "verse_id": int(row['verse_id']),
            "surah_name": row['surah_name'],
            "text": row['text'],
            "translation": row['translation'],
            "text_preview": row['translation']
        })
    return verses