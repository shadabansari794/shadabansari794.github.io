# FILE: app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List, Optional, Dict, Any
import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool executor for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Load index and metadata only (no model needed)
index = faiss.read_index("quran_faiss_index.index")
df_meta = pd.read_pickle("quran_verse_metadata.pkl")

# Create a mapping of verse IDs to their index in the dataframe for quick lookup
verse_id_map = {}
for i, row in df_meta.iterrows():
    verse_id = f"{row['surah_id']}:{row['verse_id']}"
    verse_id_map[verse_id] = i

# Extract unique surah names
unique_surahs = df_meta[['surah_id', 'surah_name']].drop_duplicates().sort_values('surah_id').to_dict('records')

app = FastAPI(title="Quran Verse Semantic Search API",
              description="API for finding semantically similar verses in the Quran",
              version="1.0.0")

# Global variables for simple rate limiting
request_counts = {}  # IP -> count
rate_limit_window = 60  # seconds

# Add middleware for request timing, logging, and rate limiting
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Simple rate limiting (only for /search endpoint)
    if request.url.path == "/search":
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean up old entries
        for ip in list(request_counts.keys()):
            if current_time - request_counts[ip]["timestamp"] > rate_limit_window:
                del request_counts[ip]
        
        # Check rate limit
        if client_ip in request_counts:
            if request_counts[client_ip]["count"] > 30:  # 30 requests per minute limit
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests. Please try again later."}
                )
            request_counts[client_ip]["count"] += 1
        else:
            request_counts[client_ip] = {"count": 1, "timestamp": current_time}
    
    # Process time tracking
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        # Global exception handler
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error", 
                "detail": str(e),
                "process_time": process_time
            }
        )

# Enable CORS for frontend testing - ensuring it works properly across all environments
origins = [
    "*",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5500",
    "https://shadabansari794.github.io",
    "https://shadabansari794-github-io-1.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["X-Requested-With", "X-HTTP-Method-Override", "Content-Type", "Accept", "Authorization"],
    expose_headers=["X-Process-Time", "X-API-Revision"],
    max_age=600,  # 10 minutes cache for preflight requests
)

# Add health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check that verifies database connectivity"""
    try:
        # Check if index is loaded properly
        index_dimension = index.d
        # Check if dataframe is loaded properly
        df_row_count = len(df_meta)
        # Check thread pool
        thread_pool_info = {
            "max_workers": thread_pool._max_workers,
            "is_shutdown": thread_pool._shutdown
        }
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "index": {"dimension": index_dimension},
            "dataframe": {"row_count": df_row_count},
            "thread_pool": thread_pool_info
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )

class SearchRequest(BaseModel):
    verse_id: str  # Format like "2:255" for Surah 2, Verse 255
    top_k: int = 5
    same_surah_only: bool = False
    different_surah_only: bool = False

# Function to run in thread pool (CPU-intensive operations)
def run_faiss_search(verse_idx, top_k, same_surah_only, different_surah_only):
    # Get the verse data
    verse_data = df_meta.iloc[verse_idx]
    verse_surah_id = verse_data['surah_id']
    
    # Get the pre-computed embedding
    query_vector = np.array([index.reconstruct(verse_idx)])
    
    # Search for similar verses - get more than needed to allow for filtering
    search_k = top_k * 3 if same_surah_only or different_surah_only else top_k + 1
    D, I = index.search(query_vector, search_k)
    
    return D, I, verse_surah_id

# Process results from FAISS search
def process_search_results(D, I, verse_idx, verse_surah_id, top_k, same_surah_only, different_surah_only):
    results = []
    for i, idx in enumerate(I[0]):
        # Skip the verse itself (it will have perfect similarity)
        if idx == verse_idx:
            continue
            
        result_data = df_meta.iloc[idx]
        result_surah_id = result_data['surah_id']
        
        # Apply the surah filters
        if same_surah_only and result_surah_id != verse_surah_id:
            continue
        if different_surah_only and result_surah_id == verse_surah_id:
            continue
            
        result = result_data.to_dict()
        result['score'] = float(D[0][i])
        results.append(result)
        
        # Stop once we have top_k results
        if len(results) >= top_k:
            break
            
    return results

@app.post("/search")
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    # Find the verse by ID
    verse_id = request.verse_id
    if verse_id not in verse_id_map:
        raise HTTPException(status_code=404, detail=f"Verse ID {verse_id} not found")
    
    # Get the index of the verse
    verse_idx = verse_id_map[verse_id]
    
    # Run the FAISS search in the thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    D, I, verse_surah_id = await loop.run_in_executor(
        thread_pool, 
        functools.partial(
            run_faiss_search, 
            verse_idx, 
            request.top_k, 
            request.same_surah_only, 
            request.different_surah_only
        )
    )
    
    # Process the search results
    results = process_search_results(
        D, I, verse_idx, verse_surah_id, 
        request.top_k, request.same_surah_only, request.different_surah_only
    )
            
    return results

@app.get("/surahs")
async def list_surahs():
    """Return a list of all surahs"""
    return unique_surahs

# Function to process verse data in thread pool
def process_verses_data(surah_id=None):
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

@app.get("/verses")
async def list_verses(surah_id: Optional[int] = None):
    """Return a list of verses, optionally filtered by surah_id"""
    # Run data processing in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    verses = await loop.run_in_executor(
        thread_pool,
        functools.partial(process_verses_data, surah_id)
    )
    return verses