from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = SentenceTransformer("all-MiniLM-L6-v2")
ENDEE = "https://as1.endee.io"
TOKEN = "cgxah1ck:hTxc6f2odYfEStH2n3iNNJTV4x5Oiw61:as1" # ← replace this
INDEX = "notes"

# 10 sample notes dataset
SAMPLE_NOTES = [
    "I had a terrible day at the office today",
    "Feeling very happy and excited about my new job",
    "The weather is so beautiful outside today",
    "I am really stressed about my exam tomorrow",
    "Had a wonderful dinner with my family tonight",
    "Feeling very tired and need some rest",
    "I went for a morning walk and felt refreshed",
    "My friend made me very angry today",
    "I bought a new phone and I love it",
    "Feeling lonely and missing my old friends"
]

@app.on_event("startup")
def startup():
    # create index
    requests.post(f"{ENDEE}/api/v1/index/create", json={
        "index_name": INDEX,
        "dimension": 384,
        "metric": "cosine"
    })
    # load 10 notes into Endee
    for i, note in enumerate(SAMPLE_NOTES):
        vector = model.encode(note).tolist()
        requests.post(f"{ENDEE}/api/v1/index/{INDEX}/insert", json={
            "id": i + 1,
            "vector": vector,
            "metadata": {"text": note}
        })
    print("✅ 10 notes loaded into Endee!")

class Note(BaseModel):
    text: str

class Query(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Smart Notes API is running!"}

@app.post("/add")
def add_note(note: Note):
    vector = model.encode(note.text).tolist()
    note_id = abs(hash(note.text)) % 1000000
    requests.post(f"{ENDEE}/api/v1/index/{INDEX}/insert", json={
        "id": note_id,
        "vector": vector,
        "metadata": {"text": note.text}
    })
    return {"status": "saved"}

@app.post("/search")
def search(query: Query):
    vector = model.encode(query.text).tolist()
    res = requests.post(f"{ENDEE}/api/v1/index/{INDEX}/search", json={
        "vector": vector,
        "top_k": 3
    })
    results = res.json().get("results", [])
    return {"results": [r["metadata"]["text"] for r in results]}
