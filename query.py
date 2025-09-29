# query.py
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

# Load
index = faiss.read_index("docs.index")
sources = np.load("sources.npy")
chunks = np.load("chunks.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    D,I = index.search(q_emb, k)
    return [(chunks[i], sources[i]) for i in I[0]]

while True:
    q = input("Ask: ")
    if q.lower() in ["exit","quit"]: break
    results = search(q)
    for chunk, src in results:
        print(f"\nFrom {src}:\n{chunk[:300]}...")
