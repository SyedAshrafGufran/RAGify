# index_docs.py
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from Load_docs import load_folder

#Folder path input
path = input("Please enter the folder path: ")

# Load docs
docs = load_folder(path)

# Chunk simple (split every 500 chars)
chunks, sources = [], []
for name, text in docs:
    for i in range(0, len(text), 500):
        chunk = text[i:i+500]
        chunks.append(chunk)
        sources.append(name)

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode(chunks, convert_to_numpy=True)

# Store in FAISS
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# Save
faiss.write_index(index, "docs.index")
np.save("sources.npy", np.array(sources))
np.save("chunks.npy", np.array(chunks))
print("Index built with", len(chunks), "chunks")
