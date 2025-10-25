# -----------------------------------------------------------------------------------------
# File name: index_docs.py
# Authors: 1. Sufiya Sarwath - 1DS22CS218, 
#          2. Supriya R - 1DS22CS223, 
#          3. Syed Ashraf Gufran - 1DS22CS229, 
#          4. Yaseen Ahmed Khan - 1DS22CS257
#
# Guide: Dr Shobhana Padmanabhan
# Description: This script dynamically segments documents into semantically meaningful 
#              chunks using embeddings, then indexes them in FAISS for efficient
#              similarity search and retrieval.
# -------------------------------------------------------------------------------------------

#import section
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from Load_docs import load_folder  
from typing import List
from query import load_models_and_indices

SEMANTIC_BREAK_THRESHOLD = 0.65  


# Splits the text into cleaned and well-formed sentences
def get_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'([.?!])\s+', text)

    merged_sentences = []
    if not sentences:
        return merged_sentences

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if not sentence:
            continue
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ''
        merged_sentences.append(sentence + delimiter)

    return [s for s in merged_sentences if s]


# Dynamically chunks text into larger sections based on topic shifts
def dynamic_semantic_chunk(text: str, model: SentenceTransformer) -> List[str]:
    sentences = get_sentences(text)
    if len(sentences) <= 1:
        return sentences

    sentence_embs = model.encode(sentences, convert_to_numpy=True)
    sentence_embs = sentence_embs / np.linalg.norm(sentence_embs, axis=1, keepdims=True)

    distances = []
    for i in range(len(sentence_embs) - 1):
        similarity = np.dot(sentence_embs[i], sentence_embs[i + 1])
        distances.append(1 - similarity)

    break_indices = [i for i, d in enumerate(distances) if d > SEMANTIC_BREAK_THRESHOLD]

    parent_chunks = []
    start_index = 0
    for end_index in break_indices:
        parent_chunk = " ".join(sentences[start_index: end_index + 1])
        parent_chunks.append(parent_chunk)
        start_index = end_index + 1

    if start_index < len(sentences):
        parent_chunks.append(" ".join(sentences[start_index:]))

    return parent_chunks


# Main pipeline for loading documents, generating embeddings, and building FAISS index
def index(path, log_fn):
    log_fn("Loading embedding model...")
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_folder(path)

    parent_chunks, parent_sources = [], []
    child_chunks, child_sources = [],[]
    parent_map_indices = []

    log_fn("Creating Dynamic Semantic Chunks...")
    print("Creating Dynamic Semantic Chunks...")

    for name, text in docs:
        parent_list = dynamic_semantic_chunk(text, embed_model)

        for parent_text in parent_list:
            parent_index = len(parent_chunks)
            parent_chunks.append(parent_text)
            parent_sources.append(name)

            child_list = get_sentences(parent_text)
            for child_text in child_list:
                child_chunks.append(child_text)
                child_sources.append(name)
                parent_map_indices.append(parent_index)

    log_fn(f"Embedding {len(child_chunks)} Child Chunks for indexing...")
    print(f"Embedding {len(child_chunks)} Child Chunks for indexing...")
    embs = embed_model.encode(child_chunks, convert_to_numpy=True, show_progress_bar=True)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    log_fn("Building FAISS index (IndexFlatIP)...")
    print("Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype('float32'))

    faiss.write_index(index, "docs.index")
    np.save("parent_chunks.npy", np.array(parent_chunks, dtype=object))
    np.save("parent_map_indices.npy", np.array(parent_map_indices))
    np.save("sources.npy", np.array(parent_sources, dtype=object))
    np.save("child_chunks.npy", np.array(child_chunks, dtype=object))
    np.save("child_sources.npy", np.array(child_sources, dtype=object))

    load_models_and_indices()

    log_fn("\n--- Indexing Complete ---")
    log_fn(f"Total Parent Chunks: {len(parent_chunks)} (Dynamic Size)")
    log_fn(f"Total Child Chunks: {len(child_chunks)} (Individual Sentences)")
    log_fn("Chunking complete! You can now start querying your documents.")

    print("\n--- Indexing Complete ---")
    print(f"Total Parent Chunks: {len(parent_chunks)} (Dynamic Size)")
    print(f"Total Child Chunks: {len(child_chunks)} (Individual Sentences)")




# # Folder path input
# path = input("Please enter the folder path: ")

# # Initialize embedding model (used here to compute semantic splits)
# print("Loading embedding model...")
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load docs
# docs = load_folder(path)

# # --- Data Containers ---
# parent_chunks, parent_sources = [], []
# child_chunks, child_sources = [], [] # Child chunks are individual sentences
# parent_map_indices = []             # Maps child index to parent index

# # 1. Iterate and create dynamic Parent Chunks
# print("Creating Dynamic Semantic Chunks...")
# for name, text in docs:
#     # Use the dynamic splitter to create Parent Chunks
#     parent_list = dynamic_semantic_chunk(text, embed_model)
    
#     for parent_text in parent_list:
#         parent_index = len(parent_chunks)
        
#         # Save Parent Chunk (the context we will retrieve)
#         parent_chunks.append(parent_text)
#         parent_sources.append(name)
        
#         # 2. Break Parent Chunk into Child Chunks (Sentences)
#         # Child chunks are the sentences that make up the Parent Chunk
#         child_list = get_sentences(parent_text)
        
#         for child_text in child_list:
#             # Save Child Chunk (the vector we will index)
#             child_chunks.append(child_text)
#             child_sources.append(name)
#             parent_map_indices.append(parent_index) # Link child back to its parent

# # 3. Embed Child Chunks (Sentences)
# print(f"Embedding {len(child_chunks)} Child Chunks (Sentences) for indexing...")
# embs = embed_model.encode(child_chunks, convert_to_numpy=True, show_progress_bar=True)

# # 4. Normalize embeddings (CRITICAL for accurate cosine distance search in FAISS)
# embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

# # 5. Store in FAISS (IndexFlatIP for Inner Product search, which is Cosine Similarity on normalized vectors)
# print("Building FAISS index (IndexFlatIP)...")
# index = faiss.IndexFlatIP(embs.shape[1])
# index.add(embs.astype('float32'))

# # 6. Save all files
# faiss.write_index(index, "docs.index")
# np.save("parent_chunks.npy", np.array(parent_chunks, dtype=object))
# np.save("parent_map_indices.npy", np.array(parent_map_indices))
# np.save("sources.npy", np.array(parent_sources, dtype=object))

# print("\n--- Indexing Complete ---")
# print(f"Total Parent Chunks (Context): {len(parent_chunks)} (Dynamic Size)")
# print(f"Total Child Chunks (Indexed): {len(child_chunks)} (Individual Sentences)")