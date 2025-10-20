import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from Load_docs import load_folder # Assuming Load_docs.py is present
from typing import List

# --- Configuration ---
# Threshold for dynamic split (cosine similarity distance)
# A distance greater than this means a topic shift. 0.3 is a common starting point.
SEMANTIC_BREAK_THRESHOLD = 0.65

# --- Helper Function: Sentence Splitting ---
def get_sentences(text: str) -> List[str]:
    """Basic sentence splitting and cleaning."""
    text = re.sub(r'\s+', ' ', text.strip())
    # Split by common sentence delimiters, preserving the delimiter and space
    sentences = re.split(r'([.?!])\s+', text)
    
    # Re-join delimiters to sentences: ['Sentence 1', '.', 'Sentence 2', '!', ...] -> ['Sentence 1.', 'Sentence 2!', ...]
    # This loop cleans up the list and merges the split parts back together
    merged_sentences = []
    if not sentences: return merged_sentences
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if not sentence: continue

        # Append the delimiter back to the sentence if it was split
        delimiter = sentences[i+1] if i + 1 < len(sentences) else ''
        merged_sentences.append(sentence + delimiter)
    
    # Filter out any remaining empty strings
    return [s for s in merged_sentences if s]

# --- Core Function: Dynamic Semantic Chunking ---
def dynamic_semantic_chunk(text: str, model: SentenceTransformer) -> List[str]:
    """Splits text into Parent Chunks based on significant shifts in topic (vector distance)."""
    
    sentences = get_sentences(text)
    if len(sentences) <= 1:
        return sentences # Return if not enough sentences
    
    # 1. Embed all sentences (these are our Child Chunks)
    sentence_embs = model.encode(sentences, convert_to_numpy=True)
    
    # 2. Normalize embeddings for accurate cosine distance calculation
    sentence_embs = sentence_embs / np.linalg.norm(sentence_embs, axis=1, keepdims=True)
    
    # 3. Calculate Cosine Distance between adjacent sentences
    # Cosine distance = 1 - Cosine similarity (dot product)
    distances = []
    for i in range(len(sentence_embs) - 1):
        # Cosine Similarity is the dot product of normalized vectors
        similarity = np.dot(sentence_embs[i], sentence_embs[i+1])
        distances.append(1 - similarity)
        
    # 4. Identify the break points (Parent Chunk boundaries)
    break_indices = [i for i, d in enumerate(distances) if d > SEMANTIC_BREAK_THRESHOLD]
    
    # 5. Assemble the Parent Chunks
    parent_chunks = []
    start_index = 0
    
    for end_index in break_indices:
        parent_chunk = " ".join(sentences[start_index : end_index + 1])
        parent_chunks.append(parent_chunk)
        start_index = end_index + 1
        
    # Add the last chunk
    if start_index < len(sentences):
        parent_chunks.append(" ".join(sentences[start_index:]))
        
    return parent_chunks

# --- Main Indexing Logic ---

def index(path, log_fn):
    # Initialize embedding model (used here to compute semantic splits)
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load docs
    docs = load_folder(path)

    # --- Data Containers ---
    parent_chunks, parent_sources = [], []
    child_chunks, child_sources = [], [] # Child chunks are individual sentences
    parent_map_indices = []             # Maps child index to parent index

    # 1. Iterate and create dynamic Parent Chunks
    log_fn("Creating Dynamic Semantic Chunks...")
    for name, text in docs:
        # Use the dynamic splitter to create Parent Chunks
        parent_list = dynamic_semantic_chunk(text, embed_model)
        
        for parent_text in parent_list:
            parent_index = len(parent_chunks)
            
            # Save Parent Chunk (the context we will retrieve)
            parent_chunks.append(parent_text)
            parent_sources.append(name)
            
            # 2. Break Parent Chunk into Child Chunks (Sentences)
            # Child chunks are the sentences that make up the Parent Chunk
            child_list = get_sentences(parent_text)
            
            for child_text in child_list:
                # Save Child Chunk (the vector we will index)
                child_chunks.append(child_text)
                child_sources.append(name)
                parent_map_indices.append(parent_index) # Link child back to its parent

    # 3. Embed Child Chunks (Sentences)
    log_fn(f"Embedding {len(child_chunks)} Child Chunks (Sentences) for indexing...")
    embs = embed_model.encode(child_chunks, convert_to_numpy=True, show_progress_bar=True)

    # 4. Normalize embeddings (CRITICAL for accurate cosine distance search in FAISS)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    # 5. Store in FAISS (IndexFlatIP for Inner Product search, which is Cosine Similarity on normalized vectors)
    log_fn("Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype('float32'))

    # 6. Save all files
    faiss.write_index(index, "docs.index")
    np.save("parent_chunks.npy", np.array(parent_chunks, dtype=object))
    np.save("parent_map_indices.npy", np.array(parent_map_indices))
    np.save("sources.npy", np.array(parent_sources, dtype=object))

    log_fn("\n--- Indexing Complete ---")
    log_fn(f"Total Parent Chunks (Context): {len(parent_chunks)} (Dynamic Size)")
    log_fn(f"Total Child Chunks (Indexed): {len(child_chunks)} (Individual Sentences)")



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