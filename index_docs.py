# index_docs.py
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, re
from Load_docs import load_folder

def semantic_chunk(text, max_words=200, overlap_words=50):
    """
    Semantic chunking: splits by sentences, groups into ~max_words chunks with overlap.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split by sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sent in sentences:
        words = sent.split()
        if current_words + len(words) <= max_words:
            current_chunk.append(sent)
            current_words += len(words)
        else:
            # Save chunk
            chunks.append(" ".join(current_chunk))

            # Overlap handling
            overlap = []
            if overlap_words > 0:
                overlap_words_acc = 0
                for s in reversed(current_chunk):
                    w = len(s.split())
                    if overlap_words_acc + w <= overlap_words:
                        overlap.append(s)
                        overlap_words_acc += w
                    else:
                        break
                overlap = list(reversed(overlap))

            # Start new chunk with overlap + current sentence
            current_chunk = overlap + [sent]
            current_words = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --- Main ---
# Folder path input
path = input("Please enter the folder path: ")

# Load docs
docs = load_folder(path)

chunks, sources = [], []
for name, text in docs:
    doc_chunks = semantic_chunk(text, max_words=200, overlap_words=50)
    chunks.extend(doc_chunks)
    sources.extend([name] * len(doc_chunks))

# Embed
print("Embedding chunks...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Store in FAISS
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# Save
faiss.write_index(index, "docs.index")
np.save("sources.npy", np.array(sources))
np.save("chunks.npy", np.array(chunks))

print("Index built with", len(chunks), "semantic chunks")
