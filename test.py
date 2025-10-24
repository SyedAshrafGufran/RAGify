import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI  # <-- Using local server
from typing import List, Tuple
import textwrap
import os

# --- Configuration ---
LOCAL_LLAMA_URL = "http://localhost:8000/v1"  # Your running LlamaCpp server
N_THREADS = os.cpu_count() or 4
MAX_NEW_TOKENS = 150 

# --- Global placeholders ---
index = None
parent_chunks = None
parent_sources = None
parent_map_indices = None
embed_model = None
llm = None  # will be the ChatOpenAI wrapper

# =======================================================
# Lazy Loader - called only after chunking completes
# =======================================================
def load_models_and_indices():
    global index, parent_chunks, parent_sources, parent_map_indices, child_chunks
    global embed_model, llm

    print("🔄 Loading FAISS index and chunk data...")
    try:
        index = faiss.read_index("docs.index")
        parent_chunks = np.load("parent_chunks.npy", allow_pickle=True)
        parent_sources = np.load("sources.npy", allow_pickle=True)
        parent_map_indices = np.load("parent_map_indices.npy", allow_pickle=True)
        child_chunks = np.load("child_chunks.npy", allow_pickle=True)
        print("✅ Chunk index and mapping loaded.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Missing indexing file: {e}. Did you run index_docs.py?")
        return

    print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model ready.")

    # --- Connect to local LlamaCpp server ---
    print(f"🚀 Connecting to local LlamaCpp server at {LOCAL_LLAMA_URL}...")
    llm = ChatOpenAI(
        model_name="local-llama",           # dummy name, ignored by server
        openai_api_key="EMPTY",             # dummy key
        openai_api_base=LOCAL_LLAMA_URL,    # local server URL
        temperature=0.2,
        max_tokens=MAX_NEW_TOKENS
    )
    print("✅ Local LlamaCpp server ready for inference!")


# =======================================================
# Retrieval + Prompt Construction + Answer Functions
# =======================================================
def search(query: str, k: int = 5) -> List[Tuple[str, str, str]]:
    if index is None or embed_model is None:
        print("Error: Models/Index not loaded. Run load_models_and_indices() first.")
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb.astype('float32'), k)

    unique_parent_indices = set()
    results = []

    for child_index in I[0]:
        parent_idx = parent_map_indices[child_index]

        if parent_idx not in unique_parent_indices:
            unique_parent_indices.add(parent_idx)
            parent_text = parent_chunks[parent_idx]
            source = parent_sources[parent_idx]
            child_text = child_chunks[child_index] 
            results.append((parent_text, source, child_text))
            
    return results


def build_prompt(query: str, retrieved: List[Tuple[str, str, str]]) -> str:
    context_blocks = []
    for i, (parent_chunk, src, child_chunk) in enumerate(retrieved[:5]):
        context_block = (
            f"[{i+1}] Source: ({src})\n"
            f"ANCHOR SENTENCE: {child_chunk}\n"
            f"FULL CONTEXT: {parent_chunk}" 
        )
        context_blocks.append(context_block)

    joined_context = "\n\n---\n\n".join(context_blocks)
    
    return (
        "You are an expert assistant with access to several reference documents.\n"
        "Use only the information in the context below to answer the question.\n"
        "If the context does not contain enough details, say 'I don't know.'\n"
        "Give a clear, factual answer in 2–4 sentences and cite sources like [1], [2].\n\n"
        f"Context:\n{joined_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def answer_query(query: str):
    if llm is None:
        return "Error: LLM not loaded. Please ensure loading succeeded."

    retrieved = search(query, k=5)
    if not retrieved:
        return "No relevant context found."

    prompt = build_prompt(query, retrieved)

    # --- Generate answer using local server ---
    response = llm(prompt)
    if hasattr(response, "content"):
        output = response.content
    else:
        output = response

    output = output.strip()
    formatted_answer = textwrap.fill(output, width=90)

    sources = "\n".join([f"[{i+1}] {src}" for i, (_, src, _) in enumerate(retrieved[:5])])

    return f"{formatted_answer}\n\n📚 Sources used:\n{sources}"