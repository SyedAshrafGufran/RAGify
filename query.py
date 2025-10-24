# -------------------------------------------------------------------------------------------------------
# File name: query.py
# Authors: 1. Sufiya Sarwath - 1DS22CS218, 
#          2. Supriya R - 1DS22CS223, 
#          3. Syed Ashraf Gufran - 1DS22CS229, 
#          4. Yaseen Ahmed Khan - 1DS22CS257
#
# Guide: Professor Shobana Padmanabhan
# Description: Handles FAISS-based document retrieval, semantic embedding search, prompt construction, 
#              and LLM-based answer generation.
#              
# ----------------------------------------------------------------------------------------------------------

#import section
import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI  
from typing import List, Tuple
import textwrap
import os


LOCAL_LLAMA_URL = "http://localhost:8000/v1"  
N_THREADS = os.cpu_count() or 4
MAX_NEW_TOKENS = 150 

index = None
parent_chunks = None
parent_sources = None
parent_map_indices = None
embed_model = None
llm = None  


# Loads FAISS index, embeddings, and connects to the local Llama model
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

    print(f"🚀 Connecting to local LlamaCpp server at {LOCAL_LLAMA_URL}...")
    llm = ChatOpenAI(
        model_name="local-llama",           # dummy name, ignored by server
        openai_api_key="EMPTY",             # dummy key
        openai_api_base=LOCAL_LLAMA_URL,    # local server URL
        temperature=0.2,
        max_tokens=MAX_NEW_TOKENS
    )
    print("✅ Local LlamaCpp server ready for inference!")


# Performs semantic search in FAISS to retrieve top document chunks
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


# Builds a structured prompt using retrieved context and source mapping
def build_prompt(query: str, retrieved: List[Tuple[str, str, str]]) -> str:
    unique_sources = {}
    source_to_id = {}
    for _, src, _ in retrieved:
        if src not in source_to_id:
            source_to_id[src] = len(unique_sources) + 1
            unique_sources[src] = []  

    for parent_chunk, src, child_chunk in retrieved:
        print(f"Adding child chunk for source: {src}")  
        unique_sources[src].append(child_chunk)  

    print("Unique Sources and Chunks: ", unique_sources)

    context_blocks = []
    for src, chunks in unique_sources.items():
        src_id = source_to_id[src]
        merged_context = "\n\n".join([f"ANCHOR: {chunk}" for chunk in chunks])
        context_blocks.append(
            f"[{src_id}] Source: ({src})\n{merged_context}"
        )

    joined_context = "\n\n---\n\n".join(context_blocks)

    return f"""
You are an AI research assistant. Your task is to synthesize a structured and well-written response 
to the user's question using ONLY the information provided below.

**Instructions:**
1. Be factual and concise — avoid speculation.
2. Use short, clear paragraphs and bold key terms.
3. When appropriate, break your response into:
   - **Definition / Summary**
   - **Key Features / Working**
   - **Use Cases / Applications**
4. End with a brief **summary line** or key takeaway.
**Context:**
{joined_context}

**Question:** {query}

**Answer:**
""", unique_sources


# Handles full query-answer pipeline using search, prompt build, and LLM generation
def answer_query(query: str):
    if llm is None:
        return "Error: LLM not loaded. Please ensure loading succeeded.", ""

    try:
        retrieved = search(query, k=5)
        if not retrieved:
            return "No relevant context found.", ""

        prompt, unique_sources = build_prompt(query, retrieved)
        response = llm.invoke(prompt)

        output = response.content.strip() if hasattr(response, "content") else str(response).strip()

        print("Chunks for sources: ", unique_sources)

        sources_with_chunks = ""
        for i, (source, chunks) in enumerate(unique_sources.items()):
            chunk_texts = "\n\n".join([f"**Chunk {j+1}:**\n{chunk}" for j, chunk in enumerate(chunks)])
            sources_with_chunks += f"\n\n📖 **Source {i+1}:** {source}\n{chunk_texts}"

        formatted_output = (
            f"{'-'*90}\n"
            f"{output}\n\n"
            f"{'-'*90}"
        )

        return formatted_output, sources_with_chunks

    except Exception as e:
        return f"Error: {str(e)}", ""




# # --- CLI Loop (Example Runner) ---
# if __name__ == "__main__":
#     load_models_and_indices()
    
#     # If loading failed, exit
#     if llm is None:
#         exit(1)

#     print("\nHierarchical RAG + LlamaCpp ready. Type 'exit' to quit.")
    
#     while True:
#         q = input("\nAsk: ").strip()
#         if q.lower() in ["exit", "quit"]:
#             print("Exiting...")
#             break
#         if q:
#             response = answer_query(q)
#             print("\n" + "="*90)
#             print("Query:", q)
#             print("-" * 90)
#             print(response)
#             print("="*90)