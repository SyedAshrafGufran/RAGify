import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI  # <-- Using local server
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
    # Create a unique map of source → citation number
    unique_sources = {}
    source_to_id = {}
    for _, src, _ in retrieved:
        if src not in source_to_id:
            source_to_id[src] = len(unique_sources) + 1
            unique_sources[src] = []  # Initialize the list of chunks for this source

    # Merge all child chunks belonging to the same source
    for parent_chunk, src, child_chunk in retrieved:
        print(f"Adding child chunk for source: {src}")  # Debugging print statement
        unique_sources[src].append(child_chunk)  # Append child chunk to its source

    # Debug: Print out the chunk structure after aggregation
    print("Unique Sources and Chunks: ", unique_sources)

    # Build unique context blocks with a single consistent citation number
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
to the user’s question using ONLY the information provided below.

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



# query.py

def answer_query(query: str):
    """
    Given a query, retrieves relevant documents, generates an answer,
    and formats it with sources and their corresponding chunks.

    Args:
        query (str): The user's query/question.

    Returns:
        tuple: A tuple containing the formatted answer string and a string with sources used.
    """
    if llm is None:
        return "Error: LLM not loaded. Please ensure loading succeeded.", ""

    try:
        # Step 1: Retrieve the top 5 relevant documents based on the query
        retrieved = search(query, k=5)  # This assumes `search` is your document retrieval function
        if not retrieved:
            return "No relevant context found.", ""

        # Step 2: Prepare the query prompt using the retrieved documents
        prompt, unique_sources = build_prompt(query, retrieved)  # Assuming `build_prompt` formats your context
        response = llm.invoke(prompt)  # This invokes your LLM with the generated prompt

        # Step 3: Handle model output
        output = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Debugging (check if the sources and chunks are returned correctly)
        print("Chunks for sources: ", unique_sources)

        # Step 4: Format the sources with their chunks into a human-readable format
        sources_with_chunks = ""
        for i, (source, chunks) in enumerate(unique_sources.items()):
            chunk_texts = "\n\n".join([f"**Chunk {j+1}:**\n{chunk}" for j, chunk in enumerate(chunks)])
            sources_with_chunks += f"\n\n📖 **Source {i+1}:** {source}\n{chunk_texts}"

        # Step 5: Compose the final formatted output
        formatted_output = (
            f"{'-'*90}\n"
            f"{output}\n\n"
            f"{'-'*90}"
        )

        return formatted_output, sources_with_chunks

    except Exception as e:
        # Handle any errors during processing
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