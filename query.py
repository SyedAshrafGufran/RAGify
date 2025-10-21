import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import textwrap

# --- Global placeholders (lazy loaded later) ---
index = None
parent_chunks = None
parent_sources = None
parent_map_indices = None
embed_model = None
tokenizer = None
model = None
generator = None


# =======================================================
# Lazy Loader - called only after chunking completes
# =======================================================
def load_models_and_indices():
    global index, parent_chunks, parent_sources, parent_map_indices
    global embed_model, tokenizer, model, generator

    print("🔄 Loading FAISS index and chunk data...")
    index = faiss.read_index("docs.index")
    parent_chunks = np.load("parent_chunks.npy", allow_pickle=True)
    parent_sources = np.load("sources.npy", allow_pickle=True)
    parent_map_indices = np.load("parent_map_indices.npy", allow_pickle=True)
    print("✅ Chunk index and mapping loaded.")

    print("🔄 Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model ready.")

    print("🔄 Loading Phi3 model...")
    GEN_MODEL = "./models/phi3"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        device_map="auto",
        dtype="auto",
        offload_folder="./models/offload"
    )
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, do_sample=False)
    print("✅ Phi3 model loaded successfully. System ready!")


# =======================================================
# Retrieval + Prompt Construction + Answer Functions
# =======================================================
def search(query, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb.astype('float32'), k)

    unique_parent_indices = set()
    results = []
    for child_index in I[0]:
        parent_idx = parent_map_indices[child_index]
        if parent_idx not in unique_parent_indices:
            unique_parent_indices.add(parent_idx)
            results.append((parent_chunks[parent_idx], parent_sources[parent_idx]))
    return results


def build_prompt(query, retrieved):
    context_blocks = "\n\n".join(
        [f"[{i+1}] ({src}) {chunk}" for i, (chunk, src) in enumerate(retrieved[:5])]
    )
    return (
        "You are an expert assistant with access to several reference documents.\n"
        "Use only the information in the context below to answer the question.\n"
        "If the context does not contain enough details, say 'I don't know.'\n"
        "Give a clear, factual answer in 2–4 sentences and cite sources like [1], [2].\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def answer_query(query):
    retrieved = search(query, k=5)
    if not retrieved:
        return "No relevant context found."

    prompt = build_prompt(query, retrieved)
    prompt_token_count = len(tokenizer(prompt)["input_ids"])
    model_max_length = getattr(tokenizer, "model_max_length", 4096)
    available_tokens = model_max_length - prompt_token_count
    max_new = max(50, min(512, available_tokens))

    output = generator(prompt, max_new_tokens=max_new, do_sample=False)[0]["generated_text"]

    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    if "Answer:" in output:
        output = output.split("Answer:", 1)[-1].strip()

    formatted_answer = textwrap.fill(output.strip(), width=90)
    sources = "\n".join([f"[{i+1}] {src}" for i, (_, src) in enumerate(retrieved[:5])])

    return f"{formatted_answer}\n\n📚 Sources used:\n{sources}"
# --- Answer Function ---
# def answer_query(query):
#     retrieved = search(query, k=5)
#     if not retrieved:
#         print("No relevant context found.")
#         return

#     prompt = build_prompt(query, retrieved)
#     output = generator(prompt, max_new_tokens=150, temperature=0.2)[0]["generated_text"]

#     # Remove repeated prompt if echoed
#     if output.startswith(prompt):
#         output = output[len(prompt):].strip()

#     print("\n" + "="*60)
#     print("Query:", query)
#     print("-"*60)
#     print(textwrap.fill(output, width=90))
#     print("\nSources used:")
#     for i, (_, src) in enumerate(retrieved[:5]):
#         print(f"[{i+1}] {src}")
#     print("="*60)
# def answer_query(query):
#     retrieved = search(query, k=5)
#     if not retrieved:
#         return "No relevant context found."

#     prompt = build_prompt(query, retrieved)
#     max_context = 4000  # adjust based on your model (phi-3 ~4k tokens)
#     max_new = max(100, min(512, max_context - len(prompt)))
#     output = generator(prompt, max_new_tokens=max_new, temperature=0.2)[0]["generated_text"]

#     # Remove repeated prompt if echoed
#     if output.startswith(prompt):
#         output = output[len(prompt):].strip()

#     # Format answer
#     formatted_answer = textwrap.fill(output.strip(), width=90)

#     # Prepare sources
#     sources = "\n".join(
#         [f"[{i+1}] {src}" for i, (_, src) in enumerate(retrieved[:5])]
#     )

#     # Return final content for GUI
#     final_response = (
#         f"{formatted_answer}\n\n"
#         f"📚 Sources used:\n{sources}"
#     )

#     return final_response


# # --- CLI Loop ---
# if __name__ == "__main__":
#     print("Hierarchical RAG + Phi3 ready. Type 'exit' to quit.")
#     while True:
#         q = input("\nAsk: ").strip()
#         if q.lower() in ["exit", "quit"]:
#             print("Exiting...")
#             break
#         if q:
#             answer_query(q)
