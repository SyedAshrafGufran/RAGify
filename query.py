import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import textwrap

# --- Load FAISS Index and Chunks ---
print("Loading FAISS index and chunks...")
index = faiss.read_index("docs.index")
parent_chunks = np.load("parent_chunks.npy", allow_pickle=True)
parent_sources = np.load("sources.npy", allow_pickle=True)
parent_map_indices = np.load("parent_map_indices.npy", allow_pickle=True)

# --- Load Embedding Model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- Load Phi3 Language Model ---
GEN_MODEL = "./models/phi3"
print("Loading Phi3 language model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    device_map="auto",
    dtype="auto",
    offload_folder="./models/offload"
)
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, do_sample=False)
print("Phi3 model loaded. Ready to answer questions!")

# --- Retrieval Function ---
def search(query, k=5):
    """
    Searches using Child Chunks (sentences) and returns linked Dynamic Parent Chunks.
    """
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

# --- Build Prompt for Phi3 ---
def build_prompt(query, retrieved):
    joined = "\n\n".join(
        [f"[{i+1}] From {src}:\n{chunk}" for i, (chunk, src) in enumerate(retrieved[:5])]
    )
    prompt = (
        "You are an expert assistant. Use the provided context to answer the question accurately.\n"
        "If the context does not contain enough information, say 'I don't know'.\n"
        "Be concise (2–4 sentences) and cite sources like [1], [2].\n\n"
        "### END INSTRUCTION ###\n\n"
        f"### CONTEXT ###\n{joined}\n\n"
        f"### QUESTION ###\n{query}\n\n"
        f"### ANSWER ###\n"
    )
    return prompt

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

def answer_query(query):
    retrieved = search(query, k=5)
    if not retrieved:
        return "No relevant context found."

    prompt = build_prompt(query, retrieved)

    # --- Use tokenizer to count actual tokens ---
    prompt_token_count = len(tokenizer(prompt)["input_ids"])
    model_max_length = getattr(tokenizer, "model_max_length", 4096)  # default ~4k for Phi-3

    # Leave room for the model’s answer (new tokens)
    available_tokens = model_max_length - prompt_token_count
    max_new = max(50, min(512, available_tokens))  # between 50 and 512 safe range

    # --- Generate answer ---
    output = generator(
        prompt,
        max_new_tokens=max_new,
        do_sample=False  # deterministic for Q&A
    )[0]["generated_text"]

    # --- Remove echoed prompt if model repeats it ---
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    # --- Format final answer for console/UI ---
    formatted_answer = textwrap.fill(output.strip(), width=90)

    sources = "\n".join(
        [f"[{i+1}] {src}" for i, (_, src) in enumerate(retrieved[:5])]
    )

    final_response = (
        f"{formatted_answer}\n\n"
        f"📚 Sources used:\n{sources}"
    )

    return final_response


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
