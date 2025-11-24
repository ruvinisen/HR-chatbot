import os
import hashlib
import faiss
import joblib
import numpy as np
from pypdf import PdfReader
from ollama import Client
from embed import load_embedder
import torch

# Load embedder
tokenizer, model, device = load_embedder()

def get_embeddings(texts):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype("float32")
            embeddings.append(emb)
    return embeddings

DATA_FOLDER = "data/contracts"
CACHE_FOLDER = "cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

INDEX_PATH = os.path.join(CACHE_FOLDER, "index.faiss")
EMBED_PATH = os.path.join(CACHE_FOLDER, "doc_embeddings.pkl")

def compute_data_hash(folder):
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder, filename), "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

def load_or_create_index():
    data_hash = compute_data_hash(DATA_FOLDER)

    if os.path.exists(INDEX_PATH) and os.path.exists(EMBED_PATH):
        cached = joblib.load(EMBED_PATH)
        if cached["hash"] == data_hash:
            print("✅ Loaded cached FAISS index and embeddings.")
            index = faiss.read_index(INDEX_PATH)
            return cached["chunks"], cached["sources"], index

    print("📄 Reading and embedding documents...")
    documents, sources = [], []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

            def chunk_text(text, size=500):
                words = text.split()
                return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

            chunks = chunk_text(text)
            documents.extend(chunks)
            sources.extend([filename] * len(chunks))

    embeddings = get_embeddings(documents)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    joblib.dump({
        "hash": data_hash,
        "chunks": documents,
        "sources": sources
    }, EMBED_PATH)
    faiss.write_index(index, INDEX_PATH)

    print("✅ Created and cached FAISS index.")
    return documents, sources, index

documents, sources, index = load_or_create_index()
client = Client(host="http://localhost:11434")

chat_history = []
print("\n💬 Type 'exit' to end the chat.\n")

while True:
    query = input("❓ Your question: ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    query_emb = get_embeddings([query])[0]
    print(f"Query embedding shape: {query_emb.shape}")
    print(f"FAISS index dimension: {index.d}")

    _, I = index.search(np.array([query_emb]).astype("float32"), k=3)

    top_chunks = [documents[i] for i in I[0]]
    top_sources = [sources[i] for i in I[0]]

    chunk_context = ""
    for src, chunk in zip(top_sources, top_chunks):
        chunk_context += f"[Source: {src}]\n{chunk}\n---\n"

    memory_text = "\n".join([
        f"Q: {q}\nA: {a}" for q, a in chat_history[-3:]
    ])

    prompt = f"""
You are a legal assistant. Use the context and previous Q&A to answer the current question.

Context from documents:
{chunk_context}

Chat history:
{memory_text}

Current question: {query}
Answer:"""

    response = client.chat(model="deepseek-r1:1.5b", messages=[
        {"role": "user", "content": prompt}
    ])

    answer = response['message']['content']
    print(f"\n🧠 Answer: {answer}\n")

    chat_history.append((query, answer))
