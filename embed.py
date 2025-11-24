from transformers import AutoTokenizer, AutoModel
import torch

def load_embedder():
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("✅ Loaded from cache without downloading.")
    return tokenizer, model, device
