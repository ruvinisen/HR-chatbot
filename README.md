# 🧑‍💼 HR Chatbot — FAISS + RAG + Ollama

A fully local **HR Policy Chatbot** built using:

- **Python**
- **FAISS** for vector search
- **Ollama local LLMs** (deepseek-r1, llama3, qwen2.5 etc.)
- **Custom PDF RAG pipeline**
- **Torch embeddings**
- **Document hashing for caching**

This chatbot allows employees to ask questions about:

- Company Policies  
- Leave Rules  
- Contracts  
- HR Guidelines  
- Code of Conduct  
- Any PDF stored inside `data/contracts/`  

Everything runs **offline** and **fully locally**.

---

# 🚀 Features

### ✔ Local PDF ingestion  
Reads all PDFs inside `data/contracts/`.

### ✔ Automatic text chunking  
Splits documents into manageable chunks (500 words each).

### ✔ Embedding caching  
Uses MD5 hashing to detect content changes — rebuilds FAISS only when needed.

### ✔ FAISS vector database  
Fast and accurate similarity search.

### ✔ Conversational memory  
Last 3 Q&A pairs used in prompt.

### ✔ Local LLM via Ollama  
Works with:
- `deepseek-r1:1.5b`
- `llama3:8b`
- `qwen2.5:7b-instruct`
- ANY model running on `localhost:11434`.

---

# 📁 Directory Structure

```
hr-chatbot/
 ├── data/contracts/       # Add HR PDFs here
 ├── cache/                # Auto-generated cache
 ├── embed.py              # Embedding model loader
 ├── hr_chatbot.py         # Main script
 ├── requirements.txt
 └── README.md
```

---

# 🛠 Installation

### 1. Clone the repo
```
git clone https://github.com/ruvinisen/hr-chatbot.git
cd hr-chatbot
```

### 2. Install Python dependencies
```
pip install -r requirements.txt
```

### 3. Install and run Ollama  
Download from:  
https://ollama.com/download

Run your model (example):
```
ollama pull deepseek-r1:1.5b
ollama serve
```

---

# 📚 Add Your HR Documents

Put your company policies into:

```
data/contracts/
```

Examples:
- Employee Handbook.pdf  
- Leave Policy.pdf  
- Code of Conduct.pdf  
- Recruitment Guidelines.pdf  
- Payroll Policy.pdf  

The chatbot will automatically index everything.

---

# ▶ Run the Chatbot

```
python hr_chatbot.py
```

Sample:

```
❓ Your question: What is the annual leave policy?
```

---

# 👨‍💻 hr_chatbot.py (Main Script)

This script:

- Reads PDFs  
- Chunks them  
- Embeds using your HuggingFace model  
- Creates FAISS index  
- Stores caches  
- Runs interactive chat loop  

---

# 🧠 Changing the LLM Model

Edit this line:

```python
response = client.chat(model="deepseek-r1:1.5b", messages=[ ... ])
```

Replace with:

```
"llama3:8b"
"qwen2.5:7b-instruct"
"deepseek-r1:7b"
```

---

# 🏢 Use Case: HR Chatbot for a Company

Examples of questions it supports:

- "How many casual leaves do employees get?"
- "What is the work-from-home policy?"
- "How is overtime calculated?"
- "What are the rules for probation employees?"
- "Explain the employee code of conduct."

All answers come strictly from your HR PDFs.

---

# 🔒 Privacy & Security

- No data sent to external servers  
- All inference happens locally  
- Internal company use only  

---
