# Conversational RAG Chatbot 
This project implements a **modern Retrieval-Augmented Generation (RAG) chatbot**
using the latest **LangChain LCEL (Runnable)** APIs with **chat history support**.
The chatbot:
- Retrieves answers from real documentation
- Rewrites follow-up questions using chat history
- Prevents hallucinations outside retrieved context
- Uses **Groq LLMs** for fast inference
---
## Features

- Modern LangChain 
- History-aware retrieval
- Web-based document ingestion
- Chroma vector store
- HuggingFace embeddings
- Multi-turn conversational loop
- Clean & extensible architecture

---

##  Architecture Overview
User Query + Chat History
↓
Question Contextualizer (LLM)
↓
Standalone Question
↓
Retriever (Chroma)
↓
Context Formatting
↓
Prompt + Context
↓
LLM Answer

##Tech Stack

**Python 3.10+**
**LangChain (LCEL / Runnables)**
**Groq (LLaMA 3.1)**
**HuggingFace Sentence Transformers**
**Chroma Vector Database**
**BeautifulSoup (optional HTML parsing)**

# Setup

# Create Virtual Environment
```bash
python -m venv .ven
source .ven/bin/activate

#Install Dependencies
pip install -r requirements.txt
#Set Environment Variable
Create a .env file:
GROQ_API_KEY=your_groq_api_key_here
#Run the Chatbot
python src/agent.py