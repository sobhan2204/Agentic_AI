# 🧠 Agentic AI

Agentic AI is a multi-agent conversational system built with [LangChain](https://www.langchain.com/), [LangGraph](https://www.langchain.com/langgraph), and [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol).  
It leverages Groq’s **LLaMA-3 70B** model for reasoning and integrates multiple specialized tools (math, translation, weather, web search, Gmail, and more) to provide contextual, professional, and memory-aware responses.

---

## 🚀 Features

- **Multi-Agent Orchestration**  
  Uses MCP clients for math, translation, weather, Gmail, and web search.
  
- **Powerful LLM Backbone**  
  Runs on Groq’s ultra-fast inference engine with **LLaMA-3-70B**.

- **Conversational Memory**  
  Stores past queries and responses using **FAISS vector database** + **HuggingFace embeddings**.

- **Context-Aware Chatting**  
  Adds relevant past context to improve responses.

- **Customizable Prompting**  
  Defaults to a *“professional news reporter”* style: precise, detailed, and compact answers.

- **Clear & Reset**  
  Ability to reset FAISS memory with a simple `clear` command.

---

## 📂 Project Structure
Agentic_AI/
│── main.py # Entry point for chat agent
│── mathserver.py # MCP math server
│── translate.py # MCP translation server
│── websearch.py # MCP web search server
│── gmail.py # MCP Gmail integration
│── Rag_model.py # (optional) Retrieval-Augmented Generation
│── mcp_use.py # MCP agent/client utilities
│── requirements.txt # Python dependencies
│── .env # Environment variables (API keys, etc.)

---

## 🔑 Requirements

- Python **3.10+**
- Groq API Key (set in `.env`)
- Hugging Face Access Token (if using embeddings/models)
- Google credentials (`client_secret.json`) for Gmail API (optional)
- MCP dependencies (`langchain-mcp-adapters`, etc.)

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/sobhan2204/Agentic_AI.git
   cd Agentic_AI
🛠️ Tech Stack

LangChain
 – LLM orchestration

LangGraph
 – Agent workflow

MCP
 – Multi-server agent tooling

Groq
 – Fast inference backend (LLaMA-3-70B)

HuggingFace Transformers
 – Sentence embeddings

FAISS
 – Vector search database

Dotenv
 – Env variable management
 
 📌 Roadmap / Future Work

Add more MCP servers (finance, news, calendar, etc.)

Expand FAISS memory management (expiration, scoring)

Web-based frontend for interaction

Dockerize deployment
