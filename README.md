# 🤖 Agentic_AI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agents-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-pink)

> **Agentic_AI** is a **multi-agent orchestration framework** powered by **LangChain + LangGraph + MCP** and accelerated by **Groq (LLaMA-3-70B)**.  
> It integrates specialized tools like **Math, Translation, Gmail, Weather, Web Search**, and features **memory persistence** with FAISS embeddings.

---

## ✨ Demo

🎥 *Demo GIF or screenshot placeholder*  
(Add a quick **terminal demo GIF** using [asciinema](https://asciinema.org/) or a screenshot of agents collaborating.)

---

## 🧠 Architecture

Here’s how the system is designed:

```mermaid
graph TB
    User[👤 User Input] --> IntentRouter{🎯 Intent-Based<br/>Router}
    
    IntentRouter -->|Conversational| DirectAgent[💬 Direct Response<br/>Agent]
    IntentRouter -->|Task-Based| Planner[📋 Task Planner<br/>LLaMA-3.1-8B]
    
    Planner -->|JSON Plan| Executor[⚙️ Plan Executor<br/>LLaMA-3.1-8B]
    
    Executor --> Tools[🛠️ MCP Tool Suite]
    Tools --> Math[🧮 Math Server<br/>stdio]
    Tools --> Translate[🌍 Translator<br/>stdio]
    Tools --> Gmail[📧 Gmail API<br/>stdio]
    Tools --> WebSearch[🔍 Web Search<br/>stdio]
    Tools --> Weather[🌦️ Weather<br/>HTTP]
    
    Executor -->|Results| Verifier{✅ Rule-Based<br/>Verifier}
    
    Verifier -->|PASS| FinalAnswer[📤 Final Answer]
    Verifier -->|RETRY<br/>Max 2x| Executor
    Verifier -->|FAIL| ErrorHandler[❌ Error Handler]
    
    DirectAgent --> Memory[(🧠 FAISS Vector DB<br/>HuggingFace Embeddings)]
    Executor --> Memory
    Memory -->|Top 3 Context| Executor
    Memory -->|Context| DirectAgent
    
    FinalAnswer --> User
    DirectAgent --> User
    ErrorHandler --> User
    
    subgraph "🌐 Web Interface"
        API[FastAPI Server<br/>CORS Enabled]
    end
    
    User -.->|HTTP| API
    API -.-> IntentRouter
    
    style IntentRouter fill:#ff9999
    style Planner fill:#99ccff
    style Executor fill:#99ccff
    style Verifier fill:#99ff99
    style Memory fill:#ffcc99
    style Tools fill:#e6b3ff
```

## 🚀 Features

### 🎯 Core Intelligence Features

 **Intent-Based Query Routing**: Automatically detects conversational vs. task-based queries, bypassing unnecessary tool planning for greetings and casual chat for more natural interactions

 **Plan-Verify-Execute Architecture**: Sophisticated 3-stage pipeline with Planner (breaks complex tasks into JSON plans), Executor (executes steps with context), and Verifier (validates outputs with smart retry logic)

 **Smart Rule-Based Verification**: Multi-layer validation including step count validation, tool compliance checking, goal satisfaction analysis, freshness detection for news queries, and generic failure detection

 **Adaptive Retry Mechanism**: Self-correcting system with up to 2 retries, providing specific retry hints to executors for automatic error recovery without user intervention

### 🛠️ Tool & Integration Features

 ⚡ **Fast & Scalable**: Powered by Groq's LLM for blazing fast inference

 🧮 **Math Agent**: Handles calculations & symbolic tasks

 🌍 **Translator Agent**: Supports multilingual conversations

 📧 **Gmail Agent**: Reads & interacts with Gmail API

 🔍 **Web Search Agent**: Searches online data for better answers

 🌦️ **Weather Agent**: Provides current weather, air quality, and environmental reports

 **Multi-Transport MCP Integration**: Supports multiple MCP servers with stdio and streamable_http transports for mixing local and remote tools seamlessly

### 🧠 Memory & Context Features

 **Persistent Conversational Memory**: FAISS vector database with HuggingFace embeddings stores and retrieves conversation context semantically

 **Context-Aware Responses**: Retrieves top 3 relevant past interactions for maintaining conversation continuity

 **Cross-Session Memory**: Maintains conversation history across sessions with persistent storage

 🧹 **Memory Reset**: Use `clear` to reset past memory when needed

### 🌐 Deployment Features

 **Web Interface Ready**: FastAPI integration with CORS support for web deployment and frontend interfaces

 **RESTful API**: Production-ready API endpoints for chat interactions


## 🗂 Project Structure
 Agentic_AI/
 ├── main.py              # Main entry point  
 ├── mathserver.py        # Math agent (MCP)
 ├── translate.py         # Translator agent (MCP)
 ├── websearch.py         # Web search agent
 ├── gmail.py             # Gmail integration
 ├── rag_model.py         # Optional RAG pipeline
 ├── mcp_use.py           # MCP agent utilities
 ├── requirements.txt     # Dependencies
 ├── .env                 # API keys & config
 └── README.md            # This file

## ⚙️ Getting Started
1️⃣ Clone the repository
 git clone https://github.com/sobhan2204/Agentic_AI.git
 cd Agentic_AI

2️⃣ Setup environment
 python3.10 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt

3️⃣ Configure .env
 GROQ_API_KEY=your_api_key_here
 HF_TOKEN=your_huggingface_token_here
 (Optional: add Gmail API credentials if using Gmail Agent)

4️⃣ Run the agent
 python main.py


## 📈 Roadmap

 Add finance/news/calendar agents

 Memory expiration + relevance scoring

 Web dashboard UI for interactions

 Dockerized deployment

## 🧰 Tech Stack

LangChain + LangGraph + MCP – multi-agent orchestration

Groq (LLaMA-3-70B) – blazing fast inference

FAISS + HuggingFace embeddings – vector memory store

Python 3.10+ – backend

## 🤝 Contributing

##💡 Contributions are welcome!

Fork the repo & create a feature branch

Submit a PR with clear description

For new MCP agents, follow modular design
