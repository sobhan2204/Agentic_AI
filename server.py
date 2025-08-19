# server.py - FastAPI integration for website.html and client.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import os
import sys
from datetime import datetime
from langchain_community.vectorstores import FAISS


# Import your MCP client code
import client

app = FastAPI(title="MCP Agent Web Server")

# Add CORS middleware to handle frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# Global variables for MCP components
mcp_initialized = False
mcp_components = {}

async def initialize_mcp():
    """Initialize MCP components from client.py"""
    global mcp_initialized, mcp_components
    
    if mcp_initialized:
        return mcp_components
        
    try:
        print("Initializing MCP components...")
        
        # Initialize the components from your client.py
        # We'll extract the initialization logic
        from dotenv import load_dotenv
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langgraph.prebuilt import create_react_agent
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        load_dotenv()
        
        # Same MCP clients as your client.py
        clients = MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    "args": ["mathserver.py"],
                    "transport": "stdio",
                },
                "weather": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                },
                "Translate": {
                    "command": "python",
                    "args": ["translate.py"],
                    "transport": "stdio",
                },
                "websearch": {
                    "command": "python",
                    "args": ["websearch.py"],
                    "transport": "stdio",
                },
                "gmail": {
                    "command": "python",
                    "args": ["gmail.py"],
                    "transport": "stdio",
                },
            }
        )
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        
        # Initialize model and agent
        tools = await clients.get_tools()
        model = ChatGroq(model="llama3-70b-8192")
        agent = create_react_agent(model, tools)
        
        # Initialize FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing FAISS index.")
        else:
            faiss_index = FAISS.from_texts(["initial text"], embeddings)
            print("Created new FAISS index.")
        
        # Chat template from your client.py
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "you are a indian collage student . So answer {user_input} using slang language and also use as many curse words and used hinglish.Also give all the informationa dn facts regarding to it but make it compact. And while sending an email don't need to write whole explanation just tell me it is sent or not , with not all just nessary information. Only tell inormation about email when the input have email or gmail tool is being called .And always give answer using LLM and web tool together always. And don't need to write note"),
            ("user", "{user_input}")
        ])
        
        mcp_components = {
            "clients": clients,
            "model": model,
            "agent": agent,
            "faiss_index": faiss_index,
            "embeddings": embeddings,
            "chat_template": chat_template
        }
        
        mcp_initialized = True
        print("MCP initialization complete!")
        return mcp_components
        
    except Exception as e:
        print(f"MCP initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize MCP: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize MCP on server startup"""
    await initialize_mcp()

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_website():
    """Serve the website.html file"""
    try:
        with open("website.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="website.html not found")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Process chat message using MCP agent"""
    try:
        if not mcp_initialized:
            await initialize_mcp()
        
        components = mcp_components
        user_input = message.message.strip()
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Add to FAISS index (same logic as client.py)
        components["faiss_index"].add_texts(
            [user_input], 
            metadatas=[{"source": "user", "timestamp": str(datetime.now())}]
        )
        
        # Get similar documents for context
        similar_docs = components["faiss_index"].similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in similar_docs])
        
        # Format prompt
        formatted_prompt = components["chat_template"].format_messages(
            user_input=user_input + "\nContext from past" + context
        )
        
        # Get AI response using your agent
        ai_response = await components["agent"].ainvoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        
        # Get model response
        model_response = await components["model"].ainvoke(formatted_prompt)
        response_content = model_response.content
        
        # Add response to FAISS
        components["faiss_index"].add_texts(
            [response_content], 
            metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}]
        )
        
        # Save FAISS index
        components["faiss_index"].save_local("faiss_index")
        
        return ChatResponse(
            response=response_content,
            status="success"
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            response=f"Oops yaar! Something went wrong: {str(e)}",
            status="error"
        )

# Clear chat history endpoint
@app.post("/clear")
async def clear_chat():
    """Clear conversation history (reset FAISS)"""
    try:
        if not mcp_initialized:
            await initialize_mcp()
            
        components = mcp_components
        
        # Reset FAISS index (same as client.py clear logic)
        components["faiss_index"] = FAISS.from_texts(["initial text"], components["embeddings"])
        components["faiss_index"].save_local("faiss_index")
        
        return {"status": "success", "message": "Chat history cleared successfully!"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if MCP services are running"""
    return {
        "status": "healthy",
        "mcp_initialized": mcp_initialized,
        "timestamp": datetime.now().isoformat()
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Web Server...")
    print("Website will be available at: http://localhost:8080")
    print("Make sure your MCP servers (mathserver.py, translate.py, etc.) are running!")
    
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True
    )