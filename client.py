from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import traceback
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

memory = MemorySaver()
history = []


async def main():
    """Run a chat using MCPAgent's built-in conversation memory."""

    load_dotenv()  # Load environment variables for API key

    print("Initializing chat...")

    clients = MultiServerMCPClient(
        {
            "math_server": {
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
            }
        }
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    os.environ["GROQ_API_KEY"] = groq_api_key

    # CREATING LLM MODEL
    tools = await clients.get_tools()
    print(f"\nLoaded {len(tools)} tools: {[tool.name for tool in tools]}\n")
    
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # System prompt to guide tool usage
    system_message_content = """You are a helpful AI assistant with access to multiple tools.

Available tools:
- solve_math: For mathematical calculations, equations, and problem solving
- translate: For translating text between languages
- search_web, find_relevant_urls, scrape_pdfs, search_and_download_pdfs: For web searching and PDF operations
- read_emails, send_email: For Gmail operations

When a user asks a question:
1. Analyze what tools you need to use
2. Call the appropriate tool(s) with correct parameters
3. Use the tool results to provide a comprehensive answer
4. Always explain your reasoning and what tools you used

Be proactive in using tools to give accurate, up-to-date information."""
    
    # Create agent with memory (system prompt will be added to messages)
    agent = create_react_agent(model, tools, checkpointer=memory)
    
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_path = "faiss_index"
        index_file = os.path.join(index_path, "index.faiss")
        
        # Check if the actual index file exists, not just the directory
        if os.path.exists(index_file):
            faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing FAISS index.")
        else:
            faiss_index = FAISS.from_texts(["initial text"], embeddings)
            faiss_index.save_local(index_path)
            print("Created new FAISS index.")

        # Session configuration for conversation threading
        config = {"configurable": {"thread_id": "main_conversation"}}
        
        # Initialize conversation with system message ONCE
        system_msg = SystemMessage(content=system_message_content)
        await agent.ainvoke({"messages": [system_msg]}, config=config)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                faiss_index.save_local(index_path)
                print("FAISS index saved.")
                break
            
            if user_input.lower() == "clear":
                faiss_index = FAISS.from_texts(["initial text"], embeddings)
                faiss_index.save_local(index_path)
                history.clear()
                # Start new thread and reinitialize with system message
                config = {"configurable": {"thread_id": f"conversation_{datetime.now().timestamp()}"}}
                await agent.ainvoke({"messages": [system_msg]}, config=config)
                print("Conversation history cleared.")
                continue

            # Store user input in FAISS
            faiss_index.add_texts([user_input], metadatas=[{"source": "user", "timestamp": str(datetime.now())}])
            
            # Retrieve relevant context from FAISS (last 3 interactions)
            try:
                relevant_context = faiss_index.similarity_search(user_input, k=3)
                context_text = "\n".join([doc.page_content for doc in relevant_context if doc.page_content != "initial text"])
            except:
                context_text = ""
            
            print("\nAssistant: ", end="", flush=True)
            
            try:
                # Build message with context
                full_input = user_input
                if context_text and len(history) > 0:
                    full_input = f"Recent conversation context:\n{context_text}\n\nCurrent question: {user_input}"
                
                # Invoke agent - checkpointer handles full conversation history
                response = await agent.ainvoke(
                    {"messages": [HumanMessage(content=full_input)]},
                    config=config
                )
                
                # Extract the final message
                assistant_message = response["messages"][-1].content
                
                # Store in history and FAISS
                history.append({"user": user_input, "assistant": assistant_message})
                faiss_index.add_texts([assistant_message], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
                faiss_index.save_local(index_path)
                
                print(assistant_message)
                
            except Exception as e:
                print(f"\nError during agent call: {e}")
                traceback.print_exc()
                print("\nPlease try rephrasing your question or check if all MCP servers are running.")

            
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Clean up clients if needed
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())