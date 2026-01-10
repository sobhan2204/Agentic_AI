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
from langchain_core.messages import HumanMessage, SystemMessage
from groq import RateLimitError
import json

memory = MemorySaver()


def load_tool_keywords(file_path="tool_keywords.json"):
    """Load tool keywords from JSON file."""
    try:
        with open(file_path, 'r') as f:
            keywords = json.load(f)
        print(f"✓ Loaded tool keywords from {file_path}")
        return keywords
    except FileNotFoundError:
        print(f"⚠️ {file_path} not found. Using empty keywords.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing {file_path}: {e}")
        return {}


def analyze_query_for_tools(query: str, tool_keywords: dict) -> list:
    """Analyze query and suggest relevant tools based on keywords."""
    query_lower = query.lower()
    suggested_tools = []
    
    for tool_name, keywords in tool_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            suggested_tools.append(tool_name)
    
    return suggested_tools


async def main():
    """Run an enhanced chat with keyword-based tool routing."""

    load_dotenv()
    print("🚀 Initializing enhanced agentic AI chat...\n")

    # Load tool keywords from JSON file
    TOOL_KEYWORDS = load_tool_keywords()

    # Initialize MCP clients
    try:
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
        print("✓ MCP clients initialized")
    except Exception as e:
        print(f"❌ Error initializing MCP clients: {e}")
        return

    # Verify API key
    groq_api_key = os.getenv("GROQ_API_KEY_2")
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file")
    os.environ["GROQ_API_KEY"] = groq_api_key
    print("✓ Groq API key loaded")

    # Load tools
    try:
        tools = await clients.get_tools()
        print(f"✓ Loaded {len(tools)} tools: {[tool.name for tool in tools]}\n")
    except Exception as e:
        print(f"❌ Error loading tools: {e}")
        return

    # Create LLM model with reduced token usage
    # Try models in order of preference
    models_to_try = [
        "llama-3.1-8b-instant",      # Fast, high token limit
        "llama3-groq-70b-8192-tool-use-preview",  # Good for tools
        "mixtral-8x7b-32768",        # Alternative model
        "llama-3.3-70b-versatile"    # Original (if rate limit resets)
    ]
    
    model = None
    for model_name in models_to_try:
        try:
            model = ChatGroq(
                model=model_name,
                temperature=0.5,
                max_tokens=1024,
                timeout=60
            )
            print(f"✓ Using {model_name} model")
            break
        except Exception as e:
            print(f"⚠️ {model_name} not available: {e}")
            continue
    
    if model is None:
        raise ValueError("❌ No available models. Check Groq console for active models.")

    # Optimized system prompt
    system_message_content = """You are an AI assistant with specialized tools. Use them efficiently:

📊 MATH: solve_math - calculations, equations, derivatives
🌤️ WEATHER: get_current_weather, get_air_quality - forecasts
🌍 TRANSLATE: translate - language conversion
🔍 WEB: search_web, find_relevant_urls - searches, PDFs
📧 GMAIL: read_emails, send_email - email management

RULES:
1. Analyze query keywords to pick the right tool
2. Use tools proactively, don't guess answers
3. If a tool fails, explain clearly
4. Be concise in responses"""

    # Create agent with memory
    agent = create_react_agent(model, tools, checkpointer=memory)
    print("✓ Agent created with memory\n")

    # Initialize FAISS for conversation context
    faiss_index = None
    embeddings = None
    index_path = "faiss_index"
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_file = os.path.join(index_path, "index.faiss")

        if os.path.exists(index_file):
            faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("✓ Loaded existing FAISS index")
        else:
            os.makedirs(index_path, exist_ok=True)
            faiss_index = FAISS.from_texts([""], embeddings)
            faiss_index.save_local(index_path)
            print("✓ Created new FAISS index")
    except Exception as e:
        print(f"⚠️ FAISS disabled: {e}")
        print("Continuing without FAISS indexing...\n")

    # Session configuration
    config = {"configurable": {"thread_id": "main_conversation"}}

    # Initialize with system message
    system_msg = SystemMessage(content=system_message_content)

    print("=" * 60)
    print("🤖 AGENTIC AI CHAT - Ready!")
    print("=" * 60)
    print("Commands: 'exit/quit' to exit, 'clear' to reset")
    print("=" * 60 + "\n")

    conversation_count = 0

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                if faiss_index:
                    try:
                        faiss_index.save_local(index_path)
                        print("\n✓ FAISS index saved.")
                    except:
                        pass
                print("👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                if faiss_index and embeddings:
                    try:
                        faiss_index = FAISS.from_texts([""], embeddings)
                        faiss_index.save_local(index_path)
                    except:
                        pass
                config = {"configurable": {"thread_id": f"conversation_{datetime.now().timestamp()}"}}
                conversation_count = 0
                print("\n✓ Conversation history cleared.\n")
                continue

            # Analyze query for relevant tools
            suggested_tools = analyze_query_for_tools(user_input, TOOL_KEYWORDS)
            if suggested_tools:
                print(f"🔧 Detected relevant tools: {', '.join(suggested_tools)}")

            # Store in FAISS if available
            if faiss_index:
                try:
                    faiss_index.add_texts(
                        [user_input],
                        metadatas=[{"source": "user", "timestamp": str(datetime.now())}]
                    )
                except Exception as e:
                    print(f"⚠️ FAISS storage warning: {e}")

            # Get minimal context from FAISS
            context_text = ""
            if faiss_index and conversation_count > 2:
                try:
                    relevant_docs = faiss_index.similarity_search(user_input, k=2)
                    context_text = "\n".join([
                        doc.page_content[:200] for doc in relevant_docs
                        if doc.page_content.strip()
                    ])
                except:
                    pass

            print("\nAssistant: ", end="", flush=True)

            # Build concise input
            enhanced_input = user_input
            if suggested_tools:
                enhanced_input = f"[Tools: {', '.join(suggested_tools)}] {user_input}"
            
            if context_text and conversation_count > 2:
                enhanced_input = f"Context: {context_text[:300]}\n\n{enhanced_input}"

            # Create messages for agent
            messages = [HumanMessage(content=enhanced_input)]
            if conversation_count == 0:
                messages.insert(0, system_msg)

            # Invoke agent
            try:
                response = await agent.ainvoke(
                    {"messages": messages},
                    config=config
                )

                # Extract assistant response
                assistant_message = response["messages"][-1].content
                print(assistant_message)

                # Store assistant response in FAISS
                if faiss_index:
                    try:
                        faiss_index.add_texts(
                            [assistant_message],
                            metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}]
                        )
                        faiss_index.save_local(index_path)
                    except:
                        pass

                conversation_count += 1

            except RateLimitError as e:
                print(f"\n❌ Rate limit exceeded!")
                print(f"Error: {e}")
                print("\n💡 Solutions:")
                print("   1. Wait for rate limit reset (check error message)")
                print("   2. Upgrade to Groq Dev Tier: https://console.groq.com/settings/billing")
                print("   3. Use a different API key")
                print("   4. Switch to llama3-8b-8192 model (change line 89)\n")
                
            except Exception as e:
                print(f"\n❌ Error during agent execution: {e}")
                print(f"Error type: {type(e).__name__}")
                
                if os.getenv("DEBUG") == "1":
                    traceback.print_exc()
                
                print("\n💡 Troubleshooting tips:")
                print("   - Ensure all MCP server scripts are running")
                print("   - Check server ports (8000 for weather)")
                print("   - Verify GROQ_API_KEY in .env file")
                print("   - Try rephrasing your question\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            breakaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if os.getenv("DEBUG") == "1":
                traceback.print_exc()
            continue

    print("\nShutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if os.getenv("DEBUG") == "1":
            traceback.print_exc()