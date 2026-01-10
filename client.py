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
import json
from rule_based_verifier import rule_based_verifier
from executioner import execute_plan

MAX_RETRIES = 2

memory = MemorySaver()
history = []

def normalize_plan(raw):
    # If already dict, return
    if isinstance(raw, dict):
        return raw

    # If string, clean and parse
    if isinstance(raw, str):
        raw = raw.strip()

        # Remove surrounding quotes if present
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
            raw = raw.replace('\\"', '"')

        # Remove markdown fences if any
        if raw.startswith("```"):
            raw = raw.split("```")[1]

        return json.loads(raw)

    raise TypeError(f"Unsupported plan type: {type(raw)}")

def is_conversational_query(user_input):
    """Detect if query is conversational and doesn't need tool planning."""
    conversational_patterns = [
        # Greetings
        r'\b(hi|hello|hey|sup|wassup|namaste)\b',
        # Introductions
        r'\b(my name is|i am|i\'m|call me)\b',
        # Simple questions about self/memory
        r'\b(how are you|what\'s up|whats up|kaise ho)\b',
        r'\b(what is my name|what\'s my name|whats my name|do you remember|who am i)\b',
        r'\b(tell me about myself|remind me|what did i say|what did i tell you)\b',
        # Thanks
        r'\b(thank|thanks|dhanyavaad|shukriya)\b',
        # General chitchat
        r'\b(good morning|good night|good evening|bye|goodbye|see you)\b',
    ]
    
    import re
    user_lower = user_input.lower().strip()
    
    # Check patterns
    for pattern in conversational_patterns:
        if re.search(pattern, user_lower):
            return True
    
    # Check if very short (likely conversational)
    if len(user_lower.split()) <= 5 and '?' in user_lower:
        # Short questions without tool keywords are likely conversational/memory questions
        tool_keywords = ['calculate', 'solve', 'weather', 'translate', 'search', 'email', 'send', 'find', 'temperature', 'forecast']
        if not any(keyword in user_lower for keyword in tool_keywords):
            return True
    
    if len(user_lower.split()) <= 3 and '?' not in user_lower:
        # Short statements without questions are likely conversational
        tool_keywords = ['calculate', 'solve', 'weather', 'translate', 'search', 'email', 'send', 'find']
        if not any(keyword in user_lower for keyword in tool_keywords):
            return True
    
    return False


async def plan_task(model , user_input):
    
    PLANNER_PROMPT = """
You are a TASK PLANNER for an agentic AI system.

Your job:
- Understand the user's goal
- Break it into ordered steps
- Decide which tool (if any) is required per step

Rules:
- DO NOT execute tools
- DO NOT answer the user
- OUTPUT ONLY RAW JSON
- NO markdown
- NO backticks

Available tools:
- math_server
- weather
- Translate
- websearch
- gmail

JSON SCHEMA:
{
  "goal": "...",
  "intent": "...",
  "steps": [
    {
      "id": 1,
      "action": "...",
      "tool": "tool_name or none",
      "description": "..."
    }
  ],
  "requires_confirmation": true or false
}
"""
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=user_input)
    ]
    response = await model.ainvoke(messages)
    try:
        plan = response.content
        return normalize_plan(plan)
    except json.JSONDecodeError:
        print("Failed to parse plan response as JSON.")
    
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
    
    planner_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
    )

    executor_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
    )

    # System prompt to guide tool usage
    system_message_content = """You are a helpful AI assistant with access to multiple tools.

    Available tools:
   - solve_math: For mathematical calculations, equations, and problem solving
- translate: For translating text between languages
- search_web, find_relevant_urls, scrape_pdfs, search_and_download_pdfs: For web searching and PDF operations
- read_emails, send_email: For Gmail operations
- get_current_weather, get_air_quality, get_geo_details, get_environment_report: For weather and location information

IMPORTANT: Only use tools when the user's request requires them. For simple conversational inputs (greetings, introductions, casual chat), respond naturally without calling any tools.

When a user asks a question that needs tools:
1. Analyze what tools you need to use
2. Call the appropriate tool(s) with correct parameters
3. Use the tool results to provide a comprehensive answer
4. Always explain your reasoning and what tools you used

For conversational inputs (like "my name is...", "hello", "how are you"), just respond naturally without using any tools.

Be proactive in using tools to give accurate, up-to-date information when needed."""
    
    # Create agent with memory (system prompt will be added to messages)
    agent = create_react_agent(executor_model, tools, checkpointer=memory)
    
    
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
                # 🎯 Intent-Based Bypass: Check if conversational
                if is_conversational_query(user_input):
                    # Direct response without planning
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config
                    )
                    assistant_message = response["messages"][-1].content
                    print(assistant_message)
                    
                    # Store in history and FAISS
                    history.append({"user": user_input, "assistant": assistant_message})
                    faiss_index.add_texts([assistant_message], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
                    faiss_index.save_local(index_path)
                    continue
                
                # Complex query: Use full pipeline
                plan = await plan_task(planner_model, user_input)
                assert isinstance(plan, dict), f"Plan is not dict: {type(plan)}"
                
                retry_count = 0
                execution_hint = None

                while retry_count <= MAX_RETRIES:
                    execution_results, final_answer = await execute_plan(
                        agent,
                        plan,
                        execution_hint
                    )

                    verdict = rule_based_verifier(
                        user_query=user_input,
                        plan=plan,
                        execution_results=execution_results,
                        final_answer=final_answer
                    )

                    if verdict["verdict"] == "PASS":
                        print("\n✅ Final Answer:\n")
                        print(final_answer)
                        
                        # Store in history and FAISS
                        history.append({"user": user_input, "assistant": final_answer})
                        faiss_index.add_texts([final_answer], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
                        faiss_index.save_local(index_path)
                        break

                    elif verdict["verdict"] == "RETRY":
                        retry_count += 1
                        execution_hint = verdict["retry_hint"]

                        print(f"\n🔁 Retry {retry_count}/{MAX_RETRIES}")
                        print("Reason:", verdict["reason"])
                        print("Hint:", execution_hint)

                        continue

                    else:  # FAIL
                        print("\n Failed:")
                        print(verdict["reason"])
                        break


            except Exception as e:
                print(f"\nError during agent call: {e}")
                traceback.print_exc()
                print(f"\nPlease try again after fixing.")

            
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Clean up clients if needed
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())