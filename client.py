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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
import json
from rule_based_verifier import rule_based_verifier
from executioner import execute_plan
from itertools import cycle


MAX_RETRIES = 2
SUMMARIZE_AFTER = 10  # Summarize conversation after every 10 exchanges

memory = MemorySaver()
history = []
conversation_summary = ""

def normalize_plan(raw):
    if isinstance(raw, dict):
        return raw

    # If string, clean and parse
    if isinstance(raw, str):
        raw = raw.strip()

        # Remove surrounding quotes if present
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
            raw = raw.replace('\\"', '"') # this function will remove any "\\" and change it into blank space 

        # Remove markdown fences if any
        if raw.startswith("```"):
            raw = raw.split("```")[1]

        return json.loads(raw)

    raise TypeError(f"Unsupported plan type: {type(raw)}")
async def summarize_conversation(model, history, previous_summary=""):
    """Summarize conversation history to reduce memory usage."""
    if not history:
        return previous_summary
    
    # Format conversation for summarization
    conversation_text = "\n".join([
        f"User: {entry['user']}\nAssistant: {entry['assistant']}"
        for entry in history
    ])
    
    SUMMARY_PROMPT = f"""Summarize the following conversation concisely, preserving key information like names, preferences, and important context.

Previous summary: {previous_summary if previous_summary else 'None'}

Recent conversation:
{conversation_text}

Provide a brief summary (2-3 sentences) that captures the essential information:"""
    
    messages = [
        SystemMessage(content="You are a conversation summarizer. Create concise summaries that preserve important context."),
        HumanMessage(content=SUMMARY_PROMPT)
    ]
    
    response = await model.ainvoke(messages)
    return response.content.strip()

async def is_conversational_query(model, user_input):
    
    CLASSIFIER_PROMPT = """You are an intent classifier. Determine if the user's input is:
- CONVERSATIONAL: Greetings, chitchat, personal questions, follow-ups, clarifications, memory queries
- TASK: Requires tools like math, weather, translation, search, email
Your role is to classify the input into one of these two categories based on intent.

Respond with ONLY one word: "CONVERSATIONAL" or "TASK"

Examples:
Input: "Hi, how are you?" → CONVERSATIONAL
Input: "What's my name?" → CONVERSATIONAL
Input: "Tell me more about that" → CONVERSATIONAL
Input: "Calculate 5+5" → TASK
Input: "What's the weather in Delhi?" → TASK
Input: "Send an email to John" → TASK
"""
    
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=f"Input: {user_input}")
    ]
    
    response = await model.ainvoke(messages)
    classification = response.content.strip().upper()
    
    return classification == "CONVERSATIONAL"


async def plan_task(model , user_input):
    
    PLANNER_PROMPT = """
You are a TASK PLANNER for an Agentic AI system.

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
        print("Failed to get plan response as JSON.")
    
async def main():
    """Run a chat using MCPAgent's built-in conversation memory."""

    load_dotenv()  

    print("Hii I am your personalized AI chatbot...")

    clients = MultiServerMCPClient(
        {
            "math_server": {
                "command": "python",
                "args": ["-u" , "mathserver.py"],
                "transport": "stdio",
            },
             "weather": {
                 "command": "python",
                 "args": ["-u" , "weather.py"],
                 "transport": "stdio",
             },
            "Translate": {
                "command": "python",
                "args": ["-u" , "translate.py"],
                "transport": "stdio",
            },
            "websearch": {
                "command": "python",
                "args": ["-u" , "websearch.py"],
                "transport": "stdio",
            },
            "gmail": {
                "command": "python",
                "args": ["-u" , "gmail.py"],
                "transport": "stdio",
            }
        }
    )

    # Load all API keys for rotation
    api_keys = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3")
    ]
    
    # Filter out None values (in case some keys are missing)
    api_keys = [key for key in api_keys if key]
    
    if not api_keys:
        raise ValueError("No GROQ_API_KEY_1, GROQ_API_KEY_2, or GROQ_API_KEY_3 found in .env")
    
    print(f"Loaded {len(api_keys)} API keys for rotation\n")
    
    # Create cycle iterator for key rotation
    key_cycle = cycle(api_keys)

    # CREATING LLM MODEL
    tools = await clients.get_tools()
    print(f"\nLoaded {len(tools)} tools: {[tool.name for tool in tools]}\n")
    
    planner_model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=1500,  # Reduced to leave room for input
    temperature=0,
    api_key=next(key_cycle)
    )

    executor_model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=1500,  # Reduced to leave room for input
    temperature=0.7,
    api_key=next(key_cycle)
    )

    # System prompt to guide tool usage
    system_message_content = """You are a helpful coversational AI assistant with access to multiple tools.

    Available tools:
   - solve_math: For mathematical calculations, equations, and problem solving
- translate: For translating text between languages (ONLY when explicitly asked to translate)
- search_web, find_relevant_urls, scrape_pdfs, search_and_download_pdfs: For web searching and PDF operations
- read_emails, send_email: For Gmail operations
- get_current_weather, get_air_quality, get_geo_details, get_environment_report: For weather and location information

CRITICAL RULES:
1. NEVER use tools for greetings (hi, hello, how are you, etc.)
2. NEVER use tools for casual conversation
3. NEVER use translate tool unless user explicitly asks to translate something
4. ONLY use tools when the user's request CLEARLY requires external data or computation

For conversational inputs:
- Greetings: "hi", "hello", "how are you" → Respond directly, NO TOOLS
- Personal questions: "what's my name", "who am I" → Use memory, NO TOOLS  
- Clarifications: "tell me more", "explain that" → Use context, NO TOOLS

For tool-requiring inputs:
- Math: "calculate 5+5" → Use solve_math
- Weather: "what's the weather in Delhi" → Use weather tools
- Translation: "translate 'hello' to Spanish" → Use translate
- Search: "search for..." → Use websearch
- Email: "send an email to..." → Use gmail

Respond naturally and conversationally. Be helpful but don't overuse tools."""
    
    # Message trimming strategy to prevent token overflow
    # Keep only system message + last 6 messages (3 exchanges)
    def trim_message_history(messages):
        """Trim messages to prevent token overflow, always keeping system message."""
        if len(messages) <= 7:  # System + 6 messages
            return messages
        
        # Always keep the first message (system prompt)
        system_msg = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
        
        # Keep only last 6 messages for conversation
        recent_messages = messages[-6:]
        
        if system_msg:
            return [system_msg] + recent_messages
        return recent_messages
    
    # Create agent with memory and message trimming
    agent = create_react_agent(
        executor_model, 
        tools, 
        checkpointer=memory,
        state_modifier=trim_message_history  # Automatically trim on each call
    )
    
    
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
                print("bye bye ! FAISS index have been saved.")
                break
            
            if user_input.lower() == "clear":
                faiss_index = FAISS.from_texts(["initial text"], embeddings)
                faiss_index.save_local(index_path)
                history.clear()
                conversation_summary = ""  # Reset summary
                # Start new thread and reinitialize with system message
                config = {"configurable": {"thread_id": f"conversation_{datetime.now().timestamp()}"}}
                await agent.ainvoke({"messages": [system_msg]}, config=config)
                print("Conversation history cleared.")
                continue

            # Store user input in FAISS
            faiss_index.add_texts([user_input], metadatas=[{"source": "user", "timestamp": str(datetime.now())}])
            
            # Retrieve relevant context from FAISS (last 3 interactions) -> we can add morecontext but it wolud take a lot of time 
            try:
                relevant_context = faiss_index.similarity_search(user_input, k=3)
                context_text = "\n".join([doc.page_content for doc in relevant_context if doc.page_content != "initial text"])
            except:
                context_text = ""
            
            print("\nAssistant: ", end="", flush=True)
            
            try:
                is_conversational = await is_conversational_query(planner_model, user_input)
                print(f"[DEBUG] Query classified as: {'CONVERSATIONAL' if is_conversational else 'TASK'}")
                
                if is_conversational:
                    # Build context with summary and recent messages
                    full_context = ""
                    if conversation_summary:
                        full_context += f"Conversation summary: {conversation_summary}\n\n"
                    if context_text:
                        full_context += f"Recent context:\n{context_text}\n\n"
                    
                    message_with_context = f"{full_context}Current query: {user_input}" if full_context else user_input
                    # Direct response without planning
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=message_with_context)]},
                        config=config
                    )
                    assistant_message = response["messages"][-1].content
                    print(assistant_message)
                    
                    # Store in history and FAISS
                    history.append({"user": user_input, "assistant": assistant_message})
                    faiss_index.add_texts([assistant_message], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
                    faiss_index.save_local(index_path)
                    
                    # Summarize and clear old history if threshold reached
                    #global summarize_conversation
                    if len(history) >= SUMMARIZE_AFTER:
                        print("\n[Summarizing conversation to reduce memory...]")
                        conversation_summary = await summarize_conversation(planner_model, history, conversation_summary)
                        history.clear()  # Clear old messages after summarization
                    
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
                        print("\n Final Answer:\n")
                        print(final_answer)
                        
                        # Store in history and FAISS
                        history.append({"user": user_input, "assistant": final_answer})
                        faiss_index.add_texts([final_answer], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
                        faiss_index.save_local(index_path)
                        
                        # Summarize and clear old history if threshold reached
                        if len(history) >= SUMMARIZE_AFTER:
                            print("\n[Summarizing conversation to reduce memory...]")
                            conversation_summary = await summarize_conversation(planner_model, history, conversation_summary)
                            history.clear()  # Clear old messages after summarization
                        
                        break

                    elif verdict["verdict"] == "RETRY":
                        retry_count += 1
                        execution_hint = verdict["retry_hint"]

                        print(f"\n Retry {retry_count}/{MAX_RETRIES}")
                        print("Reason:", verdict["reason"])
                        print("Hint:", execution_hint)

                        continue

                    else:  # FAIL
                        print("\n Failed to retrive the answer :")
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