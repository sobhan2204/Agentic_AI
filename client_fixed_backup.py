from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
import json
from rule_based_verifier import rule_based_verifier
from executioner import execute_plan
from itertools import cycle
from intent_router import detect_intent, resolve_tool


MAX_RETRIES = 2
SUMMARIZE_AFTER = 5  # Summarize conversation after every 10 exchanges

history = []
conversation_summary = ""

def summarize_text(text: str, max_tokens: int = 500) -> str:
    """Shortening the text to approximate token limit (by taking the rough estimate: 4 chars = 1 token)."""
    if not text:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [Shoterned]"

def normalize_plan(raw):
    if isinstance(raw, dict):
        return raw

    # If string, clean and parse
    if isinstance(raw, str):
        raw = raw.strip()

        # Remove surrounding quotes if present
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
            raw = raw.replace('\\"', '') # this function will remove any "\\" and change it into blank space 

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
    CLASSIFIER_PROMPT = """Classify user input as:
- CONVERSATIONAL: greetings (for example : "hi", "hello"), personal questions (for example :"what's my name?"), follow-ups ("tell me more", "elaborate") and any otherconversation that you can answer based on your LLM .
- TASK: needs external tools/data - weather, math, translation, web search, product info, recommendations, comparisons, prices, current information.

Examples:
- "so I want to buy an iphone which is the best?" → TASK (needs search for current models)
- "what's the weather?" → TASK
- "tell me more about that" → CONVERSATIONAL
- "hello how are you" → CONVERSATIONAL
- "which laptop is better?" → TASK (needs search/comparison)

Respond with one word: CONVERSATIONAL or TASK"""
    
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    response = await model.ainvoke(messages)
    classification = response.content.strip().upper()
    
    return classification == "CONVERSATIONAL"


async def plan_task(model , user_input):
    
    # Planner prompt with exact tool names
    PLANNER_PROMPT = """You are a task planner.

Your job is to convert the user query into a minimal, executable plan.
You do NOT execute tools and you do NOT describe tool internals, you give what to do for each step.
given the user query, break it down into clear steps using ONLY the available tools.

Output JSON ONLY. No explanations.

Available tools (use EXACT names):
- solve_math (for any math calculations)
- get_current_weather (requires lat/lon - use get_geo_details first if city name given)
- get_geo_details (convert city name to lat/lon)
- translate (requires sentence and target language code)
- search_web, find_relevant_urls (for web searches)
- send_email, read_emails (for Gmail operations)

Schema:
{
  "goal": "<high-level user goal>",
  "steps": [
    {
      "id": 1,
      "tool": "<tool name or 'none'>",
      "input": "<clean, explicit input for the tool>"
    }
  ]
}

Rules (STRICT):
1. Use ONLY tools from the list above.
2. If the query is ambiguous or missing required information, create a step with tool = "none" asking for clarification.
3. Each step must have ONE clear purpose.
4. Do NOT describe how a tool works.
5. Do NOT include reasoning, explanations, or commentary.
6. Keep the plan minimal: try to keep it 3 steps, you can have more only if absolutely necessaryand for complex tasks.
7. The "input" field must be directly usable by the executor without interpretation.
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
        print("Failed to parse plan.")
        return None


def intent(user_input: str):
    """Run deterministic intent detection before planning."""
    intent_result = detect_intent(user_input)

    if intent_result["needs_clarification"]:
        return {
            "type": "clarification",
            "message": intent_result["clarification_question"],
            "intent_result": intent_result,
        }

    mapped_tool = resolve_tool(intent_result["intent"])

    planner_input = (
        f"User query: {user_input}\n"
        f"Detected intent: {intent_result['intent']}\n"
        f"Entities: {intent_result.get('entities', {})}\n"
        f"Suggested tool: {mapped_tool or 'none'}"
    )

    return {
        "type": "route",
        "planner_input": planner_input,
        "tool": mapped_tool,
        "intent_result": intent_result,
    }

async def main():
    """Run a chat using MCPAgent's built-in conversation memory and all the availabe tools."""
    global history, conversation_summary

    load_dotenv()  

    print("Hii I am your personalized AI chatbot here to help you......")

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
        
    # Create cycle iterator for key rotation
    key_cycle = cycle(api_keys)

    # CREATING LLM MODEL
    tools = await clients.get_tools()
    # print(f"\nLoaded {len(tools)} tools: {[tool.name for tool in tools]}\n")
    
    planner_model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=2000,  # Reduced to leave room for input
    temperature=0,
    api_key=next(key_cycle)
    )

    executor_model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=2000,  # Reduced to leave room for input
    temperature=0.0,
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
3. YOU can add a little context from your LLM knowledge BUT only when latest information is NOT asked by the user. Do NOT make up current data especially when when using tools like weather or web search(for latest information).
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

When you answer, provide a clear, helpful explanation with a bit of depth:
- Prefer 3–5 sentences or a short list of key points
- Avoid one-line answers unlessthe output returned was not as expected or when the question is extremely simple

Respond naturally and conversationally and always try to keep the coversation alive . Be helpful but don't overuse tools."""
    
    # AGGRESSIVE message trimming to stay under token limits
    # Keep only system message + last 4 messages (2 exchanges)
    def trim_message_history(messages):
        """Trim messages aggressively to prevent token overflow."""
        if len(messages) <= 5:  # System + 4 messages
            return messages
        
        # Always keep the first message (system prompt)
        system_msg = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
        
        # Keep only last 4 messages for conversation
        recent_messages = messages[-4:]
        
        if system_msg:
            return [system_msg] + recent_messages
        return recent_messages
    
    # Create agent with memory 
    agent = create_agent(
        executor_model, 
        tools, 
    )
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # This is for embedding all the data fromthe user and the ai into vector db for sementic search
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
            
            print("\nAssistant: ", end="", flush=True)
            
            try:
                is_conversational = await is_conversational_query(planner_model, user_input)
                
                # Retrieve relevant context from FAISS for ALL queries
                try:
                    # Check for memory/recall keywords
                    memory_keywords = ["remember", "recall", "earlier", "before", "previous", "you said", "you wrote", "you told","my name", "who am i"]
                    is_memory_query = any(keyword in user_input.lower() for keyword in memory_keywords)
                    
                    if is_memory_query or is_conversational:
                        # Increase k for memory/personal queries
                        k_value = 10 if is_memory_query else 5
                        relevant_context = faiss_index.similarity_search(user_input, k=k_value)
                        
                        # For personal info queries, also search for the actual query terms
                        # if any(word in user_input.lower() for word in ["my name", "who am i"]):
                        #     name_context = faiss_index.similarity_search("name is", k=5)
                        #     relevant_context = relevant_context + name_context
                        
                        context_text = "\n".join([doc.page_content for doc in relevant_context if doc.page_content != "initial text"])
                        # Limit context to 800 tokens for memory queries, 500 for others
                        max_tokens = 800 if is_memory_query else 500
                        context_text = summarize_text(context_text, max_tokens=max_tokens)
                    else:
                        # Minimal context for regular TASK queries
                        context_text = ""
                except:
                    context_text = ""
                
                if is_conversational:
                    # Build context with summary and recent messages
                    full_context = ""
                    if conversation_summary:
                        full_context += f"Previous conversation summary: {conversation_summary}\n\n"
                    if context_text:
                        full_context += f"Relevant previous messages:\n{context_text}\n\n"
                        full_context += "IMPORTANT: Use the information above to answer the user's question.\n\n"
                    
                    message_with_context = f"{full_context}Current query: {user_input}" if full_context else user_input
                    # Direct response without planning
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=message_with_context)]},
                        config=config
                    )
                    assistant_message = response["messages"][-1].content
                    print(f"\033[92m{assistant_message}\033[0m")  # to make assistant reply Green color
                    
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
                
                # Complex query: use intent router first, then planner
                routing = intent(user_input)

                # If clarification is needed, respond directly and skip planning
                if routing["type"] == "clarification":
                    print(f"\033[92m{routing['message']}\033[0m")
                    continue

                # Use the enhanced planner input from intent router
                planner_input = routing["planner_input"]
                
                # Add memory context if this is a recall query
                if context_text and any(keyword in user_input.lower() for keyword in ["remember", "recall", "earlier", "previous"]):
                    planner_input = f"This is the plan to the goal : \n{planner_input}"
                
                plan = await plan_task(planner_model, planner_input)
                
                if plan is None:
                    print("Unable to create plan. Please rephrase your query.")
                    continue
                    
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
                        print(f"\033[92m{final_answer}\033[0m")  # Green color
                        
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
                        continue

                    else:  # FAIL
                        print(f"\033[91mUnable to answer: {verdict['reason']}\033[0m")  # Red color
                        break


            except Exception as e:
                # Concise error message - no tracebacks
                error_msg = str(e).split('\n')[0][:200]  # First line only, max 200 chars
                print(f"\nError: {error_msg}")
                print("Please try again.")

            
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Clean up clients if needed
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())