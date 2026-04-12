import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import traceback
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import json
from itertools import cycle

import re
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SUMMARIZE_AFTER = 6
MAX_REACT_STEPS = 6     # max tool calls per turn (Think→Act→Observe loops)
TOOL_TIMEOUT    = 60

# ── MEMORY ─────────────────────────────────────────────────────────────────────
chat_history         = []
conversation_summary = ""

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
# This is the ONLY place that tells the LLM how to behave.
# No hardcoded routing — the LLM sees tool schemas via bind_tools() and decides.

SYSTEM_PROMPT = (
    "You are Sobhan_AI, a helpful, warm, and intelligent personal assistant.\n\n"
    
    "You have access to tools. The tool schemas tell you what each tool does and "
    "what arguments it needs. Read them carefully before calling.\n\n"
    "RULES:\n"
    "1. For greetings or small talk: reply directly. Do NOT call any tools.\n"
    "2. For tasks needing real data (weather, math, translation, search, email): "
    "call the appropriate tool. NEVER make up facts.\n"
    "3. After getting a tool result: explain it naturally in 2-4 sentences.\n"
    "4. For MULTI-STEP tasks (e.g. 'translate hello to French and email it to bob@test.com'):\n"
    "   - Call the first tool (translate)\n"
    "   - Read the result\n"
    "   - Use that result as input for the next tool (email body = translation result)\n"
    "   - The tool-calling loop handles this automatically — just call tools in sequence.\n"
    "5. NEVER call the same tool with the same arguments twice.\n"
    "6. Be concise but warm. Feel like a real assistant.\n"
    "7. Remember context from earlier in the conversation."
)


# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(title="MCP Agent Web Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ──────────────────────────────────────────────────
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# ── APP STATE ──────────────────────────────────────────────────────────────────
backend_initialized = False
backend_components  = {}


# ── HELPERS ────────────────────────────────────────────────────────────────────

async def safe_invoke(model, messages, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return await model.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "503" in err:
                wait = delay * (attempt + 1)
                print(f"\n[Rate limited, retrying in {wait}s...]")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Model failed after {retries} retries")


async def summarize_conversation(model, history, previous_summary=""):
    if not history:
        return previous_summary
    conversation_text = "\n".join([
        f"{'User' if r == 'user' else 'Assistant'}: {c}" for r, c in history
    ])
    response = await safe_invoke(model, [
        SystemMessage(content="You are a conversation summarizer."),
        HumanMessage(content=(
            f"Previous summary: {previous_summary or 'None'}\n\n"
            f"Recent conversation:\n{conversation_text}\n\n"
            f"Write a 2-3 sentence summary preserving names, preferences, and key context:"
        ))
    ])
    return response.content.strip()


def build_messages(summary, history, user_input):
    """Build message list with system prompt, conversation context, and user input."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        messages.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    for role, content in history[-6:]:
        messages.append(
            HumanMessage(content=content) if role == "user"
            else AIMessage(content=content)
        )
    messages.append(HumanMessage(content=user_input))
    return messages


def truncate(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


# ── CORE: ReAct LOOP (LLM decides everything) ─────────────────────────────────

async def react_turn(
    llm_with_tools,
    tools_by_name: dict,
    summary: str,
    history: list,
    user_input: str,
) -> str:
    """
    One conversation turn using LangChain's native tool-calling = ReAct.

    The LLM:
      1. THINKS — decides if a tool is needed (or just replies)
      2. ACTS — calls a tool with arguments it chooses
      3. OBSERVES — sees the tool result via ToolMessage
      4. REPEATS — decides if more tools are needed
      5. ANSWERS — gives final text reply when done

    No hardcoded routing. No intent classification. The LLM sees tool schemas
    from bind_tools() and makes all decisions itself.
    """
    messages = build_messages(summary, history, user_input)

    # Track previous tool calls to detect stuck loops
    last_call_sig = None

    for step in range(MAX_REACT_STEPS):
        response = await safe_invoke(llm_with_tools, messages)

        # ── No tool calls → LLM is done, return its reply ─────────────
        if not response.tool_calls:
            return response.content.strip() if response.content else "I couldn't process that. Please try again."

        # ── Duplicate call guard ───────────────────────────────────────
        current_sig = [
            (tc["name"], json.dumps(tc["args"], sort_keys=True))
            for tc in response.tool_calls
        ]
        if current_sig == last_call_sig:
            # LLM is stuck calling the same thing — force it to answer
            messages.append(response)
            for tc in response.tool_calls:
                messages.append(ToolMessage(
                    content="This tool was already called with these arguments. Please give your final answer now.",
                    tool_call_id=tc["id"],
                ))
            final = await safe_invoke(llm_with_tools, messages)
            return final.content.strip() if final.content else "Something went wrong."
        last_call_sig = current_sig

        # ── Execute tool calls ─────────────────────────────────────────
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            print(f"  [Tool: {tool_name}({json.dumps(tool_args)[:120]})]")

            tool = tools_by_name.get(tool_name)
            if tool is None:
                result = f"Tool '{tool_name}' not found. Available tools: {', '.join(tools_by_name.keys())}"
            else:
                try:
                    raw = await asyncio.wait_for(
                        tool.ainvoke(tool_args),
                        timeout=TOOL_TIMEOUT,
                    )
                    result = truncate(str(raw))
                except asyncio.TimeoutError:
                    result = f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s."
                except Exception as e:
                    result = f"Tool '{tool_name}' error: {str(e)[:200]}"

            # Feed observation back → LLM sees it on next loop iteration
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    # ── Hit max steps → ask LLM to wrap up ─────────────────────────────
    messages.append(HumanMessage(
        content="You've used all available tool calls. Give your best answer based on what you have."
    ))
    final = await safe_invoke(llm_with_tools, messages)
    return final.content.strip() if final.content else "Max steps reached."


# ── LAZY INIT ──────────────────────────────────────────────────────────────────

async def initialize_backend():
    global backend_initialized, backend_components

    if backend_initialized:
        return backend_components

    api_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 4)]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise HTTPException(status_code=500, detail="No GROQ API keys found in .env")

    key_cycle = cycle(api_keys)

    chat_model = ChatGroq(
        model="llama-3.1-8b-instant",
        max_tokens=800,
        temperature=0.7,
        api_key=next(key_cycle),
    )
    agent_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        temperature=0.0,
        api_key=next(key_cycle),
    )

    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path  = "faiss_index"
    index_file  = os.path.join(index_path, "index.faiss")

    faiss_index = (
        FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        if os.path.exists(index_file)
        else FAISS.from_texts(["initial text"], embeddings)
    )
    if not os.path.exists(index_file):
        faiss_index.save_local(index_path)

    python_exec = sys.executable
    mcp_client  = MultiServerMCPClient({
        "math_server": {"command": python_exec, "args": ["-u", "mathserver.py"], "transport": "stdio"},
        "weather":     {"command": python_exec, "args": ["-u", "weather.py"],    "transport": "stdio"},
        "Translate":   {"command": python_exec, "args": ["-u", "translate.py"],  "transport": "stdio"},
        "websearch":   {"command": python_exec, "args": ["-u", "websearch.py"],  "transport": "stdio"},
        "gmail":       {"command": python_exec, "args": ["-u", "gmail.py"],      "transport": "stdio"},
    })

    tools         = await mcp_client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    print("Tool schemas:")
    problematic_tools = []
    for tool in tools:
        try:
            if hasattr(tool.args_schema, 'model_json_schema'):
                schema = tool.args_schema.model_json_schema()
                props  = schema.get('properties', {})
            else:
                schema = tool.args_schema
                props  = schema.get('properties', {}) if isinstance(schema, dict) else {}
            required = schema.get('required', list(props.keys()))
            print(f"  {tool.name}: args={list(props.keys())}, required={required}")
        except Exception as e:
            print(f"  {tool.name}: schema error - {str(e)[:60]}")
            problematic_tools.append(tool.name)

    try:
        llm_with_tools = agent_llm.bind_tools(tools)
    except Exception as e:
        print(f"Warning: Tool binding failed ({str(e)[:100]}), retrying with selective binding")
        working_tools  = [t for t in tools if t.name not in problematic_tools]
        llm_with_tools = agent_llm.bind_tools(working_tools) if working_tools else agent_llm

    print(f"Loaded tools: {list(tools_by_name.keys())}")

    backend_components = {
        "chat_model":     chat_model,
        "llm_with_tools": llm_with_tools,
        "tools_by_name":  tools_by_name,
        "embeddings":     embeddings,
        "faiss_index":    faiss_index,
        "index_path":     index_path,
    }
    backend_initialized = True
    return backend_components


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    await initialize_backend()


@app.get("/", response_class=HTMLResponse)
async def serve_website():
    try:
        with open("website.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="website.html not found")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    global chat_history, conversation_summary

    components = await initialize_backend()
    user_input = message.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        reply = await react_turn(
            llm_with_tools = components["llm_with_tools"],
            tools_by_name  = components["tools_by_name"],
            summary        = conversation_summary,
            history        = chat_history,
            user_input     = user_input,
        )

        chat_history.append(("user", user_input))
        chat_history.append(("assistant", reply))

        components["faiss_index"].add_texts(
            [f"User: {user_input}", f"Assistant: {reply}"],
            metadatas=[
                {"source": "user",      "timestamp": str(datetime.now())},
                {"source": "assistant", "timestamp": str(datetime.now())},
            ],
        )
        components["faiss_index"].save_local(components["index_path"])

        if len(chat_history) >= SUMMARIZE_AFTER * 2:
            print("\n[Summarizing conversation...]")
            split        = len(chat_history) // 2
            to_summarize = chat_history[:split]
            chat_history = chat_history[split:]
            conversation_summary = await summarize_conversation(
                components["chat_model"], to_summarize, conversation_summary
            )

        return ChatResponse(response=reply, status="success")

    except Exception as e:
        traceback.print_exc()
        return ChatResponse(response=f"Error: {str(e).split(chr(10))[0][:200]}", status="error")


@app.post("/clear")
async def clear_chat():
    global chat_history, conversation_summary

    components = await initialize_backend()
    components["faiss_index"] = FAISS.from_texts(
        ["initial text"], components["embeddings"]
    )
    components["faiss_index"].save_local(components["index_path"])
    chat_history         = []
    conversation_summary = ""
    return {"status": "success", "message": "Conversation cleared."}


@app.get("/health")
async def health_check():
    return {
        "status":              "healthy",
        "backend_initialized": backend_initialized,
        "tools_loaded":        list(backend_components.get("tools_by_name", {}).keys()),
        "history_turns":       len(chat_history) // 2,
        "has_summary":         bool(conversation_summary),
        "timestamp":           datetime.now().isoformat(),
    }


# ── CLI MODE (optional) ────────────────────────────────────────────────────────

async def main():
    global chat_history, conversation_summary

    print("Hii I am your personalized AI chatbot here to help you......")

    api_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 4)]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise ValueError("No GROQ API keys found in .env")

    key_cycle = cycle(api_keys)

    chat_model = ChatGroq(
        model="llama-3.1-8b-instant",
        max_tokens=800,
        temperature=0.7,
        api_key=next(key_cycle),
    )

    agent_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        temperature=0.0,
        api_key=next(key_cycle),
    )

    # ── FAISS semantic memory ──────────────────────────────────────────────
    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path  = "faiss_index"
    index_file  = os.path.join(index_path, "index.faiss")
    faiss_index = None

    if os.path.exists(index_file):
        try:
            faiss_index = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index.")
        except Exception as e:
            print(f"FAISS index corrupted ({e}), recreating...")

    if faiss_index is None:
        faiss_index = FAISS.from_texts(["initial text"], embeddings)
        faiss_index.save_local(index_path)
        print("Created new FAISS index.")

    # ── MCP: connect to tool servers ───────────────────────────────────────
    client = MultiServerMCPClient({
        "math_server": {"command": "python", "args": ["-u", "mathserver.py"], "transport": "stdio"},
        "weather":     {"command": "python", "args": ["-u", "weather.py"],    "transport": "stdio"},
        "Translate":   {"command": "python", "args": ["-u", "translate.py"],  "transport": "stdio"},
        "websearch":   {"command": "python", "args": ["-u", "websearch.py"],  "transport": "stdio"},
        "gmail":       {"command": "python", "args": ["-u", "gmail.py"],      "transport": "stdio"},
    })

    tools         = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = agent_llm.bind_tools(tools)

    print(f"Loaded tools: {list(tools_by_name.keys())}")

    # ── CONVERSATION LOOP ──────────────────────────────────────────────────
    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Bye bye!")
                break

            if user_input.lower() == "clear":
                faiss_index = FAISS.from_texts(["initial text"], embeddings)
                faiss_index.save_local(index_path)
                chat_history.clear()
                conversation_summary = ""
                print("Conversation cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)

            try:
                reply = await react_turn(
                    llm_with_tools = llm_with_tools,
                    tools_by_name  = tools_by_name,
                    summary        = conversation_summary,
                    history        = chat_history,
                    user_input     = user_input,
                )

                print(f"\033[92m{reply}\033[0m")

                chat_history.append(("user", user_input))
                chat_history.append(("assistant", reply))

                faiss_index.add_texts(
                    [f"User: {user_input}", f"Assistant: {reply}"],
                    metadatas=[
                        {"source": "user",      "timestamp": str(datetime.now())},
                        {"source": "assistant", "timestamp": str(datetime.now())},
                    ],
                )
                faiss_index.save_local(index_path)

                if len(chat_history) >= SUMMARIZE_AFTER * 2:
                    print("\n[Summarizing conversation...]")
                    split        = len(chat_history) // 2
                    to_summarize = chat_history[:split]
                    chat_history = chat_history[split:]
                    conversation_summary = await summarize_conversation(
                        chat_model, to_summarize, conversation_summary
                    )

            except asyncio.CancelledError:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"\nError: {str(e).split(chr(10))[0][:200]}")
                traceback.print_exc()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        try:
            faiss_index.save_local(index_path)
        except Exception:
            pass
        print("Shutting down...")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        asyncio.run(main())
    else:
        import uvicorn
        uvicorn.run("client:app", host="0.0.0.0", port=8080, reload=True)