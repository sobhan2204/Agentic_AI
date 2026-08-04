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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
from itertools import cycle
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY not set in environment")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Database Session Store
class SessionStore:
    def __init__(self, db_path="sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT PRIMARY KEY,
                    history TEXT,
                    summary TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    mem_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, mem_type, value)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_profile_memory_user
                ON profile_memory(user_id, confirmed, mem_type)
            """)
            
            # ── Migration: Add execution_history if missing ──────────────────
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col[1] for col in cursor.fetchall()]
            if "execution_history" not in columns:
                print("[Database] Migrating sessions table: adding execution_history column")
                cursor.execute("ALTER TABLE sessions ADD COLUMN execution_history TEXT")

            # User-submitted custom MCP tools (sandboxed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_tools (
                    tool_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    script_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, name)
                )
            """)

            conn.commit()

    def create_user(self, username, hashed_password):
        user_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (user_id, username, hashed_password) VALUES (?, ?, ?)",
                    (user_id, username, hashed_password)
                )
                conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None

    def get_user_by_username(self, username):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            return cursor.fetchone()

    def get_session(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "history": json.loads(row["history"]),
                    "summary": row["summary"],
                    "execution_history": json.loads(row["execution_history"]) if row["execution_history"] else []
                }
            return {"history": [], "summary": "", "execution_history": []}

    def save_session(self, user_id, history, summary, execution_history):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            history_json = json.dumps(history)
            exec_json = json.dumps(execution_history)
            cursor.execute("""
                INSERT INTO sessions (user_id, history, summary, execution_history, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    history = excluded.history,
                    summary = excluded.summary,
                    execution_history = excluded.execution_history,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, history_json, summary, exec_json))
            conn.commit()

    def clear_session(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()

    # ── Custom (user-submitted, sandboxed) MCP tools ──────────────────────
    def create_user_tool(self, tool_id: str, user_id: str, name: str, script_path: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_tools (tool_id, user_id, name, script_path, status)
                    VALUES (?, ?, ?, ?, 'active')
                    ON CONFLICT(user_id, name) DO UPDATE SET
                        tool_id = excluded.tool_id,
                        script_path = excluded.script_path,
                        status = 'active',
                        last_error = NULL,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (tool_id, user_id, name, script_path),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def list_user_tools(self, user_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_tools WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_user_tool(self, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_tools WHERE user_id = ? AND name = ?",
                (user_id, name),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_user_tool(self, user_id: str, name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM user_tools WHERE user_id = ? AND name = ?",
                (user_id, name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_user_tool_status(self, user_id: str, name: str, status: str, last_error: str = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_tools SET status = ?, last_error = ? WHERE user_id = ? AND name = ?",
                (status, last_error, user_id, name),
            )
            conn.commit()

    def upsert_profile_memory(
        self,
        user_id: str,
        mem_type: str,
        value: str,
        source: str,
        confidence: float = 0.5,
        confirmed: bool = False,
    ) -> None:
        if not user_id or not mem_type or not value:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO profile_memory (
                    memory_id, user_id, mem_type, value, source,
                    confidence, confirmed, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, mem_type, value) DO UPDATE SET
                    source = excluded.source,
                    confidence = MAX(profile_memory.confidence, excluded.confidence),
                    confirmed = CASE
                        WHEN excluded.confirmed = 1 THEN 1
                        ELSE profile_memory.confirmed
                    END,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                str(uuid.uuid4()),
                user_id,
                mem_type.strip().lower(),
                value.strip(),
                source.strip()[:120],
                float(max(0.0, min(1.0, confidence))),
                1 if confirmed else 0,
            ))
            conn.commit()

    def list_profile_memory(
        self,
        user_id: str,
        confirmed_only: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT mem_type, value, source, confidence, confirmed, created_at, updated_at
                FROM profile_memory
                WHERE user_id = ?
            """
            params: List[Any] = [user_id]
            if confirmed_only:
                query += " AND confirmed = 1"
            query += " ORDER BY updated_at DESC, confidence DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

db_store = SessionStore()

SUMMARIZE_AFTER  = 6
MAX_REACT_STEPS  = 6
TOOL_TIMEOUT     = 60
EVIDENCE_MAX_CHARS = 1200   # max chars per evidence block fed to solver


MCP_TOOL_POOL: Dict[str, Dict[str, str]] = {
    "math": {"server_name": "math_server", "script": "mathserver.py"},
    "weather": {"server_name": "weather", "script": "weather.py"},
    "translate": {"server_name": "Translate", "script": "translate.py"},
    "websearch": {"server_name": "websearch", "script": "websearch.py"},
    "gmail": {"server_name": "gmail", "script": "gmail.py"},
    "archive": {"server_name": "archive", "script": "archive.py"},
    #"spotify": {"server_name": "spotify", "script": "spotify.py"},
    "github": {"server_name": "github", "script": "github.py"},
}

session_store: Dict[str, Dict[str, Any]] = {}


SYSTEM_PROMPT = (
    "You are Sobhan_AI, a helpful, warm, and intelligent personal assistant.\n\n"
    "You have access to tools. The tool schemas tell you what each tool does and "
    "what arguments it needs. Read them carefully before calling.\n\n"
    "RULES:\n"
    "1. For greetings or small talk: reply directly. Do NOT call any tools.\n"
    "2. For tasks needing real data (weather, math, translation, search, email): "
    "call the appropriate tool. NEVER make up facts.\n"
    "3. After getting a tool result: explain it naturally in 2-4 sentences.\n"
    "4. For MULTI-STEP tasks: follow the plan you are given.\n"
    "5. NEVER call the same tool with the same arguments twice.\n"
    "6. Be concise but warm. Feel like a real assistant.\n"
    "7. Remember context from earlier in the conversation."
)

REWO_PLANNER_PROMPT = (
    "You are a strategic planner for a tool-using agent.\n"
    "Generate an executable ReWOO plan using ONLY the provided tools and argument names.\n\n"
    "Available tool schemas:\n{tool_schemas}\n\n"
    "Task mode: {task_mode}\n\n"
    "Resolved Context from History:\n{resolved_context}\n\n"
    "Relevant memory:\n{memory_context}\n\n"
    "Output requirements:\n"
    "1) Return ONLY valid JSON.\n"
    "2) Use this exact structure:\n"
    "{{\"steps\": [{{\"id\": \"E1\", \"tool\": \"tool_name\", \"args\": {{\"key\": \"value\"}}}}]}}\n"
    "3) Use exact tool names and exact arg keys from the schemas above.\n"
    "4) If a step depends on a previous step output, the dependent arg value must be ONLY the placeholder token (for example: {{\"arg_name\": \"$E1\"}}).\n"
    "   Do not write descriptions around placeholders and do not mix placeholder text with other words.\n"
    "5) If no tool is needed, return {{\"steps\": []}}.\n"
    "6) Never invent argument names such as arg1 or arg2 unless they exist in schema.\n"
    "7) If the user asks you to DRAFT, WRITE, or COMPOSE original text (a paragraph, message, "
    "   note, etc.) and then translate/email/send it, you must write that FULL original text "
    "   out yourself as the literal argument value — do not use a one-line stub or summary of "
    "   the topic. For example, if asked to 'draft a paragraph expressing love and translate it', "
    "   the 'sentence' arg for translate must contain the entire drafted paragraph you composed, "
    "   not just 'I love you'.\n"
    "8) Downstream steps that consume previous results in full (e.g. emailing a translation, "
    "   emailing a weather report, emailing search results) must receive the COMPLETE result via "
    "   the placeholder token — never assume only part of the prior output is wanted unless the "
    "   user explicitly asked for a partial detail (e.g. 'just the temperature').\n\n"
    "User query: {user_input}\n"
    "Previous summary: {summary}"
)


REWO_SOLVER_PROMPT = (
    "You are a data aggregator. Your only job is to combine the information blocks "
    "below into a single factual summary. Do NOT format, do NOT add flair.\n\n"
    "RULES:\n"
    "1. Output plain facts only — no greetings, no conclusions, no tone.\n"
    "2. Preserve every number, name, and measurement exactly as given.\n"
    "3. Do NOT mention tools, steps, plans, or any internal process.\n"
    "4. Do NOT add information that is not in the blocks.\n"
    "5. Combine all blocks into one continuous fact dump.\n\n"
    "INFORMATION BLOCKS:\n{evidence}\n\n"
    "User's request (for context only): {user_input}\n\n"
    "Facts:"
)

# Minimal prompt used as last-resort retry
REWO_SOLVER_STRICT_FALLBACK_PROMPT = (
    "List all facts from the information below. Plain text only. No formatting.\n\n"
    "Information:\n{evidence}\n\n"
    "Request context:\n{user_input}\n"
)

REFINER_PROMPT = (
    "You are an expert communicator rewriting a raw fact summary into a polished, "
    "natural response that a friendly human assistant would give.\n\n"
    "STRICT RULES:\n"
    "1. NEVER mention tools, APIs, steps, plans, evidence blocks, or any technical process.\n"
    "2. NEVER use labels like 'Step 1', 'E1', 'tool result', 'fact dump', or 'summary'.\n"
    "3. Sound warm, clear, and conversational — like a knowledgeable friend.\n"
    "4. Preserve EVERY fact, number, name, and measurement from the raw summary. "
    "   Do not hallucinate or add information that isn't there.\n"
    "5. Choose the best format for the query type:\n"
    "   - Travel / itinerary   → day-by-day plan with times and tips. Present it in pointerns.\n"
    "   - Cost / budget        → itemised breakdown ending with a clear total\n"
    "   - Weather              → current conditions + forecast in plain English\n"
    "   - Factual / search     → 2-4 clear sentences\n"
    "   - Translation          → translated text, then a brief register note if useful\n"
    "   - Math / calculation   → answer first, then a one-line explanation\n"
    "   - Email confirmation   → short confirmation, no technical details\n"
    "   - General              → clean paragraphs, no bullet overload\n"
    "6. If information is missing or a lookup failed, acknowledge it gracefully.\n"
    "7. Do NOT start with 'Certainly!', 'Sure!', 'Of course!', or similar filler.\n\n"
    "RAW FACTS:\n{raw_answer}\n\n"
    "User's original request: {user_input}\n\n"
    "Write your polished response now:"
)

CONTEXT_RESOLVER_PROMPT = (
    "You are a Context Resolution Layer for a tool-using AI agent.\n"
    "Your goal is to resolve ambiguous references in the user's query using past tool execution history.\n\n"
    "Execution History (Structured):\n{execution_history}\n\n"
    "Current User Query: {user_input}\n\n"
    "Instruction:\n"
    "1. Identify if the user is referring to an object or result from a previous tool call (e.g., 'it', 'that result', 'more details', 'do it again').\n"
    "2. If yes, find the most relevant previous execution from the history.\n"
    "3. Extract the relevant arguments/context from that execution that should be reused.\n"
    "4. Output a concise JSON mapping of resolved context.\n"
    "5. ONLY reuse arguments that are logically connected to the intent. DO NOT mix contexts (e.g., don't use an email address as a city name).\n\n"
    "Example Output:\n"
    "{{\"referenced_tool\": \"get_weather\", \"resolved_args\": {{\"city\": \"Delhi\"}}, \"reason\": \"User asked for more details about the weather previously looked up for Delhi.\"}}\n\n"
    "If no relevant history is found or the query is explicit, return {{\"referenced_tool\": null, \"resolved_args\": {{}}, \"reason\": \"Query is explicit.\"}}."
)

# FASTAPI APP 
app = FastAPI(title="MCP Agent Web Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST / RESPONSE MODELS 
class ChatMessage(BaseModel):
    message: str
    tools: Optional[List[str]] = None
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class UserCreate(BaseModel):
    username: str
    password: str

class CustomToolCreate(BaseModel):
    name: str
    code: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Auth Utils
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception

# APP STATE 
backend_initialized = False
backend_components: Dict[str, Any] = {}
tool_components_cache: Dict[str, Dict[str, Any]] = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  QUERY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

WEAK_RESPONSE_SIGNALS: List[str] = [
    "i don't know", "i'm not sure", "i cannot", "i can't",
    "as an ai", "i don't have access", "i'm unable",
]

# Improved classifier prompt with explicit examples and KEY RULE
CLASSIFIER_PROMPT = (
    "You are a query router for a tool-using AI assistant.\n"
    "Available tools: {tool_list}\n\n"
    "Classify the user query into exactly one of these categories:\n"
    "- 'simple'       : greetings, small talk, general knowledge answerable from memory (no live data needed)\n"
    "- 'tool-required': needs exactly ONE tool call with a known, explicit input "
    "(e.g. 'weather in Delhi', 'translate hello to French', 'what is 25 * 4')\n"
    "- 'multi-step'   : the input to one tool must first be found using another tool.\n"
    "                   Examples: 'weather where IPL is being played', "
    "'air quality of the city hosting the Olympics', "
    "'temperature at the venue of tomorrow's match'\n\n"
    "KEY RULE: If the location, value, or subject is NOT explicitly stated and must be "
    "LOOKED UP before calling another tool — always classify as 'multi-step'.\n\n"
    "Reply with ONLY one word: simple, tool-required, or multi-step.\n\n"
    "Query: {query}"
)

async def classify_query_type(user_input: str, model_cycle) -> str:
    """LLM-based query classifier — no hardcoded keywords."""
    text = (user_input or "").strip()
    if not text:
        return "simple"
    try:
        response = await safe_invoke(
            model_cycle,
            [SystemMessage(content=CLASSIFIER_PROMPT.format(
                tool_list=", ".join(MCP_TOOL_POOL.keys()),
                query=text
            ))],
        )
        result = (response.content or "").strip().lower()

        if "multi" in result:
            return "multi-step"
        elif "tool" in result:
            return "tool-required"
        else:
            return "simple"
    except Exception as e:
        print(f"[Classifier error: {e}, defaulting to tool-required]")
        return "tool-required"  # safe default: better to over-call tools than ignore them


def is_weak_response(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    lowered = text.lower()
    return any(signal in lowered for signal in WEAK_RESPONSE_SIGNALS)


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLVER SAFEGUARDS
# ═══════════════════════════════════════════════════════════════════════════════

LEAK_TERMS: List[str] = [
    "tool_call", "tool result", "mcp_tool", "rewoo",
    "tool invocation", "function call", "\"tool\":", "\"steps\":",
    "plan step", "args:",
]

LEAK_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in LEAK_TERMS) + r")\b"
    r"|^\s*E\d+[:\s]",
    re.IGNORECASE | re.MULTILINE,
)

POOR_RESPONSE_SIGNALS: List[str] = WEAK_RESPONSE_SIGNALS + LEAK_TERMS + [
    "step e1", "step e2", "e1:", "e2:", "e3:",
]

def is_poor_response(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    lowered = text.lower()
    return any(signal in lowered for signal in POOR_RESPONSE_SIGNALS)


def scrub_response(text: str) -> str:
    lines = text.splitlines()
    clean = [
        line for line in lines
        if not re.match(r"^\s*E\d+[:\s]", line)
        and not LEAK_PATTERN.search(line)
    ]
    result = "\n".join(clean).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


# ═══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def human_label(tool_name: str, tools_by_name: Dict[str, Any] = None) -> str:
    """Generate a human label from the tool's actual description, not a hardcoded map."""
    if tools_by_name and tool_name in tools_by_name:
        tool = tools_by_name[tool_name]
        desc = getattr(tool, "description", "") or ""
        if desc:
            # Take the first sentence of the description as the label
            first_sentence = desc.split(".")[0].strip()
            if first_sentence and len(first_sentence) < 60:
                return first_sentence
    # Fallback: humanize the name
    words = re.sub(r"[_\-]", " ", tool_name)
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", words)
    return words.strip().title()


def clean_result_text(raw: str, max_chars: int = EVIDENCE_MAX_CHARS) -> str:
    if not raw: 
        return ""
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # ── NEW: Preserve news JSON if type is news ──────────────────────
            if parsed.get("type") == "news":
                return text
            
            priority_keys = (
                "result", "text", "translation", "answer",
                "output", "message", "content", "data",
                "summary", "body", "response",
            )
            for key in priority_keys:
                if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                    text = parsed[key].strip()
                    break
            else:
                str_values = {k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()}
                if str_values:
                    text = max(str_values.values(), key=len)
                else:
                    lines=[]
                    for k,v in parsed.items():
                        label = k.replace("_", " ").title()
                        if isinstance(v, (str, int, float, bool)):
                            lines.append(f"{label}: {v}")
                    text = "\n".join(lines) if lines else text
        elif isinstance(parsed, list):
            text = "\n".join(
                json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list))
                else str(item)
                for item in parsed
            )
        elif isinstance(parsed, (str, int, float, bool)):
            text = str(parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    text = re.sub(r"^\s*E\d+[:\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return truncate(text.strip(), max_chars)


def format_evidence_for_solver(evidence: Dict[str, Any]) -> str:
    if not evidence:
        return "No additional information was retrieved."

    blocks:        List[str] = []
    failed_labels: List[str] = []

    for _step_id, payload in evidence.items():
        if not isinstance(payload, dict):
            cleaned = clean_result_text(str(payload))
            if cleaned:
                blocks.append(f"--- Information ---\n{cleaned}")
            continue

        tool_name = payload.get("tool", "")
        label     = human_label(tool_name)

        if payload.get("ok"):
            raw_result = str(payload.get("result", ""))
            cleaned    = clean_result_text(raw_result)
            if cleaned:
                blocks.append(f"--- {label} ---\n{cleaned}")
        else:
            error_msg = str(payload.get("error", "unknown error"))
            failed_labels.append(f"{label} (could not be retrieved: {error_msg[:120]})")

    if failed_labels and not blocks:
        blocks.append(
            "--- Notice ---\n"
            + "\n".join(f"• {f}" for f in failed_labels)
        )
    elif failed_labels:
        blocks.append(
            "--- Note ---\n"
            "Some information could not be retrieved:\n"
            + "\n".join(f"• {f}" for f in failed_labels)
        )

    return "\n\n".join(blocks) if blocks else "No additional information was retrieved."


async def safe_invoke(model_or_cycle, messages, fallback_cycle=None, retries=3, delay=2):
    is_cycle   = hasattr(model_or_cycle, "__next__")
    model      = next(model_or_cycle) if is_cycle else model_or_cycle
    num_keys   = len(backend_components.get("api_keys", [])) or 1
    max_tries  = max(num_keys, 3) if is_cycle else retries

    for attempt in range(max_tries):
        try:
            response = await model.ainvoke(messages)
            if fallback_cycle and not getattr(response, "tool_calls", None):
                if await llm_quality_judge(model_or_cycle, response.content, messages):
                    print("\n[8b gave weak response, escalating to 70b...]")
                    fallback_model = next(fallback_cycle)
                    return await fallback_model.ainvoke(messages)
            return response
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                if is_cycle:
                    if attempt == max_tries - 1 and fallback_cycle:
                        print("\n[All 8b keys rate limited, falling back to 70b...]")
                        fallback_model = next(fallback_cycle)
                        return await fallback_model.ainvoke(messages)
                    model = next(model_or_cycle)
                    print(f"\n[Rate limited, switching to next key (attempt {attempt + 1})]")
                else:
                    wait = delay * (attempt + 1)
                    print(f"\n[Rate limited, retrying in {wait}s...]")
                    await asyncio.sleep(wait)
            elif "503" in err:
                await asyncio.sleep(delay)
            else:
                raise

    raise RuntimeError(f"Model failed after {max_tries} attempts")


async def summarize_conversation(model_or_cycle, history: List[tuple], previous_summary: str = "") -> str:
    if not history:
        return previous_summary
    conversation_text = "\n".join([
        f"{'User' if role == 'user' else 'Assistant'}: {content}"
        for role, content in history
    ])
    response = await safe_invoke(model_or_cycle, [
        SystemMessage(content="You are a conversation summarizer."),
        HumanMessage(content=(
            f"Previous summary: {previous_summary or 'None'}\n\n"
            f"Recent conversation:\n{conversation_text}\n\n"
            "Write a 2-3 sentence summary preserving names, preferences, and key context:"
        )),
    ])
    return response.content.strip()


MEMORY_EXTRACTION_PROMPT = (
    "Extract durable personal memory facts from the user text.\n"
    "Only include facts the user explicitly stated or clearly confirmed.\n"
    "Do NOT infer. Do NOT include temporary requests, one-off tasks, or tool output.\n"
    "Return ONLY valid JSON.\n\n"
    "Schema:\n"
    "{\n"
    '  "facts": [\n'
    "    {\n"
    '      "type": "email|name|location|preference|project|task|relationship|other",\n'
    '      "value": "short normalized fact",\n'
    '      "source": "user_message",\n'
    '      "confidence": 0.0,\n'
    '      "confirmed": true\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "If there are no durable facts, return {\"facts\": []}.\n"
    "Never copy long paragraphs. Keep each value short.\n"
)


def parse_memory_facts(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text:
        return []
    text = raw_text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    facts = parsed.get("facts", []) if isinstance(parsed, dict) else parsed
    if not isinstance(facts, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_type = str(fact.get("type", "other")).strip().lower() or "other"
        value = str(fact.get("value", "")).strip()
        source = str(fact.get("source", "user_message")).strip() or "user_message"
        try:
            confidence = float(fact.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confirmed = bool(fact.get("confirmed", False))
        if not value:
            continue
        cleaned.append({
            "type": fact_type[:40],
            "value": value[:300],
            "source": source[:120],
            "confidence": max(0.0, min(1.0, confidence)),
            "confirmed": confirmed,
        })
    return cleaned


async def extract_memory_facts(model_or_cycle, user_input: str) -> List[Dict[str, Any]]:
    text = (user_input or "").strip()
    if not text:
        return []
    try:
        response = await safe_invoke(
            model_or_cycle,
            [
                SystemMessage(content=MEMORY_EXTRACTION_PROMPT),
                HumanMessage(content=text),
            ],
        )
        return parse_memory_facts((response.content or "").strip())
    except Exception as e:
        print(f"[Memory Extractor Error: {e}]")
        return []


def format_profile_memory_entries(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "No confirmed profile memory."
    lines = []
    for entry in entries:
        lines.append(
            f"- {entry.get('mem_type')}: {entry.get('value')} "
            f"(confirmed={bool(entry.get('confirmed'))}, confidence={entry.get('confidence')}, source={entry.get('source')})"
        )
    return "\n".join(lines)


def memory_entry_text(entry: Dict[str, Any]) -> str:
    return (
        f"{entry.get('mem_type', 'other')}: {entry.get('value', '')} "
        f"[confirmed={bool(entry.get('confirmed'))}, confidence={entry.get('confidence')}]"
    ).strip()


def build_messages(summary: str, history: List[tuple], user_input: str, memory_context: str = "") -> List:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if memory_context:
        messages.append(SystemMessage(content=f"Relevant memory:\n{memory_context}"))
    if summary:
        messages.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    for role, content in history[-6:]:
        messages.append(
            HumanMessage(content=content) if role == "user"
            else AIMessage(content=content)
        )
    messages.append(HumanMessage(content=user_input))
    return messages


def _memory_overlap_score(query: str, text: str) -> int:
    query_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not query_tokens or not text_tokens:
        return 0
    return len(query_tokens & text_tokens)


async def build_relevant_memory_context(
    user_id: str,
    query: str,
    components: Dict[str, Any],
    max_structured: int = 6,
    max_vector: int = 4,
) -> str:
    structured = db_store.list_profile_memory(user_id, confirmed_only=True, limit=30)
    ranked_structured = sorted(
        structured,
        key=lambda entry: (
            _memory_overlap_score(query, f"{entry.get('mem_type')} {entry.get('value')}"),
            float(entry.get("confidence") or 0.0),
            str(entry.get("updated_at") or ""),
        ),
        reverse=True,
    )[:max_structured]

    vector_hits: List[str] = []
    faiss_index = components.get("faiss_index")
    if faiss_index is not None and query.strip():
        try:
            hits = faiss_index.similarity_search(
                query,
                k=max_vector,
                filter={"user_id": user_id},
            )
        except TypeError:
            hits = faiss_index.similarity_search(query, k=max_vector)
        except Exception as e:
            print(f"[Memory Retrieval Error: {e}]")
            hits = []

        for doc in hits:
            content = clean_result_text(getattr(doc, "page_content", "") or "")
            if content:
                vector_hits.append(content[:400])

    blocks = []
    if ranked_structured:
        blocks.append("Confirmed profile facts:\n" + format_profile_memory_entries(ranked_structured))
    if vector_hits:
        vector_lines = "\n".join(f"- {item}" for item in vector_hits)
        blocks.append("Relevant recalled context:\n" + vector_lines)
    return "\n\n".join(blocks)


def truncate(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def get_tool_args_schema(tool: Any) -> Dict[str, Any]:
    try:
        if hasattr(tool.args_schema, "model_json_schema"):
            schema = tool.args_schema.model_json_schema()
        elif isinstance(tool.args_schema, dict):
            schema = tool.args_schema
        else:
            schema = {}
    except Exception:
        schema = {}

    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required   = schema.get("required", [])   if isinstance(schema, dict) else []
    if not required and isinstance(properties, dict):
        required = list(properties.keys())

    return {
        "properties": properties if isinstance(properties, dict) else {},
        "required":   required   if isinstance(required, list)   else [],
    }


def build_tool_schema_map(tools_by_name: Dict[str, Any]) -> Dict[str, Any]:
    schema_map = {}
    for tool_name, tool in tools_by_name.items():
        schema = get_tool_args_schema(tool)
        compact_props = {}
        for prop_name, prop_meta in schema["properties"].items():
            if isinstance(prop_meta, dict):
                compact_props[prop_name] = {
                    "type":        prop_meta.get("type", "any"),
                    "description": prop_meta.get("description", ""),
                }
            else:
                compact_props[prop_name] = {"type": "any", "description": ""}

        schema_map[tool_name] = {
            "required":   schema["required"],
            "properties": compact_props,
        }
    return schema_map


def format_tool_schemas_for_prompt(tools_by_name: Dict[str, Any]) -> str:
    return json.dumps(build_tool_schema_map(tools_by_name), indent=2, ensure_ascii=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN PARSING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def parse_plan_json(plan_text: str) -> Dict[str, Any]:
    if not plan_text:
        return {"steps": []}
    raw = plan_text.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"steps": []}
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return parsed if isinstance(parsed, dict) else {"steps": []}
        except json.JSONDecodeError:
            return {"steps": []}
    return {"steps": []}


def normalize_plan_steps(plan_text: str) -> List[Dict[str, Any]]:
    plan  = parse_plan_json(plan_text)
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        return []

    normalized = []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        args      = step.get("args", {})
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            continue
        step_id = str(step.get("id") or f"E{idx}").strip()
        if not step_id.startswith("E"):
            step_id = f"E{idx}"
        normalized.append({
            "id":   step_id,
            "tool": tool_name.strip(),
            "args": args,
        })
        if len(normalized) >= MAX_REACT_STEPS:
            break
    return normalized


def evidence_value_for_ref(step_id: str, evidence: Dict[str, Any]) -> str:
    payload = evidence.get(step_id)
    if isinstance(payload, dict):
        return str(payload.get("result", "")) if payload.get("ok") else str(payload.get("error", ""))
    return str(payload or "")


def resolve_plan_arg_value(value: Any, evidence: Dict[str, Any]) -> Any:
    """Resolve $E1 / #E1 placeholders using collected step evidence."""
    if isinstance(value, str):
        resolved = value
        for step_id in evidence:
            resolved = resolved.replace(f"${step_id}", evidence_value_for_ref(step_id, evidence))
            resolved = resolved.replace(f"#{step_id}", evidence_value_for_ref(step_id, evidence))
        return resolved
    return value


# ── ADD THIS: LLM-based entity extractor for inter-step references ────────────

# Decide whether an argument is a "content sink" (wants the FULL upstream text
# verbatim — an email body, a translated passage, etc.) versus a "lookup key"
# (wants a short distilled entity — a city name, a number, an ID).
#
# This is derived from each tool's own JSON schema rather than a hand-typed
# list of arg names, so it automatically covers new tools/args without code
# changes, as long as the tool defines its schema reasonably (Pydantic Field
# with maxLength/description, or a JSON schema with the same).
#
# Decision order:
#   1. Explicit schema maxLength <= ~80  -> short value expected (lookup key)
#   2. Explicit schema maxLength > ~80, or type/description signals free text
#      ("text", "body", "message", "paragraph", "content", "sentence" as
#      *description* hints, not just the arg name) -> full content
#   3. No usable schema metadata -> fall back to inspecting the actual
#      upstream value: short upstream results (<=60 chars, no newline) never
#      needed extraction anyway; long upstream results default to full
#      passthrough ONLY if the schema does not cap length, otherwise they go
#      through the entity extractor as before (safe default for unknown tools).

_SHORT_VALUE_SCHEMA_HINTS = ("city", "id", "code", "amount", "number", "year", "language", "mood")
_FREE_TEXT_SCHEMA_HINTS = ("text", "body", "message", "paragraph", "content", "sentence", "summary", "description")

def _arg_schema_meta(tool: Any, arg_name: str) -> Dict[str, Any]:
    schema = get_tool_args_schema(tool)
    return schema["properties"].get(arg_name, {}) if isinstance(schema["properties"], dict) else {}

def _wants_full_content(tool: Any, arg_name: str) -> bool:
    meta = _arg_schema_meta(tool, arg_name)
    if not isinstance(meta, dict):
        meta = {}

    max_len = meta.get("maxLength")
    if isinstance(max_len, int):
        # Schema explicitly bounds this field's length -> trust it directly.
        return max_len > 80

    description = str(meta.get("description", "")).lower()
    arg_lower = arg_name.lower()

    if any(hint in description or hint in arg_lower for hint in _FREE_TEXT_SCHEMA_HINTS):
        # Avoid false positives like "language" containing no overlap, but
        # also don't let a free-text hint override an explicit short hint.
        if not any(hint in arg_lower for hint in _SHORT_VALUE_SCHEMA_HINTS):
            return True

    return False


EXTRACT_ENTITY_PROMPT = (
    "Extract only the specific value needed to answer the question below.\n"
    "Return ONLY one value: a single word or short phrase (not a full sentence).\n"
    "No explanation, no labels, and no extra text.\n"
    "Maximum output length: 60 characters.\n\n"
    "Needed value type: {value_hint}\n"
    "Source text:\n{source_text}\n\n"
    "Extracted value:"
)


def _fallback_extract_short_value(source_text: str) -> str:
    text = (source_text or "").strip()
    if not text:
        return ""

    # First choice: first capitalized proper-noun-like phrase.
    proper_noun_match = re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text)
    if proper_noun_match:
        return proper_noun_match.group(0).strip()

    # Generic backup: first token-like value.
    token_match = re.search(r"\b[^\W_][\w\-]{1,59}\b", text, re.UNICODE)
    if token_match:
        return token_match.group(0).strip()

    return text[:60].strip()

async def extract_entity_from_result(
    model_cycle,
    source_text: str,
    value_hint: str,
) -> str:
    """
    Use a fast LLM call to pull a specific entity (city, number, name, etc.)
    out of a raw tool result paragraph.
    """
    if not source_text or not source_text.strip():
        return source_text

    # Skip extraction if result is already short/clean (< 60 chars, no spaces after strip)
    stripped = source_text.strip()
    if len(stripped) <= 60 and "\n" not in stripped and len(stripped.split()) <= 8:
        return stripped

    try:
        response = await safe_invoke(
            model_cycle,
            [SystemMessage(content=EXTRACT_ENTITY_PROMPT.format(
                value_hint=value_hint,
                source_text=truncate(source_text, 600),
            ))]
        )
        extracted = (response.content or "").strip().strip('"').strip("'")

        if (
            extracted
            and len(extracted) <= 60
            and "\n" not in extracted
            and len(extracted.split()) <= 8
        ):
            return extracted

        return _fallback_extract_short_value(stripped)
    except Exception as e:
        print(f"[Extractor: failed ({str(e)[:60]}), using raw result]")
        return _fallback_extract_short_value(stripped)


def validate_tool_arguments(tool: Any, args_dict: Dict[str, Any]) -> tuple:
    schema   = get_tool_args_schema(tool)
    allowed  = set(schema["properties"].keys())
    required = [name for name in schema["required"] if isinstance(name, str)]

    cleaned_args = {}
    unknown_args = []
    for key, value in args_dict.items():
        if not isinstance(key, str):
            continue
        if allowed and key not in allowed:
            unknown_args.append(key)
            continue
        # ── Flag suspiciously long scalar values, UNLESS this arg's schema
        #    indicates it's a free-text content sink (email body, translate
        #    text, etc) that's supposed to hold full-length content. ───────
        if (
            isinstance(value, str)
            and len(value) > 300
            and not _wants_full_content(tool, key)
        ):
            return False, {}, (
                f"arg '{key}' looks like a raw tool result ({len(value)} chars). "
                "Use extract_entity_from_result() before passing to next tool."
            )
        cleaned_args[key] = value

    missing_required = [
        key for key in required
        if key not in cleaned_args or cleaned_args[key] in (None, "")
    ]
    if unknown_args:
        return False, cleaned_args, f"unknown args: {unknown_args}; allowed: {sorted(list(allowed))}"
    if missing_required:
        return False, cleaned_args, f"missing required args: {missing_required}"
    return True, cleaned_args, ""

LLM_QUALITY_JUDGE_PROMPT = (
    "You are evaluating an AI assistant's response for quality.\n"
    "Answer ONLY with one word: 'good' or 'poor'.\n\n"
    "A response is 'poor' if it:\n"
    "- Is too short (under 20 words) for a non-trivial query\n"
    "- Admits inability without attempting to help\n"
    "- Leaks internal system details (tool names, step IDs, JSON, plan labels)\n"
    "- Contains placeholder text or unresolved references like $E1\n\n"
    "Response to evaluate:\n{text}\n\n"
    "User query (for context):\n{query}\n\n"
    "Quality:"
)

async def llm_quality_judge(model_cycle, text: str, query: str = "") -> bool:
    """Returns True if the response is poor quality. Replaces all keyword-list checks."""
    if not text or len(text.strip()) < 15:
        return True  # trivially poor, skip LLM call
    try:
        response = await safe_invoke(
            model_cycle,
            [SystemMessage(content=LLM_QUALITY_JUDGE_PROMPT.format(
                text=text[:600], query=query[:200]
            ))]
        )
        verdict = (response.content or "").strip().lower()
        return "poor" in verdict
    except Exception:
        return is_weak_response(text)  # graceful fallback to keyword check


def format_evidence_summary_for_fallback(evidence: Dict[str, Any]) -> str:
    lines = []
    for step_id, payload in evidence.items():
        if isinstance(payload, dict) and not payload.get("ok"):
            lines.append(
                f"{step_id} ({payload.get('tool', 'tool')}): "
                f"{payload.get('error', 'unknown failure')}"
            )
        elif isinstance(payload, dict):
            lines.append(f"{step_id}: ok")
    return "\n".join(lines) if lines else "No evidence collected."


def build_mcp_subprocess_env() -> Dict[str, str]:
    """Ensure MCP stdio subprocesses inherit runtime environment in Docker/local."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


CUSTOM_TOOL_PREFIX = "custom:"
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agentic-ai-sandbox:latest")
USER_SCRIPTS_DIR = BASE_DIR / "user_scripts"
# Name of the named Docker volume backing USER_SCRIPTS_DIR (see docker-compose.yml).
# Sandbox containers are launched via the host's Docker daemon (docker.sock is
# bind-mounted, not Docker-in-Docker), so a path like `/app/user_scripts/...`
# means nothing to it — it must be given the volume by name instead.
USER_SCRIPTS_VOLUME = os.getenv("USER_SCRIPTS_VOLUME", "agentic_ai_user_scripts")


def parse_custom_tool_id(tool_id: str) -> Optional[tuple]:
    """custom:<user_id>:<name> -> (user_id, name), else None."""
    if not tool_id.startswith(CUSTOM_TOOL_PREFIX):
        return None
    parts = tool_id.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def make_custom_tool_id(user_id: str, name: str) -> str:
    return f"{CUSTOM_TOOL_PREFIX}{user_id}:{name}"


def normalize_tool_ids(tool_ids: Optional[List[str]], user_id: Optional[str] = None) -> List[str]:
    if not tool_ids:
        return []

    normalized: List[str] = []
    for raw in tool_ids:
        if not isinstance(raw, str):
            continue
        key = raw.strip()
        lowered = key.lower()
        if lowered in MCP_TOOL_POOL:
            if lowered not in normalized:
                normalized.append(lowered)
            continue

        # Custom tools are case-sensitive on name; only allow a caller to
        # reference their OWN custom tools, never another user's.
        if user_id and key.startswith(f"{CUSTOM_TOOL_PREFIX}{user_id}:") and key not in normalized:
            normalized.append(key)
    return normalized


def get_server_name_for_tool_id(tool_id: str) -> str:
    if tool_id in MCP_TOOL_POOL:
        return MCP_TOOL_POOL[tool_id]["server_name"]
    parsed = parse_custom_tool_id(tool_id)
    if parsed:
        user_id, name = parsed
        return f"custom_{user_id}_{name}"
    return tool_id


def build_custom_tool_docker_config(user_id: str, name: str) -> Dict[str, Any]:
    """
    Sandbox config for a user-submitted MCP tool script: a throwaway,
    network-less, resource-capped Docker container with no access to this
    app's environment/secrets. See Dockerfile.sandbox.

    Mounts the `user_scripts` named volume (not a path from inside this
    container) since `docker run` here is executed by the host daemon.
    """
    return {
        "command": "docker",
        "args": [
            "run", "--rm", "-i",
            "--network", "none",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--read-only",
            "-v", f"{USER_SCRIPTS_VOLUME}:/app/user_scripts:ro",
            SANDBOX_IMAGE,
            f"/app/user_scripts/{user_id}/{name}.py",
        ],
        "transport": "stdio",
        "env": {},
    }


def build_mcp_server_configs(
    python_exec: str,
    mcp_env: Dict[str, str],
    tool_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for tool_id in tool_ids:
        if tool_id in MCP_TOOL_POOL:
            tool_def = MCP_TOOL_POOL[tool_id]
            configs[tool_def["server_name"]] = {
                "command": python_exec,
                "args": ["-u", str(BASE_DIR / tool_def["script"])],
                "transport": "stdio",
                "env": mcp_env,
            }
            continue

        parsed = parse_custom_tool_id(tool_id)
        if not parsed:
            continue
        user_id, name = parsed
        row = db_store.get_user_tool(user_id, name)
        if not row or row.get("status") != "active":
            continue
        server_name = get_server_name_for_tool_id(tool_id)
        configs[server_name] = build_custom_tool_docker_config(user_id, name)
    return configs


async def load_mcp_tools_resilient(
    python_exec: str,
    mcp_env: Dict[str, str],
    tool_ids: List[str],
) -> Dict[str, Any]:
    configs = build_mcp_server_configs(python_exec, mcp_env, tool_ids)
    failed_servers: Dict[str, str] = {}
    loaded_tools: List[Any] = []

    try:
        mcp_client = MultiServerMCPClient(configs)
        loaded_tools = await mcp_client.get_tools()
        return {
            "mcp_client": mcp_client,
            "tools": loaded_tools,
            "failed_servers": failed_servers,
        }
    except Exception as exc:
        print(f"[MCP] Bulk tool load failed; retrying per server: {exc}")

    for tool_id in tool_ids:
        server_name = get_server_name_for_tool_id(tool_id)
        try:
            single_config = build_mcp_server_configs(python_exec, mcp_env, [tool_id])
            if not single_config:
                # e.g. a custom tool whose DB row is missing/inactive
                failed_servers[server_name] = "tool config unavailable"
                continue
            single_client = MultiServerMCPClient(single_config)
            loaded_tools.extend(await single_client.get_tools())
        except Exception as exc:
            failed_servers[server_name] = str(exc)
            print(f"[MCP] Failed to load server '{server_name}': {exc}")

    return {
        "mcp_client": None,
        "tools": loaded_tools,
        "failed_servers": failed_servers,
    }


async def get_tool_components(selected_tool_ids: Optional[List[str]]) -> Dict[str, Any]:
    if selected_tool_ids is not None and not selected_tool_ids:
        return {
            "tools_by_name": {},
            "selected_tool_ids": [],
            "failed_servers": {},
        }

    tool_ids = (
        list(MCP_TOOL_POOL.keys())
        if selected_tool_ids is None
        else selected_tool_ids
    )
    selection_key = (
        "all" if selected_tool_ids is None else ",".join(sorted(tool_ids))
    )

    if selection_key in tool_components_cache:
        return tool_components_cache[selection_key]

    base = await initialize_backend()
    load_result = await load_mcp_tools_resilient(
        base["python_exec"], base["mcp_env"], tool_ids
    )
    mcp_client = load_result["mcp_client"]
    tools = load_result["tools"]
    tools_by_name = {tool.name: tool for tool in tools}

    print(f"[Tool pool] Initialized: {tool_ids}")
    print("Tool schemas:")
    problematic_tools: List[str] = []
    for tool in tools:
        try:
            if hasattr(tool.args_schema, "model_json_schema"):
                schema = tool.args_schema.model_json_schema()
                props = schema.get("properties", {})
            else:
                schema = tool.args_schema
                props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", list(props.keys()))
            print(f"  {tool.name}: args={list(props.keys())}, required={required}")
        except Exception as e:
            print(f"  {tool.name}: schema error - {str(e)[:60]}")
            problematic_tools.append(tool.name)

    tool_components_cache[selection_key] = {
        "mcp_client": mcp_client,
        "tools_by_name": tools_by_name,
        "selected_tool_ids": tool_ids,
        "problematic_tools": problematic_tools,
        "failed_servers": load_result["failed_servers"],
    }
    return tool_components_cache[selection_key]


# ═══════════════════════════════════════════════════════════════════════════════
#  ReWOO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_direct_reply(
    model_cycle_router_8b,
    model_cycle_router_70b,
    summary: str,
    history: List[tuple],
    user_input: str,
    memory_context: str = "",
) -> str:
    messages = build_messages(summary, history, user_input, memory_context=memory_context)
    response = await safe_invoke(model_cycle_router_8b, messages, fallback_cycle=model_cycle_router_70b)
    content  = (response.content or "").strip()
    if not content or await llm_quality_judge(model_cycle_router_8b, content, user_input):
        fallback = await safe_invoke(model_cycle_router_70b, messages)
        content  = (fallback.content or "").strip()
    return content or "I could not generate a response right now. Please try again."


async def resolve_context(
    model_cycle_router_70b,
    execution_history: List[Dict[str, Any]],
    user_input: str,
) -> Dict[str, Any]:
    """
    Resolve ambiguous references in the user query using structured execution history.
    """
    if not execution_history:
        return {"referenced_tool": None, "resolved_args": {}, "reason": "No history."}

    # Format history for prompt
    formatted_history = []
    for entry in execution_history[-5:]: # Look at last 5 executions
        formatted_history.append({
            "tool": entry.get("tool"),
            "args": entry.get("args"),
            "summary": clean_result_text(str(entry.get("result", "")))[:200],
            "timestamp": entry.get("timestamp")
        })

    prompt = CONTEXT_RESOLVER_PROMPT.format(
        execution_history=json.dumps(formatted_history, indent=2),
        user_input=user_input
    )

    try:
        response = await safe_invoke(model_cycle_router_70b, [SystemMessage(content=prompt)])
        resolved = parse_plan_json(response.content)
        
        print(f"\n[Context Resolver]\nQuery: {user_input}\nResolved Tool: {resolved.get('referenced_tool')}\nReason: {resolved.get('reason')}\nContext: {resolved.get('resolved_args')}\n")
        
        return resolved
    except Exception as e:
        print(f"[Context Resolver Error: {e}]")
        return {"referenced_tool": None, "resolved_args": {}, "reason": f"Error: {e}"}


async def rewoo_planner(
    model_cycle_router_70b,
    tools_by_name: Dict[str, Any],
    user_input: str,
    summary: str,
    task_mode: str,
    resolved_context: Dict[str, Any] = None,
    memory_context: str = "",
) -> str:
    # Format resolved context for prompt
    ctx_str = "None"
    if resolved_context and resolved_context.get("referenced_tool"):
        ctx_str = json.dumps({
            "previous_tool": resolved_context["referenced_tool"],
            "arguments_to_reuse": resolved_context["resolved_args"],
            "reasoning": resolved_context["reason"]
        }, indent=2)

    prompt = REWO_PLANNER_PROMPT.format(
        tool_schemas=format_tool_schemas_for_prompt(tools_by_name),
        task_mode=task_mode,
        resolved_context=ctx_str,
        memory_context=memory_context or "None",
        user_input=user_input,
        summary=summary or "None",
    )
    response = await safe_invoke(model_cycle_router_70b, [SystemMessage(content=prompt)])
    return (response.content or "").strip()


async def reflect_plan(
    model_cycle_router_70b,
    tools_by_name: Dict[str, Any],
    plan_text: str,
    user_input: str,
) -> str:
    reflection_prompt = (
        "You are validating a ReWOO JSON plan.\n"
        "Return ONLY valid JSON in the same schema: "
        "{\"steps\": [{\"id\": \"E1\", \"tool\": \"name\", \"args\": {}}]}\n"
        "Use only tool names and arg keys from this schema list:\n"
        f"{format_tool_schemas_for_prompt(tools_by_name)}\n\n"
        f"User query: {user_input}\n"
        f"Proposed plan:\n{plan_text}\n\n"
        "If the plan is valid, return it unchanged."
    )
    reflection = await safe_invoke(model_cycle_router_70b, [SystemMessage(content=reflection_prompt)])
    reflected  = (reflection.content or "").strip()
    return reflected if normalize_plan_steps(reflected) else plan_text


# ── REPLACE: rewoo_worker now takes model_cycle and extracts inter-step values ─

async def rewoo_worker(
    plan_text: str,
    tools_by_name: Dict[str, Any],
    model_cycle,
    execution_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    steps = normalize_plan_steps(plan_text)

    print(f"[Worker Debug] normalized step count: {len(steps)}")
    for step in steps:
        if isinstance(step, dict):
            print(
                f"  [Worker Debug] id={step.get('id')} tool={step.get('tool')} args={step.get('args')}"
            )

    if not steps:
        return evidence

    for step in steps:
        step_id   = step["id"]
        tool_name = step["tool"]
        raw_args  = step["args"]
        tool      = tools_by_name.get(tool_name)

        if not tool:
            evidence[step_id] = {
                "ok": False, "tool": tool_name,
                "error": f"tool not found: {tool_name}",
            }
            continue

        # ── Resolve $E-refs with LLM extraction instead of raw substitution ──
        resolved_args = {}
        for key, value in raw_args.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, str) and re.search(r'\$E\d+|#E\d+', value):
                # Find which step is being referenced
                ref_match = re.search(r'[\$#](E\d+)', value)
                ref_id    = ref_match.group(1) if ref_match else None
                raw_ref   = evidence_value_for_ref(ref_id, evidence) if ref_id else value

                if _wants_full_content(tool, key):
                    # This arg is a content sink (email body, translate text, etc) —
                    # the user wants the FULL upstream text here, not a distilled
                    # entity. Pass it through untouched.
                    resolved_args[key] = raw_ref
                    print(f"  [Passthrough] {key}: full content ({len(raw_ref)} chars) carried over")
                else:
                    # Extract just the needed entity using the arg name as the hint
                    extracted = await extract_entity_from_result(
                        model_cycle=model_cycle,
                        source_text=raw_ref,
                        value_hint=key,          # e.g. "city", "query", "amount"
                    )
                    resolved_args[key] = extracted
                    print(f"  [Extractor] {key}: '{raw_ref[:60]}...' → '{extracted}'")
            else:
                resolved_args[key] = resolve_plan_arg_value(value, evidence)

        is_valid, cleaned_args, validation_error = validate_tool_arguments(tool, resolved_args)
        if not is_valid:
            evidence[step_id] = {
                "ok": False, "tool": tool_name,
                "error": validation_error, "args": cleaned_args,
            }
            continue

        print(f"  [Worker] {tool_name} -> {step_id} | args={cleaned_args}")
        try:
            raw_result = await asyncio.wait_for(
                tool.ainvoke(cleaned_args), timeout=TOOL_TIMEOUT
            )
            result_str = str(raw_result)
            evidence[step_id] = {
                "ok": True, "tool": tool_name,
                "args": cleaned_args, "result": result_str,
            }
            # ── NEW: Store successful execution in history ───────────────────
            execution_history.append({
                "tool": tool_name,
                "args": cleaned_args,
                "result": truncate(result_str, 1000),
                "timestamp": datetime.now().isoformat()
            })
        except asyncio.TimeoutError:
            evidence[step_id] = {
                "ok": False, "tool": tool_name, "args": cleaned_args,
                "error": "tool execution timed out",
            }
        except Exception as e:
            evidence[step_id] = {
                "ok": False, "tool": tool_name, "args": cleaned_args,
                "error": f"tool execution failed: {str(e)[:160]}",
            }

    return evidence

async def rewoo_solver(
    model_cycle_router_8b,
    model_cycle_router_70b,
    plan_text: str,
    evidence: Dict[str, Any],
    user_input: str,
    summary: str,
) -> str:
    """
    Combine tool evidence into a raw fact summary.
    Deliberately kept simple — formatting/tone is handled by refine_response().
    """
    # ── NEW: If evidence contains news JSON, return it directly ───────────
    for payload in evidence.values():
        if isinstance(payload, dict) and payload.get("ok"):
            res = str(payload.get("result", ""))
            if '"type": "news"' in res:
                try:
                    # Validate it's actually JSON
                    json.loads(res)
                    return res.strip()
                except:
                    pass

    evidence_block = format_evidence_for_solver(evidence)

    def build_solver_messages(prompt_template: str) -> List:
        return [
            SystemMessage(
                content=prompt_template.format(
                    evidence=evidence_block,
                    user_input=user_input,
                )
            )
        ]

    # Tier 1: fast 8b
    response_8b = await safe_invoke(
        model_cycle_router_8b,
        build_solver_messages(REWO_SOLVER_PROMPT),
    )
    content_8b = (response_8b.content or "").strip()

    if content_8b and not await llm_quality_judge(model_cycle_router_8b, content_8b, user_input):
        return scrub_response(content_8b)

    # Tier 2: 70b
    print("\n[Solver Tier 2: escalating to 70b]")
    response_70b = await safe_invoke(
        model_cycle_router_70b,
        build_solver_messages(REWO_SOLVER_PROMPT),
    )
    content_70b = (response_70b.content or "").strip()

    if content_70b and not await llm_quality_judge(model_cycle_router_70b, content_70b, user_input):
        return scrub_response(content_70b)

    # Tier 3: strict fallback
    print("\n[Solver Tier 3: strict fallback prompt]")
    response_strict = await safe_invoke(
        model_cycle_router_70b,
        build_solver_messages(REWO_SOLVER_STRICT_FALLBACK_PROMPT),
    )
    content_strict = (response_strict.content or "").strip()

    return scrub_response(content_strict) if content_strict else ""


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE REFINER
# ═══════════════════════════════════════════════════════════════════════════════

async def refine_response(
    model_cycle_router_70b,
    user_input: str,
    raw_answer: str,
) -> str:
    """
    Rewrite a raw fact-dump from the solver into a polished, human-like response.

    - Uses the 70b model for best quality output.
    - Preserves every fact; adds warmth, structure, and appropriate formatting.
    - Falls back to the scrubbed raw_answer if the refiner itself fails or
      produces a poor result — so the pipeline never crashes.
    """
    if not raw_answer or not raw_answer.strip():
        return (
            "I was able to retrieve the information you asked for, but I'm having "
            "trouble composing a response right now. Please try again in a moment."
        )

    # ── NEW: Bypass refinement if it's news JSON ───────────────────────────
    if '"type": "news"' in raw_answer:
        return raw_answer.strip()

    refiner_messages = [
        SystemMessage(
            content=REFINER_PROMPT.format(
                raw_answer=raw_answer.strip(),
                user_input=user_input,
            )
        )
    ]

    try:
        print("[Refiner: polishing solver output...]")
        response = await safe_invoke(model_cycle_router_70b, refiner_messages)
        refined  = (response.content or "").strip()

        if not refined or len(refined) < 20:
            print("[Refiner: output too short, using scrubbed raw answer]")
            return scrub_response(raw_answer)

        return scrub_response(refined)

    except Exception as e:
        print(f"[Refiner: error ({str(e)[:80]}), falling back to raw answer]")
        return scrub_response(raw_answer)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ORCHESTRATOR: Hybrid ReWOO + Reflection + Refiner
# ═══════════════════════════════════════════════════════════════════════════════

async def react_turn(model_cycle_router_8b , model_cycle_router_70b , tools_by_name , summary , history , execution_history , user_input,memory_context: str = "" , task_mode="auto",) -> str:
    # FIX 1: Classify only if not already done by caller — no double classification
    if task_mode == "auto":
        route = await classify_query_type(user_input, model_cycle_router_8b)
    else:
        route = task_mode  
    # ── Simple queries bypass the ReWOO pipeline entirely ────────────────────
    if route == "simple":
        print("[Route: simple → direct response]")
        return await generate_direct_reply(
            model_cycle_router_8b=model_cycle_router_8b,
            model_cycle_router_70b=model_cycle_router_70b,
            summary=summary,
            history=history,
            user_input=user_input,
            memory_context=memory_context,
        )

    # ── Step 0: Resolve Context from Execution History ───────────────────────
    resolved_context = await resolve_context(
        model_cycle_router_70b=model_cycle_router_70b,
        execution_history=execution_history,
        user_input=user_input
    )

    # ── Tool-required / multi-step: run full ReWOO + Refiner pipeline ────────
    print(f"[Route: {route} → ReWOO + Reflection + Refiner]")

    print("[ReWOO Planner running...]")
    plan_text = await rewoo_planner(
        model_cycle_router_70b=model_cycle_router_70b,
        tools_by_name=tools_by_name,
        user_input=user_input,
        summary=summary,
        task_mode=route,
        resolved_context=resolved_context,
        memory_context=memory_context,
    )

    if not normalize_plan_steps(plan_text):
        print("[Planner produced no valid steps — falling back to direct response]")
        return await generate_direct_reply(
            model_cycle_router_8b=model_cycle_router_8b,
            model_cycle_router_70b=model_cycle_router_70b,
            summary=summary,
            history=history,
            user_input=user_input,
        )

    print("[Reflection validating planner output...]")
    plan_text = await reflect_plan(
        model_cycle_router_70b=model_cycle_router_70b,
        tools_by_name=tools_by_name,
        plan_text=plan_text,
        user_input=user_input,
    )

    print("[ReWOO Worker executing plan...]")
    evidence = await rewoo_worker(
        plan_text, 
        tools_by_name, 
        model_cycle_router_8b,
        execution_history,
    )
    successful_steps = [p for p in evidence.values() if isinstance(p, dict) and p.get("ok")]

    if not successful_steps:
        print("[All tool calls failed — using graceful fallback answer]")
        fallback_messages = build_messages(summary, history, user_input, memory_context=memory_context) + [
            SystemMessage(
                content=(
                    "The required information could not be retrieved. "
                    "Explain what went wrong clearly and ask the user for corrected input.\n"
                    f"Error context:\n{format_evidence_summary_for_fallback(evidence)}"
                )
            )
        ]
        fallback_response = await safe_invoke(model_cycle_router_70b, fallback_messages)
        fallback_content  = (fallback_response.content or "").strip()
        return (
            fallback_content
            or "I could not run the required tools. "
               "Please provide a more specific request and try again."
        )

    # ── Step 1: Solver — pure fact aggregation ────────────────────────────────
    print("[ReWOO Solver aggregating facts...]")
    raw_answer = await rewoo_solver(
        model_cycle_router_8b=model_cycle_router_8b,
        model_cycle_router_70b=model_cycle_router_70b,
        plan_text=plan_text,
        evidence=evidence,
        user_input=user_input,
        summary=summary,
    )

    # ── Step 2: Refiner — polish into natural human response ─────────────────
    return await refine_response(
        model_cycle_router_70b=model_cycle_router_70b,
        user_input=user_input,
        raw_answer=raw_answer,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKEND INITIALISATION (lazy, idempotent)
# ═══════════════════════════════════════════════════════════════════════════════

_init_lock = asyncio.Lock()

async def initialize_backend() -> Dict[str, Any]:
    global backend_initialized, backend_components

    async with _init_lock:
        if backend_initialized:
            return backend_components

        api_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 4)]
        api_keys = [k for k in api_keys if k]
        if not api_keys:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No GROQ_API_KEY_1..N found in environment. "
                    "Provide them via runtime env vars (recommended in Docker with --env-file) "
                    f"or define them in {BASE_DIR / '.env'}."
                ),
            )

        agent_llm_8b_pool = [
            ChatGroq(model="llama-3.1-8b-instant", max_tokens=1024, temperature=0.0, api_key=k)
            for k in api_keys
        ]
        agent_llm_70b_pool = [
            ChatGroq(model="llama-3.3-70b-versatile", max_tokens=1024, temperature=0.0, api_key=k)
            for k in api_keys
        ]
        chat_model_pool = [
            ChatGroq(model="llama-3.1-8b-instant", max_tokens=800, temperature=0.7, api_key=k)
            for k in api_keys
        ]

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_path = str(BASE_DIR / "faiss_index")
        index_file = os.path.join(index_path, "index.faiss")

        faiss_index = (
            FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            if os.path.exists(index_file)
            else FAISS.from_texts(["initial text"], embeddings)
        )
        if not os.path.exists(index_file):
            faiss_index.save_local(index_path)

        python_exec = sys.executable
        mcp_env = build_mcp_subprocess_env()

        model_cycle_router_8b = cycle(agent_llm_8b_pool)
        model_cycle_router_70b = cycle(agent_llm_70b_pool)
        chat_cycle = cycle(chat_model_pool)

        print(f"Tool pool available: {list(MCP_TOOL_POOL.keys())}")
        print(f"Loaded {len(api_keys)} API key(s) into rotation pool")

        backend_components = {
            "model_cycle_router_8b":  model_cycle_router_8b,
            "model_cycle_router_70b": model_cycle_router_70b,
            "chat_cycle":             chat_cycle,
            "embeddings":             embeddings,
            "faiss_index":            faiss_index,
            "index_path":             index_path,
            "api_keys":               api_keys,
            "python_exec":             python_exec,
            "mcp_env":                 mcp_env,
        }
        backend_initialized = True
        return backend_components


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM (USER-SUBMITTED, SANDBOXED) TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_TOOL_NAME_RE = re.compile(r"[^a-z0-9_]+")
CUSTOM_TOOL_VALIDATE_TIMEOUT = 25  # seconds
MAX_CUSTOM_TOOL_CODE_CHARS = 50_000


def sanitize_custom_tool_name(raw: str) -> Optional[str]:
    name = CUSTOM_TOOL_NAME_RE.sub("_", (raw or "").strip().lower()).strip("_")
    if not name:
        return None
    name = name[:40]
    if name in MCP_TOOL_POOL:
        return None
    return name


async def validate_and_load_custom_tool(user_id: str, name: str) -> Dict[str, Any]:
    """
    Attempt to boot the script in the sandbox container and confirm it
    exposes at least one MCP tool. Never raises — always returns a dict
    with ok/error so a broken script can never take down the caller.
    """
    try:
        config = {"__validate__": build_custom_tool_docker_config(user_id, name)}
        client = MultiServerMCPClient(config)
        tools = await asyncio.wait_for(client.get_tools(), timeout=CUSTOM_TOOL_VALIDATE_TIMEOUT)
        if not tools:
            return {"ok": False, "error": "Script loaded but exposed no @mcp.tool() functions."}
        return {"ok": True, "tool_names": [t.name for t in tools]}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "Script took too long to start (timed out)."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500]}


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    await initialize_backend()
    # Create default admin user for testing
    admin_user = db_store.get_user_by_username("admin")
    if not admin_user:
        hashed_password = get_password_hash("admin")
        db_store.create_user("admin", hashed_password)
        print("[Startup] Created default admin:admin user")
    else:
        print("[Startup] Admin user already exists")


@app.get("/", response_class=HTMLResponse)
async def serve_website():
    try:
        with open(BASE_DIR / "website.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="website.html not found")


@app.post("/register")
async def register(user: UserCreate):
    if db_store.get_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    user_id = db_store.create_user(user.username, hashed_password)
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return {"message": "User registered successfully"}


@app.post("/login", response_model=Token)
async def login(user: UserCreate):
    db_user = db_store.get_user_by_username(user.username)
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": db_user["user_id"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, user_id: str = Depends(get_current_user)):
    components = await initialize_backend()
    user_input = message.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session = db_store.get_session(user_id)
    chat_history = session["history"]
    conversation_summary = session["summary"]
    execution_history = session["execution_history"]

    requested_tool_ids = (
        normalize_tool_ids(message.tools, user_id=user_id)
        if message.tools is not None
        else None
    )
    tool_components = await get_tool_components(requested_tool_ids)
    tools_by_name = tool_components["tools_by_name"]
    memory_context = await build_relevant_memory_context(
        user_id=user_id,
        query=user_input,
        components=components,
    )

    try:
        task_mode = await classify_query_type(user_input, components["model_cycle_router_8b"])

        if task_mode == "simple":
            reply = await generate_direct_reply(
                model_cycle_router_8b=components["model_cycle_router_8b"],
                model_cycle_router_70b=components["model_cycle_router_70b"],
                summary=conversation_summary,
                history=chat_history,
                user_input=user_input,
                memory_context=memory_context,
            )
        else:
            if not tools_by_name:
                reply = "No tools are enabled for this request. Select at least one tool and try again."
            else:
                reply = await react_turn(
                    model_cycle_router_8b=components["model_cycle_router_8b"],
                    model_cycle_router_70b=components["model_cycle_router_70b"],
                    tools_by_name=tools_by_name,
                    summary=conversation_summary,
                    history=chat_history,
                    execution_history=execution_history,
                    user_input=user_input,
                    memory_context=memory_context,
                    task_mode=task_mode,
                )

        chat_history.append(("user", user_input))
        chat_history.append(("assistant", reply))

        components["faiss_index"].add_texts(
            [f"User: {user_input}", f"Assistant: {reply}"],
            metadatas=[
                {"source": "user",      "timestamp": str(datetime.now()), "user_id": user_id},
                {"source": "assistant", "timestamp": str(datetime.now()), "user_id": user_id},
            ],
        )
        components["faiss_index"].save_local(components["index_path"])

        memory_facts = await extract_memory_facts(components["model_cycle_router_8b"], user_input)
        for fact in memory_facts:
            mem_type = fact.get("type", "other")
            value = fact.get("value", "")
            source = fact.get("source", "user_message")
            confidence = float(fact.get("confidence", 0.5))
            confirmed = bool(fact.get("confirmed", False))
            db_store.upsert_profile_memory(
                user_id=user_id,
                mem_type=mem_type,
                value=value,
                source=source,
                confidence=confidence,
                confirmed=confirmed,
            )
            components["faiss_index"].add_texts(
                [f"Memory fact: {mem_type} = {value}"],
                metadatas=[
                    {
                        "source": "profile_memory",
                        "user_id": user_id,
                        "mem_type": mem_type,
                        "confirmed": int(confirmed),
                        "confidence": confidence,
                        "timestamp": str(datetime.now()),
                    }
                ],
            )
        if memory_facts:
            components["faiss_index"].save_local(components["index_path"])

        if len(chat_history) >= SUMMARIZE_AFTER * 2:
            split = len(chat_history) // 2
            to_summarize = chat_history[:split]
            chat_history = chat_history[split:]
            conversation_summary = await summarize_conversation(
                components["chat_cycle"], to_summarize, conversation_summary
            )

        db_store.save_session(user_id, chat_history, conversation_summary, execution_history)

        return ChatResponse(response=reply, status="success")

    except Exception as e:
        traceback.print_exc()
        return ChatResponse(
            response=f"Error: {str(e).split(chr(10))[0][:200]}",
            status="error",
        )


@app.post("/clear")
async def clear_chat(user_id: str = Depends(get_current_user)):
    components = await initialize_backend()
    # Note: Clearing global FAISS index might not be ideal in multi-user, 
    # but I'll stick to the original logic for now or scope it if possible.
    # The requirement says "only clear the session belonging to the authenticated user."
    db_store.clear_session(user_id)
    return {"status": "success", "message": "Conversation cleared."}


@app.post("/tools/custom")
async def create_custom_tool(payload: CustomToolCreate, user_id: str = Depends(get_current_user)):
    name = sanitize_custom_tool_name(payload.name)
    if not name:
        raise HTTPException(
            status_code=400,
            detail="Invalid tool name. Use letters, numbers, and underscores, and avoid names of built-in tools.",
        )

    code = (payload.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Script cannot be empty.")
    if len(code) > MAX_CUSTOM_TOOL_CODE_CHARS:
        raise HTTPException(status_code=400, detail="Script is too large.")

    user_dir = USER_SCRIPTS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    script_path = user_dir / f"{name}.py"
    script_path.write_text(code, encoding="utf-8")

    validation = await validate_and_load_custom_tool(user_id, name)
    if not validation["ok"]:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "status": "error",
            "error": validation["error"],
        }

    tool_id = make_custom_tool_id(user_id, name)
    db_store.create_user_tool(tool_id, user_id, name, str(script_path))

    return {
        "status": "success",
        "tool_id": tool_id,
        "name": name,
        "tools": validation["tool_names"],
    }


@app.get("/tools/custom")
async def list_custom_tools(user_id: str = Depends(get_current_user)):
    rows = db_store.list_user_tools(user_id)
    return {
        "tools": [
            {
                "tool_id": row["tool_id"],
                "name": row["name"],
                "status": row["status"],
                "last_error": row.get("last_error"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }


@app.delete("/tools/custom/{name}")
async def delete_custom_tool(name: str, user_id: str = Depends(get_current_user)):
    safe_name = sanitize_custom_tool_name(name)
    row = db_store.get_user_tool(user_id, safe_name) if safe_name else None
    if not row:
        raise HTTPException(status_code=404, detail="Custom tool not found.")

    try:
        Path(row["script_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    db_store.delete_user_tool(user_id, safe_name)
    return {"status": "success"}


@app.get("/health")
async def health_check():
    return {
        "status":              "healthy",
        "backend_initialized": backend_initialized,
        "tools_loaded":        list(MCP_TOOL_POOL.keys()),
        "tool_pools_cached":   list(tool_components_cache.keys()),
        "timestamp":           datetime.now().isoformat(),
        "agent_mode":          "Hybrid ReWOO + Reflection + Refiner with JWT Auth",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI MODE (optional — run with --cli flag)
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    cli_user_id = "cli"
    session = db_store.get_session(cli_user_id)
    chat_history: List[tuple] = session["history"]
    conversation_summary: str = session["summary"]
    execution_history: List[Dict[str, Any]] = session["execution_history"]

    print("Hii I am your personalized AI chatbot here to help you......")

    api_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 6)]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise ValueError(
            "No GROQ_API_KEY_1..N found in environment. "
            "Provide env vars (or define them in .env for local runs)."
        )

    agent_llm_8b_pool = [
        ChatGroq(model="llama-3.1-8b-instant", max_tokens=1024, temperature=0.0, api_key=k)
        for k in api_keys
    ]
    agent_llm_70b_pool = [
        ChatGroq(model="llama-3.3-70b-versatile", max_tokens=1024, temperature=0.0, api_key=k)
        for k in api_keys
    ]
    chat_model_pool = [
        ChatGroq(model="llama-3.1-8b-instant", max_tokens=800, temperature=0.7, api_key=k)
        for k in api_keys
    ]

    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path  = str(BASE_DIR / "faiss_index")
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

    python_exec = sys.executable
    mcp_env = build_mcp_subprocess_env()
    client = MultiServerMCPClient(
        build_mcp_server_configs(python_exec, mcp_env, list(MCP_TOOL_POOL.keys()))
    )

    tools         = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    llm_8b_pool_with_tools  = [m.bind_tools(tools) for m in agent_llm_8b_pool]
    llm_70b_pool_with_tools = [m.bind_tools(tools) for m in agent_llm_70b_pool]

    model_cycle_router_8b  = cycle(agent_llm_8b_pool)
    model_cycle_router_70b = cycle(agent_llm_70b_pool)
    chat_cycle             = cycle(chat_model_pool)

    print(f"Loaded tools: {list(tools_by_name.keys())}")
    print(f"Loaded {len(api_keys)} API key(s) into rotation pool")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Bye bye!")
                break
            if user_input.lower() == "clear":
                faiss_index = FAISS.from_texts(["initial text"], embeddings)
                faiss_index.save_local(index_path)
                chat_history.clear()
                conversation_summary = ""
                execution_history.clear()
                db_store.clear_session(cli_user_id)
                print("Conversation cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)

            try:
                # FIX 2: Classify once and pass result into react_turn
                task_mode = await classify_query_type(user_input, model_cycle_router_8b)
                reply = await react_turn(
                    model_cycle_router_8b=model_cycle_router_8b,
                    model_cycle_router_70b=model_cycle_router_70b,
                    tools_by_name=tools_by_name,
                    summary=conversation_summary,
                    history=chat_history,
                    execution_history=execution_history,
                    user_input=user_input,
                    memory_context=await build_relevant_memory_context(
                        user_id=cli_user_id,
                        query=user_input,
                        components={"faiss_index": faiss_index},
                    ),
                    task_mode=task_mode,  # pass the already-classified value
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

                memory_facts = await extract_memory_facts(model_cycle_router_8b, user_input)
                for fact in memory_facts:
                    mem_type = fact.get("type", "other")
                    value = fact.get("value", "")
                    source = fact.get("source", "user_message")
                    confidence = float(fact.get("confidence", 0.5))
                    confirmed = bool(fact.get("confirmed", False))
                    db_store.upsert_profile_memory(
                        user_id=cli_user_id,
                        mem_type=mem_type,
                        value=value,
                        source=source,
                        confidence=confidence,
                        confirmed=confirmed,
                    )
                    faiss_index.add_texts(
                        [f"Memory fact: {mem_type} = {value}"],
                        metadatas=[
                            {
                                "source": "profile_memory",
                                "user_id": cli_user_id,
                                "mem_type": mem_type,
                                "confirmed": int(confirmed),
                                "confidence": confidence,
                                "timestamp": str(datetime.now()),
                            }
                        ],
                    )
                if memory_facts:
                    faiss_index.save_local(index_path)

                if len(chat_history) >= SUMMARIZE_AFTER * 2:
                    print("\n[Summarizing conversation...]")
                    split        = len(chat_history) // 2
                    to_summarize = chat_history[:split]
                    chat_history = chat_history[split:]
                    conversation_summary = await summarize_conversation(
                        chat_cycle, to_summarize, conversation_summary
                    )
                db_store.save_session(cli_user_id, chat_history, conversation_summary, execution_history)

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
            db_store.save_session(cli_user_id, chat_history, conversation_summary, execution_history)
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