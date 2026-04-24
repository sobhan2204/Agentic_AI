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

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

SUMMARIZE_AFTER  = 6
MAX_REACT_STEPS  = 6
TOOL_TIMEOUT     = 60
EVIDENCE_MAX_CHARS = 1200   # max chars per evidence block fed to solver

chat_history:         List[tuple] = []
conversation_summary: str         = ""


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
    "Output requirements:\n"
    "1) Return ONLY valid JSON.\n"
    "2) Use this exact structure:\n"
    "{{\"steps\": [{{\"id\": \"E1\", \"tool\": \"tool_name\", \"args\": {{\"key\": \"value\"}}}}]}}\n"
    "3) Use exact tool names and exact arg keys from the schemas above.\n"
    "4) If a step depends on a previous step output, the dependent arg value must be ONLY the placeholder token (for example: {{\"arg_name\": \"$E1\"}}).\n"
    "   Do not write descriptions around placeholders and do not mix placeholder text with other words.\n"
    "5) If no tool is needed, return {{\"steps\": []}}.\n"
    "6) Never invent argument names such as arg1 or arg2 unless they exist in schema.\n\n"
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
    "Request context: {user_input}\n\n"
    "Facts:"
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
    "   - Travel / itinerary   → day-by-day plan with times and tips\n"
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

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# APP STATE 
backend_initialized = False
backend_components: Dict[str, Any] = {}


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
    "Available tools: weather, air quality, web search, math/calculator, translation, email.\n\n"
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
            [SystemMessage(content=CLASSIFIER_PROMPT.format(query=text))],
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

_LEAK_TERMS: List[str] = [
    "tool_call", "tool result", "mcp_tool", "rewoo",
    "tool invocation", "function call", "\"tool\":", "\"steps\":",
    "plan step", "args:",
]

_LEAK_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _LEAK_TERMS) + r")\b"
    r"|^\s*E\d+[:\s]",
    re.IGNORECASE | re.MULTILINE,
)

_POOR_RESPONSE_SIGNALS: List[str] = WEAK_RESPONSE_SIGNALS + _LEAK_TERMS + [
    "step e1", "step e2", "e1:", "e2:", "e3:",
]

def is_poor_response(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    lowered = text.lower()
    return any(signal in lowered for signal in _POOR_RESPONSE_SIGNALS)


def scrub_response(text: str) -> str:
    lines = text.splitlines()
    clean = [
        line for line in lines
        if not re.match(r"^\s*E\d+[:\s]", line)
        and not _LEAK_PATTERN.search(line)
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


def build_messages(summary: str, history: List[tuple], user_input: str) -> List:
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
        # ── NEW: flag suspiciously long scalar values ──────────────────────
        if isinstance(value, str) and len(value) > 300:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ReWOO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_direct_reply(
    model_cycle_router_8b,
    model_cycle_router_70b,
    summary: str,
    history: List[tuple],
    user_input: str,
) -> str:
    messages = build_messages(summary, history, user_input)
    response = await safe_invoke(model_cycle_router_8b, messages, fallback_cycle=model_cycle_router_70b)
    content  = (response.content or "").strip()
    if not content or await llm_quality_judge(model_cycle_router_8b, content, user_input):
        fallback = await safe_invoke(model_cycle_router_70b, messages)
        content  = (fallback.content or "").strip()
    return content or "I could not generate a response right now. Please try again."


async def rewoo_planner(
    model_cycle_router_70b,
    tools_by_name: Dict[str, Any],
    user_input: str,
    summary: str,
    task_mode: str,
) -> str:
    prompt = REWO_PLANNER_PROMPT.format(
        tool_schemas=format_tool_schemas_for_prompt(tools_by_name),
        task_mode=task_mode,
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
    model_cycle,                          # ← new param (pass model_cycle_router_8b)
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
            evidence[step_id] = {
                "ok": True, "tool": tool_name,
                "args": cleaned_args, "result": str(raw_result),
            }
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

async def react_turn(
    model_cycle_router_8b,
    model_cycle_router_70b,
    tools_by_name,
    summary,
    history,
    user_input,
    task_mode="auto",
) -> str:
    # FIX 1: Classify only if not already done by caller — no double classification
    if task_mode == "auto":
        route = await classify_query_type(user_input, model_cycle_router_8b)
    else:
        route = task_mode  # already classified upstream — trust it

    # ── Simple queries bypass the ReWOO pipeline entirely ────────────────────
    if route == "simple":
        print("[Route: simple → direct response]")
        return await generate_direct_reply(
            model_cycle_router_8b=model_cycle_router_8b,
            model_cycle_router_70b=model_cycle_router_70b,
            summary=summary,
            history=history,
            user_input=user_input,
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
    evidence = await rewoo_worker(plan_text, tools_by_name, model_cycle_router_8b)
    successful_steps = [p for p in evidence.values() if isinstance(p, dict) and p.get("ok")]

    if not successful_steps:
        print("[All tool calls failed — using graceful fallback answer]")
        fallback_messages = build_messages(summary, history, user_input) + [
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

async def initialize_backend() -> Dict[str, Any]:
    global backend_initialized, backend_components

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
    mcp_env     = build_mcp_subprocess_env()
    mcp_client  = MultiServerMCPClient({
        "math_server": {"command": python_exec, "args": ["-u", str(BASE_DIR / "mathserver.py")], "transport": "stdio", "env": mcp_env},
        "weather":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "weather.py")],    "transport": "stdio", "env": mcp_env},
        "Translate":   {"command": python_exec, "args": ["-u", str(BASE_DIR / "translate.py")],  "transport": "stdio", "env": mcp_env},
        "websearch":   {"command": python_exec, "args": ["-u", str(BASE_DIR / "websearch.py")],  "transport": "stdio", "env": mcp_env},
        "gmail":       {"command": python_exec, "args": ["-u", str(BASE_DIR / "gmail.py")],      "transport": "stdio", "env": mcp_env},
        "archive":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "archive.py")],   "transport": "stdio", "env": mcp_env},
        "spotify":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "spotify.py")],   "transport": "stdio", "env": mcp_env},
    })

    tools         = await mcp_client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    print("Tool schemas:")
    problematic_tools: List[str] = []
    for tool in tools:
        try:
            if hasattr(tool.args_schema, "model_json_schema"):
                schema = tool.args_schema.model_json_schema()
                props  = schema.get("properties", {})
            else:
                schema = tool.args_schema
                props  = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", list(props.keys()))
            print(f"  {tool.name}: args={list(props.keys())}, required={required}")
        except Exception as e:
            print(f"  {tool.name}: schema error - {str(e)[:60]}")
            problematic_tools.append(tool.name)

    try:
        llm_8b_pool_with_tools  = [m.bind_tools(tools) for m in agent_llm_8b_pool]
        llm_70b_pool_with_tools = [m.bind_tools(tools) for m in agent_llm_70b_pool]
    except Exception as e:
        print(f"Warning: Tool binding failed ({str(e)[:100]}), retrying with selective binding")
        working_tools = [t for t in tools if t.name not in problematic_tools]
        if working_tools:
            llm_8b_pool_with_tools  = [m.bind_tools(working_tools) for m in agent_llm_8b_pool]
            llm_70b_pool_with_tools = [m.bind_tools(working_tools) for m in agent_llm_70b_pool]
        else:
            llm_8b_pool_with_tools  = agent_llm_8b_pool
            llm_70b_pool_with_tools = agent_llm_70b_pool

    model_cycle_router_8b  = cycle(agent_llm_8b_pool)
    model_cycle_router_70b = cycle(agent_llm_70b_pool)
    model_cycle_8b         = cycle(llm_8b_pool_with_tools)
    model_cycle_70b        = cycle(llm_70b_pool_with_tools)
    chat_cycle             = cycle(chat_model_pool)

    print(f"Loaded tools: {list(tools_by_name.keys())}")
    print(f"Loaded {len(api_keys)} API key(s) into rotation pool")

    backend_components = {
        "model_cycle_router_8b":  model_cycle_router_8b,
        "model_cycle_router_70b": model_cycle_router_70b,
        "model_cycle_8b":         model_cycle_8b,
        "model_cycle_70b":        model_cycle_70b,
        "chat_cycle":             chat_cycle,
        "tools_by_name":          tools_by_name,
        "embeddings":             embeddings,
        "faiss_index":            faiss_index,
        "index_path":             index_path,
        "api_keys":               api_keys,
    }
    backend_initialized = True
    return backend_components


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    await initialize_backend()


@app.get("/", response_class=HTMLResponse)
async def serve_website():
    try:
        with open(BASE_DIR / "website.html", "r", encoding="utf-8") as f:
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
        # FIX 1: Classify exactly once — result is passed through and never re-classified
        task_mode = await classify_query_type(user_input, components["model_cycle_router_8b"])

        if task_mode == "simple":
            print("[Chat endpoint: simple query — bypassing ReWOO]")
            reply = await generate_direct_reply(
                model_cycle_router_8b=components["model_cycle_router_8b"],
                model_cycle_router_70b=components["model_cycle_router_70b"],
                summary=conversation_summary,
                history=chat_history,
                user_input=user_input,
            )
        else:
            # FIX 1: Removed duplicate classify_query_type call that was here
            reply = await react_turn(
                model_cycle_router_8b=components["model_cycle_router_8b"],
                model_cycle_router_70b=components["model_cycle_router_70b"],
                tools_by_name=components["tools_by_name"],
                summary=conversation_summary,
                history=chat_history,
                user_input=user_input,
                task_mode=task_mode,  # pass the already-classified value
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
                components["chat_cycle"], to_summarize, conversation_summary
            )

        return ChatResponse(response=reply, status="success")

    except Exception as e:
        traceback.print_exc()
        return ChatResponse(
            response=f"Error: {str(e).split(chr(10))[0][:200]}",
            status="error",
        )


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
        "agent_mode":          "Hybrid ReWOO + Reflection + Refiner",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI MODE (optional — run with --cli flag)
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    global chat_history, conversation_summary

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
    client = MultiServerMCPClient({
        "math_server": {"command": python_exec, "args": ["-u", str(BASE_DIR / "mathserver.py")], "transport": "stdio", "env": mcp_env},
        "weather":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "weather.py")],    "transport": "stdio", "env": mcp_env},
        "Translate":   {"command": python_exec, "args": ["-u", str(BASE_DIR / "translate.py")],  "transport": "stdio", "env": mcp_env},
        "websearch":   {"command": python_exec, "args": ["-u", str(BASE_DIR / "websearch.py")],  "transport": "stdio", "env": mcp_env},
        "gmail":       {"command": python_exec, "args": ["-u", str(BASE_DIR / "gmail.py")],      "transport": "stdio", "env": mcp_env},
        "archive":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "archive.py")],   "transport": "stdio", "env": mcp_env},
        "spotify":     {"command": python_exec, "args": ["-u", str(BASE_DIR / "spotify.py")],   "transport": "stdio", "env": mcp_env},

    })

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
                    user_input=user_input,
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

                if len(chat_history) >= SUMMARIZE_AFTER * 2:
                    print("\n[Summarizing conversation...]")
                    split        = len(chat_history) // 2
                    to_summarize = chat_history[:split]
                    chat_history = chat_history[split:]
                    conversation_summary = await summarize_conversation(
                        chat_cycle, to_summarize, conversation_summary
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