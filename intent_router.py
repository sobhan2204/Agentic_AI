import os
import re
import json
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ── ONLY regex that stays: email address detection (always unambiguous) ────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ── LLM SETUP ──────────────────────────────────────────────────────────────────
_router_llm = None

def _get_llm() -> ChatGroq:
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=400,
            api_key=os.getenv("GROQ_API_KEY_1"),
        )
    return _router_llm


# ── PROMPT ─────────────────────────────────────────────────────────────────────
ROUTER_PROMPT = """You are a task router for an AI assistant. Analyze the user query and return a JSON plan.

Available tools:
- solve_math          : solve any math expression or calculation
- get_current_weather : get weather for a city
- translate           : translate text to another language
- web_search          : search the web for information, news, facts
- send_email          : send an email (needs recipient, subject, body)
- read_emails         : read/check emails from inbox

Rules:
1. Return ONLY valid JSON — no explanation, no markdown, no extra text.
2. For multi-step queries (e.g. "search X and email it"), return multiple steps in order.
3. For send_email: always extract recipient, subject, body as separate fields.
4. For chained tasks where body = previous step output: set body to empty string "".
5. Keep tool inputs minimal — city name only for weather, expression only for math.
6. confidence: how sure you are this plan is correct (0.0 to 1.0).

Schema:
{
  "steps": [
    {
      "tool": "<tool_name>",
      "input": "<clean input for non-email tools>",
      "recipient": "<email address or empty string>",
      "subject": "<email subject or empty string>",
      "body": "<email body, or empty string if body = previous step output>"
    }
  ],
  "confidence": <0.0-1.0>
}

Examples:

Query: "what is the weather in London"
{"steps": [{"tool": "get_current_weather", "input": "London", "recipient": "", "subject": "", "body": ""}], "confidence": 1.0}

Query: "translate 'good morning' to French"
{"steps": [{"tool": "translate", "input": "good morning → French", "recipient": "", "subject": "", "body": ""}], "confidence": 1.0}

Query: "search for latest AI news and email the results to bob@example.com with subject AI Update"
{"steps": [{"tool": "web_search", "input": "latest AI news", "recipient": "", "subject": "", "body": ""}, {"tool": "send_email", "input": "", "recipient": "bob@example.com", "subject": "AI Update", "body": ""}], "confidence": 1.0}

Query: "send an email to alice@test.com with subject Hello and body How are you?"
{"steps": [{"tool": "send_email", "input": "", "recipient": "alice@test.com", "subject": "Hello", "body": "How are you?"}], "confidence": 1.0}

Query: "what is 5 + 5"
{"steps": [{"tool": "solve_math", "input": "5 + 5", "recipient": "", "subject": "", "body": ""}], "confidence": 1.0}

Query: "check my emails"
{"steps": [{"tool": "read_emails", "input": "inbox", "recipient": "", "subject": "", "body": ""}], "confidence": 1.0}
"""


def _call_router_llm(query: str) -> Dict[str, Any]:
    """Call LLM router and parse the JSON response."""
    llm = _get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ])
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "steps": [{"tool": "web_search", "input": query,
                       "recipient": "", "subject": "", "body": ""}],
            "confidence": 0.4,
        }
    except Exception:
        return {
            "steps": [{"tool": "web_search", "input": query,
                       "recipient": "", "subject": "", "body": ""}],
            "confidence": 0.3,
        }


def _build_intent(step: dict) -> dict:
    """
    Convert a raw LLM step into a structured intent.
    For send_email: pack recipient/subject/body into the pipe format the executioner expects.
    For all other tools: input field is used directly.
    """
    tool = step.get("tool", "web_search")

    if tool == "send_email":
        recipient  = step.get("recipient", "").strip()
        subject    = step.get("subject", "Hello").strip() or "Hello"
        body       = step.get("body", "").strip()
        tool_input = f"recipient={recipient} | subject={subject} | body={body}"
    else:
        tool_input = step.get("input", "").strip()

    return {
        "name":       tool,
        "input":      tool_input,
        "extraction": "llm",
        "confidence": 1.0,
        "entities":   {},
    }


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def route_intent(query: str) -> Dict[str, Any]:
    """
    Route a user query to one or more tools using LLM understanding.
    Returns a structured route dict compatible with build_plan() in client.py.
    """
    parsed     = _call_router_llm(query)
    steps      = parsed.get("steps", [])
    confidence = parsed.get("confidence", 0.5)

    if not steps:
        steps      = [{"tool": "web_search", "input": query,
                       "recipient": "", "subject": "", "body": ""}]
        confidence = 0.3

    intents   = [_build_intent(step) for step in steps]
    task_type = "multi" if len(intents) > 1 else "single"

    return {
        "task_type":  task_type,
        "intents":    intents,
        "complex":    task_type == "multi",
        "confidence": confidence,
        "source":     "llm",
        "execution_hints": {
            "needs_sequential_execution": task_type == "multi",
            "priority": "normal",
        },
    }


# Keep old signatures so nothing else breaks
def detect_intent(query: str) -> Dict[str, Any]:
    route   = route_intent(query)
    intents = route.get("intents", [{}])
    first   = intents[0] if intents else {}
    return {
        "intent":                 first.get("name", "web_search"),
        "confidence":             route.get("confidence", 0.5),
        "entities":               first.get("entities", {}),
        "reasoning":              "",
        "needs_clarification":    route.get("confidence", 0.5) < 0.35,
        "clarification_question": "Could you please clarify what you'd like to do?",
    }


def resolve_tool(intent_name: str) -> Optional[str]:
    return intent_name


# ── CLI TEST ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "what is the weather in London?",
        "solve 5 + 5",
        "translate 'good morning' to French",
        "search for latest AI news",
        "search for AI news and email the results to pandasobhan22@gmail.com",
        "Search for AI news and then email the results to pandasobhan22@gmail.com",
        "send an email to bob@test.com with subject Hello and body How are you?",
        "what's the weather in the city where the Eiffel Tower is?",
        "check my emails",
        "search epstein files and send them to john@test.com with subject Important",
        "translate This is good to spanish",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = route_intent(q)
        print(f"  Confidence: {result['confidence']} | Type: {result['task_type']}")
        for i in result["intents"]:
            print(f"  → Tool: {i['name']} | Input: {i['input']}")
        print("-" * 60)