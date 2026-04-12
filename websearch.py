import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

import warnings
warnings.filterwarnings("ignore", message=r"Field name \"output_schema\" in \"TavilyResearch\" shadows", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Field name \"stream\" in \"TavilyResearch\" shadows", category=UserWarning)

import os
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import concurrent.futures

mcp = FastMCP("websearch")
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")

client = TavilyClient(api_key=tavily_api_key)


def with_timeout(func, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        return future.result(timeout=timeout)


def truncate(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def format_results(results: list, answer: str = "") -> str:
    lines = []

    # If Tavily returned a direct answer, show it first
    if answer:
        lines.append(f"Summary: {answer}\n")

    if not results:
        return lines[0] if lines else "No results found."

    for idx, item in enumerate(results, 1):
        title   = item.get("title", "No title")
        content = truncate(item.get("content") or item.get("raw_content") or item.get("snippet", ""), max_chars=300)
        url     = item.get("url", "")
        score   = item.get("score", 0)
        lines.append(f"{idx}. {title}\n   {content}\n   Source: {url} (relevance: {score:.2f})")

    return "\n\n".join(lines)


@mcp.tool(name="web_search")
def search_web(query: str) -> str:
    """
    Search the web for a query and return a detailed summary.
    Use this for general questions, news, facts, and current information.
    """
    try:
        response = with_timeout(
            lambda: client.search(
                query,
                max_results=3,           # sweet spot: enough coverage, low tokens
                search_depth="advanced", # richer content than basic
                include_answer=True,     # Tavily summary = main answer, sources = backup
            ),
            timeout=30,
        )

        results = response.get("results", [])
        answer  = response.get("answer", "")

        if not results and not answer:
            return "No results found."

        return format_results(results, answer)

    except concurrent.futures.TimeoutError:
        # Fallback: try a faster basic search before giving up
        try:
            response = with_timeout(
                lambda: client.search(query, max_results=3, search_depth="basic"),
                timeout=15,
            )
            results = response.get("results", [])
            return format_results(results) if results else "Search timed out."
        except Exception:
            return "Search timed out. Please try again."

    except Exception as e:
        return f"Search failed: {str(e)[:200]}"


if __name__ == "__main__":
    #print(search_web("What are the latest news on AI?"))
    mcp.run(transport="stdio")