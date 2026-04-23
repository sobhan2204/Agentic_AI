import json
import requests
import os
import xml.etree.ElementTree as ET
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List

mcp = FastMCP("archive")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq_summarizer(system_prompt: str, user_content: str) -> str:
    """Internal LLM summarizer using Groq."""
    api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found. Cannot summarize."

    body = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        response = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=25,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception:
        return "Summary generation failed. Please try again."


# ====================== ARXIV SEARCH TOOL (NEW) ======================
@mcp.tool(name="arxiv_research_search")
def arxiv_research_search(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",  # "relevance" | "lastUpdatedDate" | "submittedDate"
) -> str:
    """
    Search arXiv for academic papers (ML, AI, physics, math, CS, etc.)
    and return an LLM-summarized research overview.

    Use this tool for any query about:
    - AI / ML papers (e.g. "Attention is all you need", "RLHF", "diffusion models")
    - Physics, mathematics, computer science research
    - Named papers or authors in academic contexts

    Args:
        query:       Natural language or paper title / keywords.
        max_results: How many papers to fetch (default 5, max 15).
        sort_by:     "relevance" (default), "lastUpdatedDate", or "submittedDate".
    """
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(int(max_results), 15),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()

        # Parse Atom XML returned by arXiv
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(response.text)
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"No arXiv papers found for: '{query}'"

        raw_results = []
        for entry in entries:
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
            published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]
            arxiv_id_raw = entry.findtext("atom:id", namespaces=ns) or ""
            arxiv_url = arxiv_id_raw.strip()

            authors = [
                a.findtext("atom:name", namespaces=ns) or ""
                for a in entry.findall("atom:author", namespaces=ns)
            ]
            author_str = ", ".join(authors[:4])
            if len(authors) > 4:
                author_str += " et al."

            raw_results.append({
                "title": title,
                "authors": author_str,
                "published": published,
                "summary": summary[:500],
                "url": arxiv_url,
            })

        # === LLM Summarization ===
        system_prompt = (
            "You are an expert academic researcher. "
            "Turn the following arXiv search results into a clear, well-structured research summary. "
            "For the top result (most relevant paper), give a detailed explanation of its key contributions, "
            "methodology, and impact. For the remaining results, use brief bullet points. "
            "Always include the paper title, authors, year, and arXiv link. "
            "Be factual, precise, and concise."
        )

        user_content = f"Query: {query}\n\nPapers found:\n"
        for i, r in enumerate(raw_results, 1):
            user_content += (
                f"{i}. Title: {r['title']}\n"
                f"   Authors: {r['authors']}\n"
                f"   Published: {r['published']}\n"
                f"   Link: {r['url']}\n"
                f"   Abstract: {r['summary']}\n\n"
            )

        summary = _call_groq_summarizer(system_prompt, user_content)
        return f"**arXiv Research Summary: {query}**\n\n{summary}"

    except Exception as e:
        return f"arXiv search error: {str(e)[:200]}"


# ====================== FIXED ARCHIVE SEARCH TOOL ======================
@mcp.tool(name="archive_research_search")
def archive_research_search(
    query: str,
    mediatype: str = "texts",
    year_from: str = "",
    year_to: str = "",
    limit: int = 6,
) -> str:
    """
    Search Internet Archive for digitized books, historical texts, web archives,
    audio/video recordings, and other non-academic content.

    NOTE: For academic research papers (AI, ML, physics, CS, math), prefer
    arxiv_research_search instead — Archive.org does not reliably index them.

    Args:
        query:      Search keywords.
        mediatype:  "texts" (default), "audio", "video", "image", "software",
                    "collection", or "all".
        year_from:  Optional start year filter (e.g. "1990").
        year_to:    Optional end year filter (e.g. "2000").
        limit:      Number of results to fetch (default 6, max 15).
    """
    base_url = "https://archive.org/advancedsearch.php"

    q = query.strip()
    # Only add mediatype filter if explicitly set and not "all"
    if mediatype and mediatype.lower() not in ("", "all"):
        q += f" mediatype:{mediatype}"
    if year_from:
        q += f" year:[{year_from} TO {year_to or '*'}]"

    params: Dict[str, Any] = {
        "q": q,
        "fl[]": ["title", "creator", "date", "description", "identifier", "mediatype", "year"],
        "rows": min(int(limit), 15),
        "output": "json",
        # FIX: Sort by downloads (relevance proxy) only — not year desc,
        # which was causing recent unrelated content to surface first.
        "sort[]": ["downloads desc"],
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        raw_results: List[Dict] = []
        for doc in data.get("response", {}).get("docs", []):
            identifier = doc.get("identifier", "")
            item_url = f"https://archive.org/details/{identifier}" if identifier else ""
            raw_results.append({
                "title": doc.get("title", "Untitled"),
                "creator": doc.get("creator") or "Unknown",
                "year": doc.get("year") or (doc.get("date", "") or "")[:4],
                "description": (doc.get("description") or "")[:400],
                "direct_url": item_url,
            })

        if not raw_results:
            return f"No results found for: '{query}'"

        system_prompt = (
            "You are an expert researcher. "
            "Turn the following Internet Archive search results into a clear, "
            "well-structured summary. Use bullet points for key findings. "
            "Always include title, author/year, and direct link. "
            "Be factual, neutral, and concise."
        )

        user_content = f"Query: {query}\n\nRaw Results:\n"
        for i, r in enumerate(raw_results, 1):
            user_content += (
                f"{i}. Title: {r['title']}\n"
                f"   Author/Year: {r['creator']} ({r['year']})\n"
                f"   Link: {r['direct_url']}\n"
                f"   Description: {r['description']}\n\n"
            )

        summary = _call_groq_summarizer(system_prompt, user_content)
        return f"**Archive.org Research Summary: {query}**\n\n{summary}"

    except Exception as e:
        return f"Archive search error: {str(e)[:200]}"


# ====================== WAYBACK SNAPSHOT TOOL (unchanged) ======================
@mcp.tool(name="wayback_snapshot")
def wayback_snapshot(url: str, year: str = "") -> str:
    """
    Get an archived Wayback Machine snapshot of a URL, with an LLM summary.

    Args:
        url:  The original URL to look up.
        year: Optional year to find a snapshot near (e.g. "2010").
    """
    api_url = "https://archive.org/wayback/available"
    params = {"url": url}
    if year and year.strip():
        params["timestamp"] = f"{year.strip()}0101000000"

    try:
        response = requests.get(api_url, params=params, timeout=15)
        data = response.json()

        snapshot = data.get("archived_snapshots", {}).get("closest")
        if snapshot and snapshot.get("available"):
            raw_info = (
                f"Original URL: {url}\n"
                f"Archived URL: {snapshot['url']}\n"
                f"Timestamp: {snapshot['timestamp']}"
            )
            summary = _call_groq_summarizer(
                "You are a research assistant. Summarize this Wayback Machine snapshot in 2-3 short sentences.",
                raw_info,
            )
            return summary
        else:
            return f"No snapshot found for {url}."
    except Exception as e:
        return f"Wayback error: {str(e)[:150]}"


if __name__ == "__main__":
    mcp.run(transport="stdio")