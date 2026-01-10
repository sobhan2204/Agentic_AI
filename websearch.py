import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Field name \"output_schema\" in \"TavilyResearch\" shadows",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Field name \"stream\" in \"TavilyResearch\" shadows",
    category=UserWarning,
)

import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional
import json

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

mcp = FastMCP("websearch-mcp-server")

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")

tool = TavilySearch(max_results=2, api_key=tavily_api_key)
client = TavilyClient(api_key=tavily_api_key)


@mcp.tool()
async def search_web(query: str) -> str:
    """Search the web for the query and return a summary string."""
    try:
        result = tool.invoke(query)
        
        # Handle different return types
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Extract useful information from dict
            if 'results' in result:
                # Format results as readable string
                formatted_results = []
                for idx, item in enumerate(result['results'], 1):
                    title = item.get('title', 'No title')
                    content = item.get('content', 'No content')
                    url = item.get('url', '')
                    formatted_results.append(
                        f"{idx}. {title}\n   {content}\n   Source: {url}\n"
                    )
                return "\n".join(formatted_results)
            elif 'answer' in result:
                return result['answer']
            else:
                # Fallback: convert dict to readable string
                return json.dumps(result, indent=2)
        elif isinstance(result, list):
            # If it's a list of results
            formatted_results = []
            for idx, item in enumerate(result, 1):
                if isinstance(item, dict):
                    title = item.get('title', 'No title')
                    content = item.get('content', item.get('snippet', 'No content'))
                    url = item.get('url', '')
                    formatted_results.append(
                        f"{idx}. {title}\n   {content}\n   Source: {url}\n"
                    )
                else:
                    formatted_results.append(f"{idx}. {str(item)}\n")
            return "\n".join(formatted_results) if formatted_results else "No results found."
        else:
            return str(result)
    except Exception as e:
        return f"Error searching web: {str(e)}"


def search_urls(query: str, max_results: int = 5):
    """Return a list of URLs for a query using Tavily client."""
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )
    return [r.get("url") for r in response.get("results", []) if r.get("url")]


@mcp.tool()
def find_relevant_urls(query: str) -> str:
    """Return URLs with titles/snippets for a query as a formatted string."""
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
        )
        
        results = []
        for idx, r in enumerate(response.get("results", []), 1):
            if r.get("url"):
                title = r.get("title", "No title")
                snippet = r.get("content", "No snippet")
                url = r.get("url")
                results.append(
                    f"{idx}. {title}\n"
                    f"   Summary: {snippet}\n"
                    f"   URL: {url}\n"
                )
        
        if results:
            return f"Search results for '{query}':\n\n" + "\n".join(results)
        else:
            return f"No results found for query: {query}"
    except Exception as e:
        return f"Error finding URLs: {str(e)}"


def _extract_pdf_links(page_url: str) -> set[str]:
    """Fetch a page and return absolute PDF links found."""
    resp = requests.get(page_url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            links.add(urljoin(page_url, href))
    return links


def _safe_filename(url: str) -> str:
    """Derive a safe filename from URL path, fallback to hash if empty."""
    name = Path(urlparse(url).path).name
    return name or "download.pdf"


@mcp.tool()
def scrape_pdfs(page_url: str, query: Optional[str] = None, data_dir: str = "data") -> str:
    """
    Download PDF links from a page into the data directory.

    - Creates data_dir if missing
    - Skips files that already exist
    - If query is provided, only PDFs whose URL/filename contains query keywords are downloaded
    
    Returns a formatted string summary of the operation.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        pdf_links = _extract_pdf_links(page_url)
    except Exception as e:
        return f"Error accessing page {page_url}: {str(e)}"

    # Optional keyword filter
    if query:
        keywords = [w for w in query.lower().split() if len(w) > 2]
        def keep(link: str) -> bool:
            text = link.lower()
            return any(k in text for k in keywords)
        pdf_links = {l for l in pdf_links if keep(l)}

    saved, skipped, errors = [], [], []

    for link in sorted(pdf_links):
        fname = _safe_filename(link)
        target = data_path / fname
        if target.exists():
            skipped.append(str(target))
            continue
        try:
            with requests.get(link, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            saved.append(str(target))
        except Exception as e:
            errors.append({"url": link, "error": str(e)})

    # Format output as string
    output = [f"PDF Scraping Results for: {page_url}"]
    output.append(f"PDFs found: {len(pdf_links)}")
    
    if saved:
        output.append(f"\nSaved ({len(saved)}):")
        for path in saved:
            output.append(f"  ✓ {path}")
    
    if skipped:
        output.append(f"\nSkipped (already exist) ({len(skipped)}):")
        for path in skipped[:5]:  # Show first 5
            output.append(f"  - {path}")
        if len(skipped) > 5:
            output.append(f"  ... and {len(skipped) - 5} more")
    
    if errors:
        output.append(f"\nErrors ({len(errors)}):")
        for err in errors[:3]:  # Show first 3 errors
            output.append(f"  ✗ {err['url']}: {err['error']}")
    
    return "\n".join(output)


@mcp.tool()
def search_and_download_pdfs(query: str, max_results: int = 5, data_dir: str = "data") -> str:
    """
    Pipeline: search for URLs matching the query, then download only PDFs relevant to the query from those pages.
    - Uses Tavily search to get URLs
    - Filters PDFs by query keywords (via scrape_pdfs)
    - Creates the data directory if needed; skips existing files
    
    Returns a formatted string summary.
    """
    try:
        urls = search_urls(query, max_results=max_results)
        
        if not urls:
            return f"No URLs found for query: {query}"
        
        output = [f"Searching and downloading PDFs for: {query}"]
        output.append(f"Found {len(urls)} pages to scan\n")
        
        total_found = 0
        total_saved = 0
        
        for idx, url in enumerate(urls, 1):
            output.append(f"[{idx}/{len(urls)}] Scanning: {url}")
            result = scrape_pdfs(url, query=query, data_dir=data_dir)
            output.append(result)
            output.append("")  # Blank line between results
            
            # Try to extract counts from result string
            if "PDFs found:" in result:
                try:
                    found = int(result.split("PDFs found:")[1].split()[0])
                    total_found += found
                except:
                    pass
            if "Saved (" in result:
                try:
                    saved = int(result.split("Saved (")[1].split(")")[0])
                    total_saved += saved
                except:
                    pass
        
        output.append("=" * 50)
        output.append(f"SUMMARY:")
        output.append(f"  Total PDFs found: {total_found}")
        output.append(f"  Total PDFs saved: {total_saved}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error in search_and_download_pdfs: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")