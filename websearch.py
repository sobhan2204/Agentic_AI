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
import asyncio

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

mcp = FastMCP("websearch")

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





if __name__ == "__main__":
    # print("Enter somthing you want to search on web:")
    # input_query = input()
    # ans = search_web(input_query)
    # print("The answer is : ",asyncio.run(ans))
     mcp.run(transport="stdio")