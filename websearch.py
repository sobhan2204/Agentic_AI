from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os

mcp = FastMCP("websearch-mcp-server")

from dotenv import load_dotenv
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")

tool = TavilySearch(max_results = 2, api_key=tavily_api_key)


@mcp.tool()
async def search_web(query : str) -> str:
    """
    summary_
    search the web for the query    
    Args:
      query : write any query in english
    """
    print("Web serach is being started")

    return tool.invoke(query)

mcp.run(transport="stdio")
#if __name__ == "__main__":
#    import asyncio
#    query = input("Enter query to search: ")
#    result  = asyncio.run(search_web(query))
#    print("AI : ",result)
