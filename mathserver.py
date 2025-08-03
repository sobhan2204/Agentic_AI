from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b:int) -> int:
    """_summary_
    Add two number together
    """
    return a+b

@mcp.tool()
def multiply(a: int, b:int) -> int:
    """_summary_
    Multiply two number together
    """
    return a*b

if __name__ == "__main__":
    mcp.run(transport="stdio")  # useful if we want to run the server in the terminal locally in this
    #we will get the inpu and output in the terminal itself
    