#!/usr/bin/env python3
"""
Test script to check which MCP servers are working
Run this before running your main client
"""

import asyncio
import sys
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_individual_server(name, config):
    """Test a single MCP server"""
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        # Create client with just this server
        test_client = MultiServerMCPClient({name: config})
        tools = await test_client.get_tools()
        print(f"✅ OK ({len(tools)} tools)")
        
        # Print tool names for debugging
        tool_names = [tool.name for tool in tools]
        if tool_names:
            print(f"   Tools: {', '.join(tool_names)}")
        
        return True, len(tools), tool_names
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False, 0, []

async def main():
    """Test all MCP servers"""
    print("MCP Server Connectivity Test")
    print("=" * 40)
    
    servers = {
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
        "Translate": {
            "command": "python",
            "args": ["translate.py"],
            "transport": "stdio",
        },
        "websearch": {
            "command": "python",
            "args": ["websearch.py"],
            "transport": "stdio",
        },
        "gmail": {
            "command": "python",
            "args": ["gmail.py"],
            "transport": "stdio",
        },
        "Rag": {
            "command": "python",
            "args": ["Rag.py"],
            "transport": "stdio",
        }
    }
    
    working_servers = {}
    failed_servers = {}
    total_tools = 0
    
    for name, config in servers.items():
        # Check if file exists for stdio servers
        if config.get("transport") == "stdio":
            script_file = config["args"][0]
            if not os.path.exists(script_file):
                print(f"Testing {name}... ❌ FAILED: File {script_file} not found")
                failed_servers[name] = f"File {script_file} not found"
                continue
        
        success, tool_count, tool_names = await test_individual_server(name, config)
        if success:
            working_servers[name] = {
                "config": config,
                "tool_count": tool_count,
                "tools": tool_names
            }
            total_tools += tool_count
        else:
            failed_servers[name] = "Connection failed"
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if working_servers:
        print(f"✅ Working servers ({len(working_servers)}):")
        for name, info in working_servers.items():
            print(f"   {name}: {info['tool_count']} tools")
        print(f"\nTotal tools available: {total_tools}")
    else:
        print("❌ No servers are working!")
    
    if failed_servers:
        print(f"\n❌ Failed servers ({len(failed_servers)}):")
        for name, reason in failed_servers.items():
            print(f"   {name}: {reason}")
    
    print("\n" + "=" * 40)
    
    if working_servers:
        print("✅ You can run client.py now")
        return True
    else:
        print("❌ Fix the server issues before running client.py")
        print("\nTroubleshooting tips:")
        print("1. Make sure all .py files exist in the current directory")
        print("2. Check that your .env file has GROQ_API_KEY")
        print("3. For gmail.py, ensure client_secret.json exists")
        print("4. For weather server, make sure localhost:8000 is running")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)