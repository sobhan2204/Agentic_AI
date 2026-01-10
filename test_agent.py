"""
Test script for the agentic AI agent.
This script verifies that all components are working correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

async def test_agent():
    """Test the agent with sample queries"""
    
    # Import the client module
    from client import main
    
    print("=" * 60)
    print("AGENTIC AI AGENT - TEST SCRIPT")
    print("=" * 60)
    print("\nThis will start your agent. Try these test queries:")
    print("\n1. Calculate: What is the derivative of x^2 + 3x?")
    print("2. Weather: What's the weather in New York?")
    print("3. Translate: Translate 'Hello world' to Spanish")
    print("4. Search: Search for latest AI news")
    print("5. Email: (requires Gmail setup)")
    print("\nType 'exit', 'quit', or 'q' to stop.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 60)
    print()
    
    # Run the main agent
    await main()

if __name__ == "__main__":
    load_dotenv()
    
    # Check for required API key
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in .env file!")
        print("Please create a .env file with your GROQ_API_KEY")
        exit(1)
    
    asyncio.run(test_agent())
