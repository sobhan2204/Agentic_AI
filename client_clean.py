from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import sentence_transformers
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from .env file
load_dotenv()

async def main():

    clients = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "Translate": {
                "command": "python",
                "args": ["translate.py"],
                "transport": "stdio",
            },
            "gmail": {
                "command": "python",
                "args": ["gmail.py"],
                "transport": "stdio",
            }
        }
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(" GROQ_API_KEY not found in .env")
    os.environ["GROQ_API_KEY"] = groq_api_key

    
    tools = await clients.get_tools()
    

    model = ChatGroq(model="llama3-70b-8192")
    agent = create_react_agent(model, tools)

    #math_response = await agent.ainvoke(
    #    {"messages": [{"role": "user", "content": "What is (3 + 5) * 2?"}]}
    #)
    
    #print("Math response:" , math_response['messages'][-1].content)
    

    #sentence = input("Enter the sentence you want to convert")
    #translate_response = await agent.ainvoke(
    #    {"messages": [{"role": "user", "content": f"translate {sentence}"}]}
    #)
    
    #print("Translate response:" , translate_response['messages'][-1].content)
    
    gmail_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "send a email to sobhan2204@gmial.com with subject 'hii' and body 'hello'"}]}
    )
    print("Mail sent")
    #print("Math response:" , math_response['messages'][-1].content)
    

if __name__ == "__main__":
    asyncio.run(main())
