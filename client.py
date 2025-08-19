from pyexpat import model
from langchain_core.callbacks import LLMManagerMixin
from langchain_core.language_models import llms
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
from mcp_use import MCPAgent, MCPClient
import os
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime


memory = MemorySaver()
history = []


async def main():
    """Run a chat using MCPAgent's built-in conversation memory."""

    load_dotenv()  # Load environment variables for API key

    print("Initializing chat...")

    clients = MultiServerMCPClient(
        {
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
            #"Rag": {
            #    "command": "python",
            #    "args": ["Rag_model.py"],
            #    "transport": "stdio",
            #}
        }
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    os.environ["GROQ_API_KEY"] = groq_api_key

    # CREATING LLM MODEL
    tools = await clients.get_tools()
    model = ChatGroq(model="llama3-70b-8192")
    
    agent = create_react_agent(model, tools )
    
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_path = "faiss_index"
        if os.path.exists(index_path):
           faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
           print("Loaded existing FAISS index.")
        else:
         faiss_index = FAISS.from_texts(["initial text"], embeddings)
         print("Created new FAISS index.")

        chat_template = ChatPromptTemplate.from_messages([
           ("system" , "you are a news reporter . So answer {user_input} with detailed information and be proffesional and don't assume anything and don't answer beyond the question's need . be presize and compact."),
            ("user" , "{user_input}")
            ])
        while True:
            user_input = input("\nYou: ")
            faiss_index.add_texts([user_input], metadatas=[{"source": "user", "timestamp": str(datetime.now())}])
            if user_input.lower() in ["exit", "quit","q"]:
                faiss_index.save_local(index_path)
                print("Fasiss index saved")
                print("Ending conversation...")
                break
            if user_input.lower() == "clear":
                # Conversation history is cleared by craeting a new faiss vector dataabase
                faiss_index = FAISS.from_texts(["initial text"], embeddings)
                faiss_index.save_local(index_path)
                print("Conversation history cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)
            #Format and invoke directly
            results = faiss_index.similarity_search(user_input, k=3)
            RELEVANCE_THRESHOLD = 0.78  # tune this experimentally
            similar_docs = [doc for doc, score in results if score > RELEVANCE_THRESHOLD]
            context = "\n".join([doc.page_content for doc in similar_docs])
            formatted_prompt = chat_template.format_messages(user_input=user_input + "\nContext from past" + context)
            Ai_response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            response = await model.ainvoke(formatted_prompt)
            faiss_index.add_texts([response.content], metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}])
            faiss_index.save_local(index_path)
            print(response.content)

            
            #print("Ai_response:", Ai_response['messages'][-1].content)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up clients if needed
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())