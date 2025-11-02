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
import traceback



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
            "maths-mcp-server": {
                "command": "python",
                "args": ["maths_server2.py"],
                "transport": "stdio",
            },
            "music-player": {
                "command": "python",
                "args": ["music_player.py"],
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
        # for faiss
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_path = "faiss_index"
        if os.path.exists(index_path):
           faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
           print("Loaded existing FAISS index.")
        else:
            # if os path of fasiss is not present then create a new one
         faiss_index = FAISS.from_texts(["initial text"], embeddings)
         print("Created new FAISS index.")


        chat_template = ChatPromptTemplate.from_messages([
           ("system" , "You are a helpful assistant. Always base your answer on the given search results. If the answer is not in the results, say: 'I couldn't find reliable info.'"),
            ("user" , "{user_input}")
            ])
        while True:
            user_input = input("\nYou: ")
            #search_tool = next(t for t in tools if t.name == "websearch")
            
            #search_results = await search_tool.invoke({"query": user_input})
            #context_from_web = "\n".join(
            #[f"- {r.get('snippet','')} ({r.get('url','')})" for r in search_results.get("results", [])])   

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
            #RELEVANCE_THRESHOLD = 0.78  # tune this experimentally
            similar_docs = results
            context_from_faiss = "\n".join([doc.page_content for doc in similar_docs])
            #Web Search Results: {context_from_web}
            context = f"""
            Conversation Context: {context_from_faiss}
            """
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
        traceback.print_exc()
    finally:
        # Clean up clients if needed
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())