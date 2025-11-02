from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import traceback


class MCPChatServer:
    """MCP Chat Server with tool routing and FAISS memory."""
    
    def __init__(self):
        load_dotenv()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index_path = "faiss_index"
        self.faiss_index = None
        self.model = None
        self.tools = None
        self.clients = None
        
        # Keywords for tool selection
        self.websearch_keywords = {
            "today", "yesterday", "now", "latest", "current", "news",
            "weather", "who won", "price", "open in", "near me"
        }
        self.math_keywords = {
            "calculate", "solve", "math", "equation", "derivative", "integral",
            "simplify", "evaluate", "derive", "integrate", "limit", "series",
            "trigonometry", "algebra", "geometry", "calculus", "statistics",
            "probability", "matrix", "vector", "function", "limit", "series",
            "trigonometry", "algebra", "geometry", "calculus", "statistics",
        }
        self.gmail_keywords = {
            "send" , "mail" , "gmail" , "email" , "message" , "compose" , "send email" , "send mail" , "send message" , 
            "send gmail to " , "send email" , "send mail" , "send message" , "send gmail", "subject" , "read email" 
        }
        self.math_keywords = {
            "calculate", "solve", "math", "equation", "derivative", "integral"
        }
        self.music_keywords = {
            "play", "music", "song", "album", "artist", "playlist", "radio", "station",
             "shuffle", "repeat", "volume", "lyrics", "lyric"
        }
    
    def pick_tool(self, query: str) -> str:
        """Simple rule-based tool selection."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in self.websearch_keywords):
            return "websearch"

        if any(word in query_lower for word in self.gmail_keywords):
            return "gmail"

        if any(word in query_lower for word in self.music_keywords):
            return "music-player"
        
        if any(word in query_lower for word in self.math_keywords):
            return "math"
        
        return "llm"
    
    async def initialize(self):
        """Initialize all components."""
        print("Initializing chat server...")
        
        # Validate API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        # Initialize MCP clients
        self.clients = MultiServerMCPClient({
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
        })
        
        # Get tools and initialize model
        self.tools = await self.clients.get_tools()
        self.model = ChatGroq(model="llama3-70b-8192")
        
        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.faiss_index = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✓ Loaded existing FAISS index")
        else:
            self.faiss_index = FAISS.from_texts([""], self.embeddings)
            print("✓ Created new FAISS index")
        
        print("✓ Server initialized successfully\n")
    
    def get_tool_by_name(self, name: str):
        """Get tool by name, returns None if not found."""
        return next((t for t in self.tools if t.name == name), None)
    
    async def handle_websearch(self, user_input: str) -> str:
        """Handle web search queries."""
        search_tool = self.get_tool_by_name("websearch")
        if not search_tool:
            return "Web search tool not available."
        
        search_results = await search_tool.invoke({"query": user_input})
        
        # Extract results safely
        results = search_results.get("results", []) if isinstance(search_results, dict) else []
        context = "\n".join([
            f"- {r.get('snippet', '')} ({r.get('url', '')})" 
            for r in results if r.get('snippet')
        ])
        
        if not context:
            context = "No results found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Always base your answer on the given search results. "
                      "If the answer is not in the results, say: 'I couldn't find reliable information.'"),
            ("user", "{user_input}\n\nWeb Search Results:\n{context}")
        ])
        
        formatted = prompt.format_messages(user_input=user_input, context=context)
        response = await self.model.ainvoke(formatted)
        return response.content
    
    async def handle_math(self, user_input: str) -> str:
        """Handle math queries."""
        math_tool = self.get_tool_by_name("math")
        if not math_tool:
            return "Math tool not available."
        
        result = await math_tool.invoke({"query": user_input})
        return str(result) if not isinstance(result, str) else result
    
    async def handle_llm(self, user_input: str) -> str:
        """Handle LLM queries with FAISS memory."""
        # Get relevant context from FAISS
        results = self.faiss_index.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in results if doc.page_content])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the conversation context to provide relevant answers."),
            ("user", "{user_input}\n\nConversation Context:\n{context}")
        ])
        
        formatted = prompt.format_messages(user_input=user_input, context=context)
        response = await self.model.ainvoke(formatted)
        return response.content
    
    async def process_query(self, user_input: str) -> str:
        """Process user query and return response."""
        # Add user input to FAISS
        self.faiss_index.add_texts(
            [user_input], 
            metadatas=[{"source": "user", "timestamp": str(datetime.now())}]
        )
        
        # Route to appropriate handler
        chosen_tool = self.pick_tool(user_input)
        print(f"[Router] Selected: {chosen_tool}")
        
        if chosen_tool == "websearch":
            response = await self.handle_websearch(user_input)
        elif chosen_tool == "math":
            response = await self.handle_math(user_input)
        else:
            response = await self.handle_llm(user_input)
        
        # Add response to FAISS
        self.faiss_index.add_texts(
            [response], 
            metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}]
        )
        
        # Save index after each interaction
        self.faiss_index.save_local(self.index_path)
        
        return response
    
    async def run(self):
        """Run the chat loop."""
        try:
            await self.initialize()
            
            print("Chat started. Type 'exit' to quit, 'clear' to reset history.\n")
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ["exit", "quit", "q"]:
                        print("\n✓ Conversation saved. Goodbye!")
                        break
                    
                    if user_input.lower() == "clear":
                        self.faiss_index = FAISS.from_texts([""], self.embeddings)
                        self.faiss_index.save_local(self.index_path)
                        print("✓ Conversation history cleared.")
                        continue
                    
                    print("\nAssistant: ", end="", flush=True)
                    response = await self.process_query(user_input)
                    print(response)
                    
                except KeyboardInterrupt:
                    print("\n\n✓ Interrupted. Saving and exiting...")
                    break
                except Exception as e:
                    print(f"\n✗ Error processing query: {e}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"\n✗ Fatal error: {e}")
            traceback.print_exc()
        finally:
            print("\nShutting down...")


async def main():
    """Entry point."""
    server = MCPChatServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())