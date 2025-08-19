#!pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
#!pip install transformers sentence-transformers faiss-cpu langchain chromadb
#!pip install pip install huggingface_hub
#!pip install pypdf


import os 
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # This splits the text into chunks for the model to understand
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from mcp.server.fastmcp import FastMCP



mcp = FastMCP("Rag_model")

from dotenv import load_dotenv
load_dotenv()


LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") ## to retrive langchian API key
if not LANGCHAIN_API_KEY:
     raise ValueError(" LANGCHAIN_API_KEY not found in .env")
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

@mcp.tool()
def rag(document_adress: str) -> str:
    """
    Get any answer form any pdf you want

    Arg:
        document_adress: Upload the adress of document you want the model to answer from
    """
    loader = PyPDFLoader(document_adress)
    data = loader.load()
    print("Reading document")

    #### INDEXING ####

    # Load blog

    blog_docs = loader.load()  

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50) # chunk_overlap is the overlap between chunks 

    # Make splits
    splits = text_splitter.split_documents(blog_docs)
    print("Splitting document")
    # Index
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {"allow_download": "True"}
    #embedding_model = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, # vectorstore is used to store the embeddings of the chunks
                                        embedding=embedding_model)

    print("Indexing document")
    retriever = vectorstore.as_retriever() # Retriver is used to retrieve the chunks from the vectorstore

    print("Searching for answer")
    question = input("Enter the question you want to ask related to document : ")
    docs = vectorstore.similarity_search(question) # Searches for similar chunks in the vectorstore
    print(len(docs) , " " , docs[0])


    from langchain_community.llms import Ollama
    from langchain_core.callbacks import StreamingStdOutCallbackHandler

    llm = Ollama(
        model="mistral",
        #callbacks=[StreamingStdOutCallbackHandler()], # StreamingStdOutCallbackHandler is used to stream the output to the console
        verbose=True,
    )
    from langchain import hub
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt.messages

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    chain = (
        RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"context": docs, "question": question})

    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = qa_chain.invoke(question)
    return answer


if __name__ == "__main__":
   mcp.run(transport="stdio")


