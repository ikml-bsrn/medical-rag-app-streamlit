from fastapi import FastAPI
from endpoints import router
import uvicorn
import os
from dotenv import load_dotenv

from rag_retrieve import (
    initialise_embedding_model, 
    load_vector_store, 
    create_retriever, 
    create_prompt_template, 
    initialise_llm, 
    create_rag_chain
)

# load environment variables from .env file
load_dotenv()

# initialise the FastAPI app
app = FastAPI(
    title="Medical RAG Assistant API",
    description="API for answering medical questions based on the MedQuAD corpus.",
    version="1.0.0"
)

app.state.rag_chain = None  # initialise rag_chain as None

@app.on_event("startup")
async def startup_event():
    global rag_chain

    # Get API keys from environment variables
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    hf_api_key = os.getenv("HUGGING_FACE_API_KEY")

    if not langchain_api_key or not hf_api_key:
        raise ValueError("API keys for LangChain and Hugging Face must be set in environment variables.")
    
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["HUGGING_FACE_API_KEY"] = hf_api_key
    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "medical-qa-chatbot"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    try:
        embedding_model = initialise_embedding_model()  # initialise embedding model
        # load vector store
        vector_store = load_vector_store(
            embedding_model, 
            collection_name="medquad_collection", 
            persist_directory="medquad_chroma_db_2"
            ) 
        
        retriever = create_retriever(vector_store, k=3) # create retriever
        prompt = create_prompt_template() # initialise prompt template
        hf_chat_model = initialise_llm(repo_id="openai/gpt-oss-20b", provider="together")  # initialise Hugging Face LLM

        app.state.rag_chain = create_rag_chain(retriever, prompt, hf_chat_model) # initialise RAG chain
    
    except Exception as e:
        print(f"Error initialising RAG chain: {e}")
        app.state.rag_chain = None

# set up root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Medical RAG Assistant API. Visit /docs for API documentation, and /api/v1 for RAG endpoints.",
        "status": "online" if app.state.rag_chain else "initialising",
        "docs": "/docs"
    }

# setup health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if app.state.rag_chain else "unhealthy",
        "rag_chain_initialised": app.state.rag_chain is not None # returns True if rag_chain is initialised
    }

# include the router
app.include_router(
    router, tags = ["RAG"]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)