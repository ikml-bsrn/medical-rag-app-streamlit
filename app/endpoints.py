from fastapi import APIRouter, HTTPException, Request
from rag_retrieve import invoke_rag_by_query
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

# define POST endpoint for querying the RAG system
# ...

# define GET endpoint for querying the RAG system
@router.get("/api/v1", response_model=QueryResponse)
async def query_rag_system(request: Request, query: str):
    # initialise RAG chain from app state
    rag_chain = request.app.state.rag_chain

    # ensure RAG chain is initialised
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system has not been initliased yet. Please try again later."
        )
    
    # invoke RAG chain asynchronously
    try:
        full_response = ""
        async for chunk in rag_chain.astream(query):
            full_response += chunk

        return QueryResponse(query=query, response=full_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))