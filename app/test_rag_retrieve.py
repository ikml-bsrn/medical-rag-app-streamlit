import pytest

# import functions from rag_retrieve.py
from rag_retrieve import (
    initialise_embedding_model, 
    load_vector_store,
    create_retriever, 
    create_prompt_template, 
    initialise_llm, 
    create_rag_chain, 
    invoke_rag_by_query)

# ------------------ fixtures ------------------
@pytest.fixture
def embedding_model():
    model = initialise_embedding_model(model_name="NeuML/pubmedbert-base-embeddings")
    assert model is not None, "Failed to initialise embedding model"
    return model

@pytest.fixture
def vector_store(embedding_model):
    vec_store = load_vector_store(embedding_model, collection_name="medquad_collection", persist_directory="medquad_chroma_db_2")
    assert vec_store is not None, "Failed to load vector store"
    return vec_store

@pytest.fixture
def retriever(vector_store):
    r = create_retriever(vector_store, k=3)
    assert r is not None, "Failed to create retriever"
    return r

@pytest.fixture
def llm():
    llm_instance = initialise_llm(repo_id="openai/gpt-oss-20b", provider="together")
    assert llm_instance is not None, "Failed to initialise LLM"
    return llm_instance

@pytest.fixture
def prompt():
    prompt_template = create_prompt_template()
    assert prompt_template is not None, "Failed to create prompt template"
    return prompt_template

@pytest.fixture
def rag_chain(retriever, prompt, llm):
    rag = create_rag_chain(retriever, prompt, llm)
    assert rag is not None, "Failed to create RAG chain"
    return rag

# ------------------ tests ------------------

def test_initialise_embedding_model(embedding_model):
    from langchain_huggingface import HuggingFaceEmbeddings

    model = embedding_model

    assert model is not None
    assert isinstance(model, HuggingFaceEmbeddings)
    assert model.model_name == "NeuML/pubmedbert-base-embeddings"

def test_load_vector_store(vector_store):
    from langchain_core.vectorstores import VectorStore
    vs = vector_store

    assert vs is not None
    assert isinstance(vs, VectorStore)
    assert vs._collection_name == "medquad_collection"

def test_create_retriever(vector_store):
    from  langchain_core.vectorstores.base import VectorStoreRetriever

    retriever = create_retriever(vector_store, k=3)

    assert retriever is not None
    assert isinstance(retriever, VectorStoreRetriever)
    assert retriever.search_kwargs["k"] == 3

def test_create_prompt_template():
    from langchain_core.prompts.chat import ChatPromptTemplate

    prompt_template = create_prompt_template()

    assert prompt_template is not None
    assert isinstance(prompt_template, ChatPromptTemplate)

def test_initialise_llm(llm):
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    hf_llm = llm

    assert hf_llm is not None
    assert isinstance(hf_llm, ChatHuggingFace)
    assert isinstance(hf_llm.llm, HuggingFaceEndpoint)
    assert hf_llm.llm.repo_id == "openai/gpt-oss-20b"

def test_create_rag_chain(retriever, prompt, llm):
    from langchain_core.runnables.base import Runnable

    rag_chain = create_rag_chain(retriever, prompt, llm)

    assert rag_chain is not None
    assert isinstance(rag_chain, Runnable)

@pytest.mark.asyncio
async def test_invoke_rag_by_query(rag_chain, query="What are the symptoms of diabetes?"):
    full_response = ""

    async for chunk in invoke_rag_by_query(rag_chain, query):
        full_response += chunk

    assert full_response is not None
    assert isinstance(full_response, str)
    assert len(full_response) > 0, "Response must not be empty"


