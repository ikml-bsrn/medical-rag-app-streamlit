import os
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv(".env")

import asyncio # for async support
import streamlit as st

from rag_retrieve import initialise_embedding_model, load_vector_store, create_retriever, create_prompt_template, initialise_llm, create_rag_chain, invoke_rag_by_query

async def run_streamlit_app():
    st.set_page_config(
        page_title="Medical RAG Assistant",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical RAG Assistant")
    st.markdown("Ask medical questions and get answers based on the MedQuAD corpus.")

    with st.sidebar:
        st.header("Configuration")

        langchain_api_key = st.text_input(
            "LangChain API Key",
            type="password",
            help="Enter your LangChain API key to use the agent"
        )

        hf_api_key = st.text_input(
            "Hugging Face API Key",
            type="password",
            help="Enter your Hugging Face API key to access models"
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.input_items = []
            st.rerun()

        if st.checkbox("Enable LangSmith Tracing"):
            os.environ["LANGSMITH_ENDPOINT"] = "https://eu.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = "medical-qa-chatbot"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_items" not in st.session_state:
        st.session_state.input_items = []

    # initialise RAG chain in session state
    if "rag_chain" not in st.session_state and langchain_api_key and hf_api_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["HUGGING_FACE_API_KEY"] = hf_api_key

        embedding_model = initialise_embedding_model()  # initialise embedding model
        # load vector store
        vector_store = load_vector_store(
            embedding_model, 
            collection_name="medquad_colllection", 
            persist_directory="medquad_chroma_db_2"
            )  
        retriever = create_retriever(vector_store, k=3) # initialise retriever
        prompt = create_prompt_template() # initialise prompt template
        hf_chat_model = initialise_llm()  # initialise Hugging Face LLM

        st.session_state.rag_chain = create_rag_chain(retriever, prompt, hf_chat_model) # initialise RAG chain
    else:
        st.info("Please enter your LangChain and Hugging Face API keys to initialise the RAG chain.")

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if user_input := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
    
        with st.chat_message("assistant"):
            if "rag_chain" in st.session_state: # check if RAG chain is initialised
                response_placeholder = st.empty()
                full_response = ""

                try:
                    # initiate async response from the RAG chain
                    async for response_chunk in invoke_rag_by_query(
                        st.session_state.rag_chain,
                        query=user_input
                    ):
                        full_response += response_chunk
                        response_placeholder.markdown(full_response + " ")
                except Exception as e:
                    st.error(f"Error during RAG chain invocation: {e}")
                    full_response = "An error occurred while processing your request."

        st.session_state.messages.append({"role":"assistant", "content": full_response})

if __name__ == "__main__":
    asyncio.run(run_streamlit_app())