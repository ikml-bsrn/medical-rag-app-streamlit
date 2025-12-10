def ingest_data(file_path, hf_embedding_model_name="NeuML/pubmedbert-base-embeddings"):
    """
    This function ingests, embeds, and stores data in a vector store using Langchain from a given file path.
    """
    import pandas as pd
    from langchain_core.documents import Document
    from langchain.text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # read the CSV file into a pandas DataFrame
    dataset = pd.read_csv('MedQuAD_Dataset_RAG_Scenario.csv')

    # convert DF rows into LangChain Documents
    documents = []
    for index, row in dataset.iterrows():
        # combine the question, answer into a single text chunk
        content = f"Question: {row['question']} Answer: {row['answer']}"

        # create a LangChain Document (with source and focus_area as metadata)
        doc = Document(
            page_content=content,
            metadata={
                'source':row['source'],
                'area':row['focus_area']
                }
        )
        # append the created Documents to the 'documents' list
        documents.append(doc)

    # splits documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
        )
    chunks = text_splitter.split_documents(documents)

    # initialise HuggingFace embedding model for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=hf_embedding_model_name
        )

    # create and saves a Chroma vector store to disk
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name="medquad_collection",
        persist_directory="./medquad_chroma_db"
    )

    return vector_store

if __name__ == "__main__":
    ingest_data('MedQuAD_Dataset_RAG_Scenario.csv')