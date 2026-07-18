import os
import shutil

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from the .env file
load_dotenv()


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    model_path = os.path.join(
        os.path.expanduser("~"),
        "Models",
        "Qwen3-Embedding-0.6B",
    )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Embedding model not found: {model_path}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        # Add more loaders if required, i.e. JSONLoader, TxtLoader, etc.
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types

        # Load and split document into chunks
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # Add chunks to Chroma vector store
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = "./data"

    # Create the Chroma database
    chroma = create_chroma_db(
        folder_path=folder_path,
        db_name="chroma_db_qwen3_test",
        delete_chroma_db=True,
    )
    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\nResult {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
