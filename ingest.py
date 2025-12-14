import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# 1. Load the secrets from your .env file
load_dotenv()

def ingest_data():
    print("--- Starting Ingestion ---")
    
    # 2. Load the PDF
    # Make sure you have a file named 'sample.pdf' in your folder
    pdf_path = "sample.pdf" 
    
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 3. Split the text into chunks
    # We split the text because the AI can't read a whole book at once.
    # We create chunks of 1000 characters with a little overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Document split into {len(docs)} chunks.")

    # 4. Connect to Azure OpenAI (for Embeddings)
    # This prepares the tool that converts text -> numbers
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )

    # 5. Connect to Azure AI Search (The Vector DB)
    print("Connecting to Azure AI Search...")
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query
    )
    
    # 6. Upload Data (This automatically creates the index if missing)
    print("Uploading vectors... (this might take a moment)")
    vector_store.add_documents(documents=docs)
    
    print("--- Success! Data is now in the Vector Database ---")

if __name__ == "__main__":
    ingest_data()