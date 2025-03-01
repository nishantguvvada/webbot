from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# 3 Components from LangChain's suite of integrations 1. Chat Model 2. Embeddings Model 3. Vector Store

# 1. Chat Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.getenv('GEMINI_API_KEY'))

# 2. Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('GEMINI_API_KEY'))

# DocumentLoader (objects) that load in data from sources and return a list of Document objects

def load_source(url: str):
    loader = WebBaseLoader(
        web_paths=(f"{url}",),
        requests_kwargs={"verify":False},
    )

    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    all_splits = text_splitter.split_documents(docs)

    print(f"Split source into {len(all_splits)} sub-documents.")

    return all_splits

# 3. Vector Store

client = MongoClient(os.getenv('MONGODB_URL'))
DB_NAME = "web_db"
COLLECTION_NAME = "vectorstore"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index-vectorstore"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

def create_vector_store(url: str, collection_name: str):

    all_splits = load_source(url=url)

    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection=client[DB_NAME][f"{collection_name}-{COLLECTION_NAME}"],
        index_name=f"{collection_name}-{ATLAS_VECTOR_SEARCH_INDEX_NAME}"
    )

    print("Vector Store created.")

    index = vector_search.create_vector_search_index(dimensions=768)

    print("Vector Index created.")

def initialize_vector_store(collection_name: str):

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        os.getenv('MONGODB_URL'),
        f"{DB_NAME}.{collection_name}-{COLLECTION_NAME}",
        embeddings,
        index_name=f"{collection_name}-{ATLAS_VECTOR_SEARCH_INDEX_NAME}"
    )

    print("Vector Search initialized!")

    return vector_search


# Retrieval & Generation

def create_agent(collection_name: str):

    vector_search = initialize_vector_store(collection_name=collection_name)

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_search.similarity_search(
            query=query,
            k=2,
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    agent_executor = create_react_agent(llm, [retrieve])

    return agent_executor


# Invoke

def invoke(user_message: str, collection_name):
    input_message = (
        user_message
    )

    agent = create_agent(collection_name=collection_name)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values"
    )

    return response["messages"][-1].content