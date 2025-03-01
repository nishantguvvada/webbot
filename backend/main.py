from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import bs4
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Annotated
from typing import Literal
from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv

load_dotenv()

# 3 Components from LangChain's suite of integrations 1. Chat Model 2. Embeddings Model 3. Vector Store

# 1. Chat Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.getenv('GEMINI_API_KEY'))

# 2. Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('GEMINI_API_KEY'))

# DocumentLoader (objects) that load in data from sources and return a list of Document objects
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

assert len(docs) == 1
# print(f"Total characters: {len(docs[0].page_content)}")

# print(docs[0].page_content[:500])

# Split documents using RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter recursively splits the document based on a list of separators ["\n\n", "\n", " ", ""]
# until each chunk is appropriate in size. This splitting strategy has the effect of trying to keep the paragraph then sentences and then words together.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# print(f"Split blog post into {len(all_splits)} sub-documents.")

# 3. Vector Store

# vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
# vector_store.save_local("faiss_index")
# print("Index created.")

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Adding section to the metadata
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


# all_splits[0].metadata

_ = db.add_documents(all_splits)

# Retrieval & Generation

# Using LangGraph, we need 3 things:
# 1. State of our application = data input to the application, transferred between steps and output by the application
# State is typically TypedDict, can be a Pydantic BaseModel
# 2. Nodes of our application (steps)
# 3. Control Flow (order of steps)

# Ignore Search class until you've read till line 139

class Search(BaseModel): # Schema for the retrieval query, it contains a string query and a section
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Defining the State of the application

class State(TypedDict):
    question: str
    query: Search
    context: List[Document] # Document object returned by the retriever
    answer: str

# Defining the nodes (steps of the application)

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = db.similarity_search(
        query.query,
        filter={"section": query.section},
    )
    # retrieved_docs = db.similarity_search(state["question"]) # performing a similarity search
    return { "context": retrieved_docs }

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"]) # joining all the retrieved documents
    messages = f"""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {state["question"]}
        Context: {docs_content}
        Answer:
    """
    response = llm.invoke(messages)
    return { "answer": response.content }

# Defining control flow

# StateGraph: A graph whose nodes communicate by reading and writing to a shared state.
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate]) # add_sequence: add nodes that will be executed in the provided order.
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# # Invoke
# result = graph.invoke({"question": "What is Task Decomposition?"})

# print(f"Context: {len(result["context"])}\n\n")
# print(f"Answer: {result["answer"]}")

# # Stream steps
# for step in graph.stream(
#     {"question": "What is Task Decomposition?"}, stream_mode="updates"
# ):
#     print(f"{step}\n\n----------------\n")

# # Stream tokens
# for message, metadata in graph.stream(
#     {"question": "What is Task Decomposition?"}, stream_mode="messages"
# ):
#     print(message.content, end="|")


# Analysing user query, transforming it to improve relevancy and structure and retrieving using the transformed query

for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")