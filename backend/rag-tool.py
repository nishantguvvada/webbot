# Adding logic for incorporating historical messages
# Involves management of a chat history

# 2 approaches: Chains (execute at most 1 retrieval step) and Agents (allow LLM's discretion to execute multiple retrieval steps)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import bs4
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
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

# Previously, the state contained user input (question), retrieved context and generated answer as separate keys
# Now, we represent the state using a sequence of messages (messages from user, assistant and retriever)
# User Input -> HumanMessage
# Vector Store Query -> AIMessage
# Retrieved Document -> ToolMessage
# Final Response -> AIMessage

# State is not required to be defined separately, we will use MessagesState
# class State(TypedDict):
#     question: str
#     query: Search
#     context: List[Document] # Document object returned by the retriever
#     answer: str

# Defining the nodes (steps of the application)

# Our graph consists of 3 nodes:
# 1. A node for user input that generates a query for retrieval or responds back to user directly
# 2. A node for the retrieval tool
# 3. A node that generates the final response

# Not required: tool calling allows model to rewrite user queries, there's no need for a separate step to rewrite queries
# tool calling also allows for direct response to queries that do not require the retrieval step such as "Hi, how are you?"
# def analyze_query(state: State):
#     structured_llm = llm.with_structured_output(Search)
#     query = structured_llm.invoke(state["question"])
#     return {"query": query}

# Converting the retrieve step to a tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = db.similarity_search(
        query,
        k=2,
    )
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# We will use ToolNode that executes the tool and adds result to the state as a ToolMessage

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Defining control flow

# StateGraph: A graph whose nodes communicate by reading and writing to a shared state.


graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

# # Invoke
# input_message = "Hello"
input_message = "What is Task Decomposition?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config
):
    step["messages"][-1].pretty_print()