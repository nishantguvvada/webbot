from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentic_rag import create_vector_store, invoke, list_collection
from pydantic import BaseModel
import uvicorn

app = FastAPI()

origins = [
    "https://webbot-flax.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
    return { "response": "working" }

class VectorInput(BaseModel):
    web_url: str
    index_name: str

@app.post("/vector")
def create_vector_db(vector_input: VectorInput):
    try:
        create_vector_store(vector_input.web_url, vector_input.index_name)
        return { "response": "vector store created!" }
    except Exception:    
        return { "error": "vector store not created."}
    
class UserInput(BaseModel):
    user_input: str
    collection_name: str

@app.post("/response")
def llm_response(user_query: UserInput):
    response = invoke(user_query.user_input, user_query.collection_name)
    return { "response": response }

@app.get("/collection")
def get_collection():
    response = list_collection()
    return { "response": response }


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)

