
# Pincone + LangGraph + FastAPI RAG API 
#  Hybrid search with BM25 + BAAI/bge-small-en-v1.5 embeddings
#  Reranking with BAAI/bge-reranker-base

import os
import uuid
from dotenv import load_dotenv  

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel
from typing import TypedDict

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv() # Load .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="HR Policy RAG API",
    version="1.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# ---------------------------------------------------
# Initialize Pinecone
# ---------------------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# ---------------------------------------------------
# HuggingFace Embedding Model
# ---------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# --------------------------------------------------
# Cross Encoder Reranker
# --------------------------------------------------

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# ---------------------------------------------------
# Load Trained BM25 Model
# ---------------------------------------------------

bm25 = BM25Encoder().load("bm25_values.json")


# --------------------------------------------------
# Hybrid Search
# --------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.7
):

    dense = embedding_model.embed_query(query)

    sparse = bm25.encode_queries(query)

    dense, sparse = hybrid_convex_scale(
        dense,
        sparse,
        alpha=alpha
    )

    response = index.query(
        vector=dense,
        sparse_vector=sparse,
        top_k=top_k,
        include_metadata=True
    )

    return response

# --------------------------------------------------
# rerank
# --------------------------------------------------
def rerank_documents(
    query: str,
    matches: list,
    top_k: int = 3
):
    """
    Rerank Pinecone matches using
    BAAI/bge-reranker-base
    """
    if not matches:
        return []
    # Create (query, document) pairs
    pairs = [
        (
            query,
            match["metadata"]["text"]
        )
        for match in matches
    ]
    # Predict relevance scores
    scores = reranker.predict(  # create an instance of the CrossEncoder class and use it to predict relevance scores for the query-document pairs
        pairs,                  # The `pairs` variable contains the query-document pairs to be scored.
        batch_size=16           # The `batch_size` parameter specifies the number of pairs to process in each batch during prediction. A larger batch size can speed up the prediction process but may require more memory.
    )
    ranked = []
    for match, score in zip(matches, scores):   # combine the original matches with their corresponding relevance scores
        ranked.append({                         # create a new dictionary for each match that includes the match ID, rerank score, Pinecone score, text, source, and page number
            "id": match["id"],
            "rerank_score": float(score),
            "pinecone_score": match["score"],
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source"),
            "page": match["metadata"].get("page")
        })
    ranked.sort(   # sort the ranked list of matches based on the rerank score in descending order
        key=lambda x: x["rerank_score"],
        reverse=True
    )
    return ranked[:top_k]


# --------------------------------------------------
# Graph State
# --------------------------------------------------
class GraphState(TypedDict):
    question: str
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    context: str
    answer: str

# --------------------------------------------------
# Retrieve Node
# --------------------------------------------------

def retrieve_node(state: GraphState):
    results = hybrid_search(
        state["question"],
        top_k=5     # retrieve more documents
        )
    return {
        "retrieved_docs": results["matches"]
    }

# --------------------------------------------------
# Rerank Node
# --------------------------------------------------
def rerank_node(state):
    reranked = rerank_documents(
        query=state["question"],
        matches=state["retrieved_docs"],
        top_k=3
    )

    context = "\n\n".join(
        doc["text"]
        for doc in reranked
    )

    return {
        "reranked_docs": reranked,
        "context": context
    }


# --------------------------------------------------
# Generate Node
# --------------------------------------------------

def generate_node(state: GraphState):

    prompt = f"""
You are an HR Policy Assistant.

Answer only from the provided context.

Context:

{state["context"]}

Question:

{state["question"]}
"""

    response = llm.invoke(prompt)

    return {

        "answer": response.content

    }

# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(GraphState)

builder.add_node(
    "retrieve",
    retrieve_node
)

builder.add_node(
    "rerank",
    rerank_node
)

builder.add_node(
    "generate",
    generate_node
)

builder.add_edge(
    START,
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "rerank"
)

builder.add_edge(
    "rerank",
    "generate"
)

builder.add_edge(
    "generate",
    END
)

graph = builder.compile()


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str

# --------------------------------------------------
# Response Model
# --------------------------------------------------

class QuestionResponse(BaseModel):
    question: str
    answer: str

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "HR Policy RAG API is running."
    }

# --------------------------------------------------
# Ask Endpoint
# --------------------------------------------------

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(request: QuestionRequest):

    try:

        result = graph.invoke(
            {
                "question": request.question
            }
        )

        return QuestionResponse(
            question=request.question,
            answer=result["answer"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )