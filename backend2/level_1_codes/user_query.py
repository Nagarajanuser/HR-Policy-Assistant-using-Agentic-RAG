import os
import uuid
from dotenv import load_dotenv  

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv() # Load .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


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

# ---------------------------------------------------
# Load Trained BM25 Model
# ---------------------------------------------------

bm25 = BM25Encoder().load("bm25_values.json")


# ---------------------------------------------------
# Hybrid Search Function
# ---------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.7,
):
    """
    alpha = 1.0 -> Pure Semantic Search
    alpha = 0.0 -> Pure Keyword Search
    alpha = 0.7 -> Recommended Hybrid
    """

    # Dense Embedding
    dense_vector = embedding_model.embed_query(query)

    # Sparse Vector
    sparse_vector = bm25.encode_queries(query)

    # Hybrid Scaling
    dense_vector, sparse_vector = hybrid_convex_scale(
        dense_vector,
        sparse_vector,
        alpha=alpha
    )

    # Query Pinecone
    response = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return response



# ---------------------------------------------------
# Example
# ---------------------------------------------------

query = "What is the leave policy?"

results = hybrid_search(
    query=query,
    top_k=5,
    alpha=0.7
)


# ---------------------------------------------------
# Display Results
# ---------------------------------------------------

print("\nTop Results\n")

for i, match in enumerate(results["matches"], start=1):

    metadata = match["metadata"]

    print(f"Rank   : {i}")
    print(f"Score  : {match['score']:.4f}")
    print(f"Source : {metadata.get('source')}")
    print(f"Page   : {metadata.get('page')}")
    print(f"Text   : {metadata.get('text')[:250]}")
    print("-" * 80)