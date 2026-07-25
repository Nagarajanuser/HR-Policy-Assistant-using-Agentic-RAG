import os
from dotenv import load_dotenv
load_dotenv()   # Load .env file

from pinecone import Pinecone,ServerlessSpec

index_name = os.getenv("INDEX_NAME")
 
#Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

#Create the index
if index_name not in pc.list_indexes().names():
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",   #Sparse vector search is not supported for hybrid search, so we use dotproduct instead
        #serverless=ServerlessSpec(min_node_count=1, max_node_count=1),
        spec=ServerlessSpec(
            cloud='aws',
            region="us-east-1"
        )
    )
    print("Index created.")
else:
    print("Index already exists.")

print("Done")