import os
from dotenv import load_dotenv
load_dotenv()   # Load .env file

from pinecone import Pinecone

index_name = os.getenv("INDEX_NAME")
 
# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if index_name in pc.list_indexes().names():
    print(f"Connecting to index: {index_name}...")
    index = pc.Index(index_name)
    try:
        index.delete(delete_all=True)
        print(f"Successfully deleted all vector data from index '{index_name}'.")
    except Exception as e:
        print(f"Note/Error while clearing index '{index_name}': {e}")
else:
    print(f"Index '{index_name}' does not exist.")
