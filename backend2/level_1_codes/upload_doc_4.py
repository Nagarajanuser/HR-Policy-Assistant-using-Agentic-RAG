
# Upload Documents to Pinecone
# During Upload Metadata filter added

import os
import uuid
from dotenv import load_dotenv
import warnings

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from metadata import DOCUMENT_METADATA

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

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

embedding_model = HuggingFaceEmbeddings(   # create an instance of the HuggingFaceEmbeddings class with the specified model name
    model_name="BAAI/bge-small-en-v1.5"    # The model produces embeddings of 384-dimensional vectors
)

# ---------------------------------------------------
# PDF Folder Path
# ---------------------------------------------------

PDF_FOLDER = "docs"

# ---------------------------------------------------
# Load PDF Files
# ---------------------------------------------------

loader = PyPDFDirectoryLoader(PDF_FOLDER)

documents = loader.load()

print(f"Total Pages Loaded : {len(documents)}")

# ---------------------------------------------------
# Create Text Splitter
# ---------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

# ---------------------------------------------------
# Split Documents into Chunks
# ---------------------------------------------------

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks Created : {len(chunks)}")

chunk_texts = [doc.page_content for doc in chunks]
print(f"Total Chunks Created : {len(chunk_texts)}")

# ---------------------------------------------------
# BM25 Sparse Encoder
# ---------------------------------------------------

bm25 = BM25Encoder().default()  # create an instance of the BM25Encoder class using the default configuration

bm25.fit(chunk_texts)             # Create a BM25 encoder and fit it to the sample documents.
                                # This step is necessary to build the vocabulary and compute the IDF values for the documents.

bm25.dump("bm25_values.json")   # This simply saves the trained BM25 model to disk. (Optional)

bm25 = BM25Encoder().load("bm25_values.json")   # This loads the previously saved BM25 model. (Optional)


# ---------------------------------------------------
# Prepare Hybrid Vectors
# ---------------------------------------------------
vectors = []
for doc in chunks:
    text = doc.page_content
    dense = embedding_model.embed_documents([text])[0]
    sparse = bm25.encode_documents(text)
    source_file = os.path.basename(doc.metadata["source"])
    document_metadata = DOCUMENT_METADATA.get(
        source_file,
        {}
    )
    vectors.append({
        "id": str(uuid.uuid4()),
        "values": dense,
        "sparse_values": {
            "indices": sparse["indices"],
            "values": sparse["values"]
        },
        "metadata": {
            "text": text,
            "source": source_file,
            "page": doc.metadata["page"] + 1,
            "category": document_metadata.get("category"),
            "department": document_metadata.get("department"),
            "country": document_metadata.get("country"),
            "location": document_metadata.get("location"),
            "language": document_metadata.get("language"),
            "document_type": document_metadata.get("document_type"),
            "version": document_metadata.get("version"),
            "access_level": document_metadata.get("access_level"),
            "effective_date": document_metadata.get("effective_date"),
            "expiry_date": document_metadata.get("expiry_date"),
            "owner": document_metadata.get("owner"),
            "tags": document_metadata.get("tags")
        }

    })

print(f"Vectors Prepared : {len(vectors)}")

response = index.upsert(vectors=vectors)

print(response)

print(index.describe_index_stats())