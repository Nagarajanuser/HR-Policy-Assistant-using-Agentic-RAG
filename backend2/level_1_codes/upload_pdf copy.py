
import re
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from metadata import DOCUMENT_METADATA
file_path = "pdfs/AttendancePolicy.pdf"

# ---------------------------------------------------
# HuggingFace Embedding Model
# ---------------------------------------------------
embedding_model = HuggingFaceEmbeddings(   # create an instance of the HuggingFaceEmbeddings class with the specified model name
    model_name="BAAI/bge-small-en-v1.5"    # The model produces embeddings of 384-dimensional vectors
)

# Step3 : Clean the Elements
#Instead of cleaning raw text, clean each element.
def clean_text(text: str) -> str:  
    text = re.sub(r'[-=_]{3,}', '', text)
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Step 4 — Keep Only Useful Elements
def clean_elements(elements):
    allowed = {
        "Title",
        "NarrativeText",
        "ListItem",
        "Table"
    }
    clean_elements = []
    for e in elements:
        if type(e).__name__ in allowed:
            e.text = clean_text(e.text)
            clean_elements.append(e)


#Step 5 — Chunk the Elements
def chunk_elements():
    chunks = partition_pdf(
        filename="AttendancePolicy.pdf",
        strategy="fast",
        chunking_strategy="by_title",
        max_characters=1200,
        new_after_n_chars=1000,
        combine_text_under_n_chars=300
    )
    return chunks


# Step 6 — Attach Metadata
def build_metadata(chunks):
    documents = []
    for chunk in chunks:
        documents.append({
            "text": chunk.text,
            "metadata": {
                "source": "AttendancePolicy.pdf",
                "page": chunk.metadata.page_number,
                "category": "Attendance",
                "version": "1.0"
            }
        })

'''def create_embeddings(documents):
    for doc in documents:
        embedding = embedding_model.embed_query(doc["text"])
        index.upsert([(
            str(uuid.uuid4()),
            embedding,
            doc["metadata"]
        )])'''


def process_pdf(file_path):

    # Step1 : Parse the PDF
    elements = partition_pdf(
        filename=file_path,
        strategy="fast",
    )
    elements = clean_elements(elements)
    chunks = chunk_elements(elements)
    docs = build_metadata(chunks)
    print('docs', docs)
    #embeddings = create_embeddings(docs)
    #upload_to_pinecone(embeddings)



