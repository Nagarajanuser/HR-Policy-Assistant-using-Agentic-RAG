# HR Policy Assistant using Agentic RAG
Developed an enterprise-grade HR Policy Assistant leveraging Agentic Retrieval-Augmented Generation (Agentic RAG) to provide employees with accurate, context-aware answers to HR policy queries through natural language interactions. Designed and implemented an intelligent multi-stage RAG pipeline using LangGraph for workflow orchestration, integrating hybrid intent detection, history-aware query rewriting, semantic caching, hybrid retrieval, and LLM-based response generation. The solution intelligently routes user requests to HR APIs (Leave Balance, Payroll ) or the RAG pipeline, improving response accuracy, reducing latency, and optimizing LLM inference costs.


# Key Responsibilities & Features:
```text
Designed and implemented an Agentic RAG architecture using LangGraph with multi-step workflow orchestration.
Built RESTful APIs using FastAPI for HR policy search, conversation management, and API routing.
Implemented Hybrid Intent Detection (Rule-based + Semantic Similarity) and LLM-based Query Classification to route requests between HR APIs and the RAG pipeline.
Developed History-aware, Hybrid, and Canonical Query Rewriting to improve retrieval quality for follow-up and ambiguous user queries.
Implemented Semantic Cache using Pinecone to reduce repeated LLM calls, improve response latency, and optimize operational costs.
Built a Hybrid Search solution combining Dense Vector Search (Embeddings) and Sparse BM25 Search, followed by Cross-Encoder Reranking for highly relevant document retrieval.
Implemented metadata-based filtering (department, location, access level, document type) to support secure and personalized document retrieval.
Developed a multi-document PDF ingestion pipeline with document chunking, embedding generation, and vector indexing in Pinecone.
Integrated MySQL for conversation history, enabling context-aware interactions and persistent chat memory.
Used Ollama Local LLM for secure on-premises inference and LangSmith for tracing, monitoring, and debugging AI workflows.
Implemented enterprise-grade logging, environment-based configuration, source citation, and modular architecture following production best practices.
```

# Technology Stack
•	Python 
•	FastAPI 
•	LangGraph 
•	LangChain 
•	Ollama 
•	Pinecone 
•	HuggingFace Embeddings 
•	BM25 
•	CrossEncoder Reranker 
•	MySQL 
•	dotenv 
•	LangSmith 
•	RAGAS (Retrieval-Augmented Generation Assessment )


# My final RAG Application
```text
                    Start
                      │
                      ▼
            Question Validation
                      │
                      ▼
     Hybrid Intent Detection (Rule + Semantic)
          ├──────────────┬──────────────┬──────────────┐
          │              │              │              │
          ▼              ▼              ▼              ▼
   LLM Query      Leave API     Payroll API    Document API
 Classification         │              │              │
          │             │              │              │
          ▼             └─────── End ──┴────── End ───┘
 History-aware Query Rewrite
          │
          ▼
 Hybrid Query Rewrite
          │
          ▼
 Semantic Cache Lookup
     ┌───────────┴───────────┐
     │                       │
 Cache Hit              Cache Miss
     │                       │
     ▼                       ▼
Answer Generation      Hybrid Search
                             │
                             ▼
                 Cross-Encoder Reranking
                             │
                             ▼
                   Answer Generation
                             │
                             ▼
                            End
```