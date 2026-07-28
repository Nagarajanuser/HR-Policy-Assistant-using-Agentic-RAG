```text
rag_langgraph_hr_assist/
│
├──backend/
│    │
│    ├── main.py
│    │
│    ├── core/
│    │   ├── config.py
│    │   ├── logger.py
│    │   ├── security.py
│    │   ├── database.py
│    │   ├── redis.py
│    │   ├── middleware.py
│    │   ├── constants.py
│    │   └── startup.py
│    │
│    ├── api/
│    │   ├── v1/
│    │   │
│    │   │── routes/
│    │   │     ├── chat.py
│    │   │     ├── feedback.py
│    │   │     ├── upload.py
│    │   │     ├── admin.py
│    │   │     └── health.py
│    │   │
│    │   │── schemas/
│    │   │     ├── chat_schema.py
│    │   │     ├── feedback_schema.py
│    │   │     └── upload_schema.py
│    │   │
│    │   │── services/
│    │   │     ├── chat_service.py
│    │   │     ├── feedback_service.py
│    │   │     └── upload_service.py
│    │   │
│    │   └── dependencies/
│    │
│    ├── ai/
│    │   │
│    │   ├── graph/
│    │   │     ├── graph.py
│    │   │     ├── state.py
│    │   │     └── nodes/
│    │   │           ├── validation.py
│    │   │           ├── intent_detection.py
│    │   │           ├── history_rewrite.py
│    │   │           ├── query_rewrite.py
│    │   │           ├── semantic_cache.py
│    │   │           ├── retrieve.py
│    │   │           ├── rerank.py
│    │   │           ├── answer.py
│    │   │           ├── guardrails.py
│    │   │           ├── citation.py
│    │   │           └── save_chat.py
│    │   │
│    │   ├── prompts/
│    │   │     ├── system_prompt.py
│    │   │     ├── rewrite_prompt.py
│    │   │     ├── answer_prompt.py
│    │   │     └── guard_prompt.py
│    │   │
│    │   ├── embeddings/
│    │   │     └── embedding_model.py
│    │   │
│    │   ├── llm/
│    │   │     ├── llm.py
│    │   │     └── model_factory.py
│    │   │
│    │   ├── retriever/
│    │   │     ├── pinecone.py
│    │   │     ├── hybrid_search.py
│    │   │     └── metadata_filter.py
│    │   │
│    │   ├── reranker/
│    │   │     └── cross_encoder.py
│    │   │
│    │   ├── cache/
│    │   │     └── semantic_cache.py
│    │   │
│    │   └── evaluation/
│    │         └── ragas.py
│    │
│    ├── repositories/
│    │   ├── user_repository.py
│    │   ├── chat_repository.py
│    │   ├── feedback_repository.py
│    │   └── session_repository.py
│    │
│    ├── models/
│    │   ├── user.py
│    │   ├── session.py
│    │   ├── feedback.py
│    │   └── audit.py
│    │
│    ├── shared/
│    │   ├── exceptions/
│    │   ├── schemas/
│    │   ├── services/
│    │   └── utils/
│    │
│    ├── tests/
│    │   ├── unit/
│    │   ├── integration/
│    │   └── api/
│    │
│    ├── logs/
│    │
│    ├── .env
│    ├── requirements.txt
│    └── README.md
├──DOCS/
│   └── Architecture.md
│   └── components.md
│   └── services.md
│   └── tests.md
│   └── deployment.md
│   └── user_authentication.md
│   └── user_interaction.md
│   └── document_upload.md
│   └── admin_panel.md
│   └── feedback_system.md
│   └── performance_monitoring.md
│
├── frontend/                           # Angular Application
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/              # Reusable UI components
│   │   │   │   ├── chat-widget/
│   │   │   │   │   ├── chat-widget.component.ts
│   │   │   │   │   ├── chat-widget.component.html
│   │   │   │   │   ├── chat-widget.component.scss
│   │   │   │   ├── login/               # Login/Authentication
│   │   │   │   │   ├── login.component.ts
│   │   │   │   │   ├── login.component.html
│   │   │   │   ├── registration/          # Employee Registration
│   │   │   │   │   ├── registration.component.ts
│   │   │   │   │   ├── registration.component.html
│   │   │   │   ├── upload/
│   │   │   │   ├── admin/
│   │   │   │   └── feedback/
│   │   │   ├── services/
│   │   │   │   ├── auth.service.ts
│   │   │   │   ├── chat.service.ts
│   │   │   │   ├── upload.service.ts
│   │   │   │   └── admin.service.ts
│   │   │   ├── guards/                 # Route Guards
│   │   │   │   ├── auth.guard.ts
│   │   │   │   ├── admin.guard.ts
│   │   │   ├── pipes/                  # Pipes for date formatting, etc.
│   │   │   ├── models/                 # TypeScript interfaces
│   │   │   │   ├── user.model.ts
│   │   │   │   ├── message.model.ts
│   │   │   │   ├── upload.model.ts
│   │   │   │   └── feedback.model.ts
│   │   │   ├── environments/           # Environment variables
│   │   │   │   ├── environment.ts
│   │   │   │   └── environment.prod.ts
│   │   │   ├── interceptors/           # HTTP Interceptors (add auth tokens)
│   │   │   │   └── auth.interceptor.ts
│   │   │   └── utils/
│   │   ├── assets/                 # Static assets (logos, favicons)
│   │   ├── favicon.ico
│   │   └── index.html
│   │
│   ├── angular.json                   # Angular CLI configuration
│   ├── package.json                   # Frontend dependencies
│   ├── tsconfig.json                  # TypeScript configuration
│   └── README.md                      # Frontend README

```