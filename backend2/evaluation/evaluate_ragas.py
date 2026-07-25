import os
import sys
import json
import asyncio

# Fix for asyncio 'Event loop is closed' error on Windows during teardown
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

# Import your RAG application
from main import graph, logged_in_user, embedding_model     # Python can find main.py
from langchain_ollama import ChatOllama

ragas_llm = ChatOllama(model="qwen2.5:1.5b")
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def ask_internal(question: str):
    """
    Runs one question through the existing LangGraph pipeline.
    """

    result = graph.invoke(
        {
            "question": question,
            "department": logged_in_user["department"],
            "country": logged_in_user["country"],
            "location": logged_in_user["location"],
            "access_level": logged_in_user["access_level"]
        }
    )

    return result


def load_questions():

    with open(
        "evaluation/test_dataset.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def build_evaluation_dataset():

    questions = load_questions()

    evaluation_dataset = []

    for item in questions:

        print("=" * 80)
        print("Question:", item["question"])

        result = ask_internal(item["question"])

        contexts = []

        # Collect reranked document texts
        for doc in result.get("reranked_docs", []):

            contexts.append(doc["text"])

        evaluation_dataset.append(
            {
                "user_input": item["question"],

                "retrieved_contexts": contexts,

                "response": result.get("answer", ""),

                "reference": item["ground_truth"]
            }
        )

        print("Answer:")
        print(result.get("answer", ""))

        print("\nRetrieved Contexts:")

        for i, context in enumerate(contexts, start=1):

            print(f"\nContext {i}")
            print(context)

    return evaluation_dataset


if __name__ == "__main__":

    dataset = build_evaluation_dataset()

    print("\n")
    print("=" * 80)
    print("Evaluation Dataset Built. Starting Ragas Evaluation...")
    print("=" * 80)

    hf_dataset = Dataset.from_list(dataset)
    
    try:
        result = evaluate(
            hf_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=ragas_llm,
            embeddings=embedding_model,
            raise_exceptions=False,
            # Set a high timeout because local LLMs can be slow
            run_config=RunConfig(timeout=300, max_workers=1)
        )
        
        print("\n" + "=" * 80)
        print("Ragas Evaluation Results")
        print("=" * 80)
        print(result)
        
        df = result.to_pandas()
        df.to_csv("evaluation/ragas_results_latest.csv", index=False)
        print("\nResults saved to evaluation/ragas_results_latest.csv")
    except Exception as e:
        print(f"\nRagas evaluation failed: {e}")