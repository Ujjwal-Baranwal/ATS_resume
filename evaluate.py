import sys
import types

# --- RAGAS / LANGCHAIN FIX ---
# Prevents ragas from crashing due to missing legacy VertexAI path
dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex


import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Ragas requires LLM and Embedding objects for evaluation
eval_llm = ChatOllama(model="gemma3", temperature=0)  # Use deterministic output for evaluation
eval_embeddings = OllamaEmbeddings(model="embeddinggemma")

# Wrap models for Ragas compatibility
wrapped_eval_llm = LangchainLLMWrapper(eval_llm)
wrapped_eval_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

def evaluate_rag_pipeline(question, response_text, retrieved_contexts, ground_truth=None):
    """
    Evaluates RAG pipeline accuracy using Ragas metrics.
    """
    # Extract raw text from retrieved LangChain Document objects
    context_strings = [doc.page_content for doc in retrieved_contexts]

    data = {
        "question": [question],
        "contexts": [context_strings],
        "answer": [response_text],
    }

    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    # Selected metrics
    metrics = [faithfulness]
    if ground_truth:
        metrics.append([context_recall, context_precision])

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=wrapped_eval_llm,
        embeddings=wrapped_eval_embeddings
    )

    return results

