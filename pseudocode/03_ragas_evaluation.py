"""Step 3 - RAGAS evaluation."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from lab_common import PROMPT_V1, PROMPT_V2, build_vectorstore, ensure_environment, make_embeddings, make_llm
from qa_pairs import QA_PAIRS


ensure_environment()

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Run the RAG chain for one question and return answer plus contexts."""

    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm).invoke({"context": ctx_str, "question": question})
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    return {"answer": answer_text, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Run all QA pairs through the given prompt version."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def build_ragas_dataset(rag_results: list):
    """Convert RAG outputs into a RAGAS EvaluationDataset."""

    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["reference"],
        )
        for item in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Evaluate RAG outputs with the four required RAGAS metrics."""

    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = make_llm()
    emb_eval = make_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([value for value in raw if value is not None]))

    for key, value in scores.items():
        star = " ⭐" if key == "faithfulness" and value >= 0.8 else ""
        print(f"  {key:30s}: {value:.4f}{star}")
    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\nComparison table:")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    output_path = ROOT / "data" / "ragas_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("💾 Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
