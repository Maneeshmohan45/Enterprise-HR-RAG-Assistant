"""Calibrate the relevance gate on YOUR machine with the real reranker.

    python scripts/calibrate_relevance.py

It retrieves for HR questions and for out-of-domain questions, prints the best reranker score for each,
and suggests a threshold. Nothing is changed: you copy the suggested lines into .env yourself.
The gate is an extra layer; the deterministic domain rules already catch arithmetic, code, creative
writing and common trivia without any threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HR_QUESTIONS = [
    "How many casual leave days do I get?", "How many annual leave days do employees get?",
    "What is the process to apply for leave?", "What are the office working hours?",
    "What is the work from home policy?", "What is the onboarding process for new joiners?",
    "Which public holidays are listed in the holiday calendar?", "What is the notice period for resignation?",
    "How is attendance recorded?", "What is the probation period?", "Can I carry forward unused leave?",
    "What is the dress code?", "Who should I contact for HR queries?", "What is the sick leave policy?",
    "What is the maternity leave policy?", "What documents do I need on my first day?",
    "What are the responsibilities of a Data Scientist?", "What skills does the Data Scientist role require?",
    "Compare the leave policy v1 and v2.", "How do I claim expense reimbursement?",
    # in-domain questions the documents may not answer: they must still reach the model
    "According to the handbook, what is the penalty for bringing a pet to the office?",
    "What is the policy on sabbatical leave?", "Is there a gym reimbursement benefit?",
    "What is the policy on bereavement leave?", "What happens if I am late three times in a month?",
    "How many casual leaves can I take at once, and who approves them?",
]
OOD_QUESTIONS = [
    "Who won the 2018 football World Cup?", "What is the capital of France?", "Explain how photosynthesis works.",
    "Who is the CEO of Google?", "What is quantum computing?", "How do I lose weight fast?",
    "Tell me about the French Revolution.", "What is the best smartphone to buy?", "Who painted the Mona Lisa?",
    "What is the tallest mountain in the world?", "How does a car engine work?", "What is machine learning?",
    "Recommend a good movie to watch.", "What is the population of India?", "How do airplanes fly?",
    "What is the stock market?", "Explain the theory of relativity.", "What causes earthquakes?",
    "What is 600 * 700 + 0?", "Write a poem about the moon.", "Write a Python function to reverse a string.",
]


def suggest_threshold(hr_scores: list[float], ood_scores: list[float]) -> dict:
    """Pick a threshold between the two groups. Pure function (tested without any model)."""
    if not hr_scores or not ood_scores:
        raise ValueError("Need scores for both HR and out-of-domain questions.")
    lowest_hr, highest_ood = min(hr_scores), max(ood_scores)
    if lowest_hr > highest_ood:
        return {"separable": True, "threshold": round((lowest_hr + highest_ood) / 2, 3),
                "margin": round(lowest_hr - highest_ood, 3), "lowest_hr": lowest_hr, "highest_ood": highest_ood,
                "hr_below": 0, "ood_above": 0}
    # overlap: be conservative (never refuse a real HR question): sit just under the lowest HR score
    threshold = round(lowest_hr - 1e-3, 3)
    return {"separable": False, "threshold": threshold, "margin": round(lowest_hr - highest_ood, 3),
            "lowest_hr": lowest_hr, "highest_ood": highest_ood,
            "hr_below": sum(s < threshold for s in hr_scores), "ood_above": sum(s >= threshold for s in ood_scores)}


def top_score(pipeline, question: str, levels: list[str]) -> float | None:
    from hr_rag.retrieval import process_query, split_subqueries
    sub = [process_query(p) for p in split_subqueries(question)]
    results = pipeline.retriever.retrieve(process_query(question), levels, False, sub)
    scores = [d.metadata.get("rerank_score") for d in results["reranked"] if d.metadata.get("rerank_score") is not None]
    return max(scores) if scores else None


def main() -> None:
    from hr_rag.pipeline import RAGPipeline
    from hr_rag.retrieval import allowed_levels_for
    pipeline = RAGPipeline()
    levels = allowed_levels_for("employee")
    print(f"Reranker: {pipeline.cfg.reranker_model} (mode after first use: see below)\n")
    hr = [(q, top_score(pipeline, q, levels)) for q in HR_QUESTIONS]
    ood = [(q, top_score(pipeline, q, levels)) for q in OOD_QUESTIONS]
    mode = getattr(pipeline.retriever.reranker, "mode", "?")
    print(f"reranker mode: {mode}")
    if mode == "fallback":
        print("The real cross-encoder did not load, so these scores are NOT usable for calibration.")
        return
    for title, rows in (("HR questions (must NOT be refused)", hr), ("Out-of-domain questions (should be refused)", ood)):
        print(f"\n{title}")
        for q, s in sorted(rows, key=lambda r: (r[1] is None, r[1])):
            print(f"  {('%.3f' % s) if s is not None else 'none':>8}  {q}")
    hr_scores = [s for _, s in hr if s is not None]
    ood_scores = [s for _, s in ood if s is not None]
    result = suggest_threshold(hr_scores, ood_scores)
    print(f"\nlowest HR score {result['lowest_hr']:.3f} | highest out-of-domain score {result['highest_ood']:.3f}")
    if result["separable"]:
        print(f"The groups separate (margin {result['margin']}). Suggested settings for .env:")
    else:
        print("The groups OVERLAP, so no threshold separates them perfectly. A conservative threshold that never "
              f"refuses an HR question here would let {result['ood_above']} out-of-domain question(s) through. "
              "Suggested (conservative) settings for .env:")
    print(f"  RELEVANCE_MIN_RERANK_SCORE={result['threshold']}\n  RELEVANCE_GATE_MODE=enforce")
    print("\nCheck the lowest HR questions above first: a real HR question near the threshold is the risk.")


if __name__ == "__main__":
    main()
