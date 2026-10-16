"""Ask the HR assistant questions from the terminal.

    python ask.py                         # interactive, as an employee
    python ask.py --role hr               # HR staff can also see confidential files
    python ask.py -q "How many casual leave days do I get?"
    python ask.py --include-archived      # also search superseded policy versions

NOTE: --role is a simulation for learning. A real deployment must take the role
from a login system (SSO / JWT), never from something the user types.
"""
import argparse
import sys

from hr_rag import evaluation
from hr_rag.config import ROLE_ACCESS, settings
from hr_rag.pipeline import RAGPipeline, format_response


def show_evaluation(report: dict | None) -> None:
    if not report:
        return
    failed = [k for k, v in report.items() if isinstance(v, bool) and not v]
    print(f"\n[quality check] score {report['evaluation_score']}%"
          + (f" | failed: {', '.join(failed)}" if failed else " | all checks passed"))
    if report.get("ungrounded_numbers"):
        print("[warning] numbers in the answer that are not in the sources: "
              + ", ".join(report["ungrounded_numbers"]))


def show_usage(response: dict) -> None:
    usage, timings = response.get("usage") or {}, response.get("timings") or {}
    if usage:
        mark = "~" if usage.get("estimated") else ""
        print(f"[usage] {response['style']} | tokens in {mark}{usage['input_tokens']:,} / out {mark}{usage['output_tokens']:,}"
              f" | {usage['context_chunks']} sources | {timings.get('total_s', 0):.1f}s")


def collect_feedback(response: dict) -> None:
    try:
        rating = int(input("Rate the answer 1-5 (Enter to skip): ") or 0)
    except ValueError:
        return
    if not 1 <= rating <= 5:
        return
    helpful = input("Was it helpful? (y/n): ").strip().lower() in {"y", "yes"}
    comment = input("Comment (optional): ").strip()
    record = evaluation.record_feedback(response, rating, helpful, comment, settings)
    print("Thanks! Suggested follow-up:", "; ".join(record["improvement_actions"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="HR RAG assistant")
    parser.add_argument("-q", "--question", help="Ask one question and exit.")
    parser.add_argument("--role", default="employee", choices=sorted(ROLE_ACCESS))
    parser.add_argument("--include-archived", action="store_true")
    parser.add_argument("--style", choices=["concise", "detailed"], default=settings.answer_style,
                        help="concise = short, simple answers (default); detailed = fuller answers.")
    parser.add_argument("--feedback", action="store_true", help="Ask for a rating after each answer.")
    args = parser.parse_args()

    if not settings.chroma_dir.exists():
        print("No index found. Build it first with:  python ingest.py")
        return 1
    try:
        pipeline = RAGPipeline(settings)
    except RuntimeError as error:
        print(error)
        return 1
    user = {"user_id": "CLI_USER", "role": args.role}
    print(f"HR assistant ready | role: {args.role} | LLM: {settings.llm_provider} | style: {args.style}")

    def answer_one(question: str) -> None:
        try:
            response = pipeline.ask(question, user, include_archived=args.include_archived, style=args.style)
        except (ValueError, PermissionError) as error:
            print(error)
            return
        except Exception as error:  # LLM/network problems
            print(f"Error: {error}")
            return
        print("\n" + format_response(response))
        show_evaluation(response["evaluation"])
        show_usage(response)
        if args.feedback:
            collect_feedback(response)

    if args.question:
        answer_one(args.question)
        return 0
    while True:
        try:
            question = input("\nYour HR question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            answer_one(question)
    return 0


if __name__ == "__main__":
    sys.exit(main())