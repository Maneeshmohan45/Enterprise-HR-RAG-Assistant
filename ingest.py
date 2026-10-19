"""Build (or rebuild) the vector index from data/raw.

    python ingest.py            # full run: load, clean, chunk, embed, store
    python ingest.py --dry-run  # everything except embedding (no model download)
"""
import argparse
import json
import sys

from hr_rag.config import settings
from hr_rag.ingestion import prepare_chunks, run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the HR documents.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and chunk only; do not embed or write the index.")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Add to the existing index instead of rebuilding it from scratch.")
    args = parser.parse_args()

    if args.dry_run:
        chunks, report = prepare_chunks(settings)
    else:
        report = run_ingestion(settings, reset=not args.keep_existing)

    print("\nINGESTION SUMMARY")
    print("=" * 60)
    for key, value in report["totals"].items():
        print(f"{key:<22}: {value}")
    for label in ("excluded", "invalid_documents", "indexed_with_warnings", "duplicate_documents", "failed", "unlisted_files"):
        if report[label]:
            print(f"\n{label}:\n{json.dumps(report[label], indent=2)}")
    if not args.dry_run:
        print(f"\nIndex saved in: {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
