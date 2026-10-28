"""The whole data-preparation pipeline: files -> chunks -> vector store."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from . import cleaning, enrichment, roster
from .chunking import chunk_documents
from .config import PROJECT_ROOT, Settings, settings as default_settings
from .loaders import load_file
from .manifest import load_manifest, resolve_entry
from .vectorstore import build_vector_store, get_embeddings


def _document_id(meta: dict) -> str:
    identity = "|".join(
        str(meta.get(k, ""))
        for k in ("file_name", "section_index", "section_title", "page_number", "row_number", "sheet_name")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def prepare_chunks(cfg: Settings = default_settings, verbose: bool = True) -> tuple[list[Document], dict]:
    """Run every step up to (and including) chunking. No models are needed."""
    log = print if verbose else (lambda *a, **k: None)
    manifest = load_manifest(cfg.manifest_path)
    report: dict = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "files": {}, "excluded": [], "failed": [], "unsupported": [],
        "unlisted_files": [], "invalid_documents": [], "duplicate_documents": [],
        "indexed_with_warnings": [],
    }
    keep_if_invalid: set[str] = set()

    if not cfg.raw_data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {cfg.raw_data_dir}")

    # 1-4. discover, load, attach manifest metadata -------------------------
    loaded: list[Document] = []
    for path in sorted(p for p in cfg.raw_data_dir.iterdir() if p.is_file()):
        if path.name.startswith(".") or path.name == "manifest.yaml":
            continue
        if path.suffix.lower() not in cfg.supported_extensions:
            report["unsupported"].append(path.name)
            continue
        entry, listed = resolve_entry(manifest, path.name)
        if not listed:
            report["unlisted_files"].append(path.name)
            log(f"WARNING: {path.name} is not in manifest.yaml -> treated as '{entry['access_level']}'.")
        if entry.get("exclude"):
            report["excluded"].append({"file": path.name, "reason": entry.get("reason", "excluded")})
            log(f"Excluded: {path.name} ({entry.get('reason', 'excluded')})")
            continue
        if entry.get("keep_if_invalid"):
            keep_if_invalid.add(path.name)
        try:
            docs = load_file(path)
        except Exception as error:  # one bad file must not stop the whole run
            report["failed"].append({"file": path.name, "error": str(error)})
            log(f"FAILED to load {path.name}: {error}")
            continue

        for doc in docs:
            doc.metadata.update({
                "title": entry["title"],
                "document_type": entry["document_type"],
                "department": entry["department"],
                "access_level": entry["access_level"],
                "is_current": bool(entry["is_current"]),
                "status": "current" if entry["is_current"] else "archived",
                "version_number": int(entry["version_number"]),
                "source_path": _relative(path),
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
            })
            if entry.get("policy_family"):
                doc.metadata["policy_family"] = entry["policy_family"]
        report["files"][path.name] = {"sections": len(docs), "access_level": entry["access_level"],
                                       "is_current": bool(entry["is_current"])}
        loaded.extend(docs)
        log(f"Loaded {path.name}: {len(docs)} sections")

    # 5. clean ---------------------------------------------------------------
    cleaned = cleaning.clean_documents(loaded)

    # 6. validate ------------------------------------------------------------
    valid: list[Document] = []
    for doc in cleaned:
        errors = cleaning.validate_document(doc)
        if errors:
            item = {"file": doc.metadata["file_name"],
                    "section": doc.metadata.get("section_title"), "errors": errors}
            if doc.metadata["file_name"] in keep_if_invalid:
                # Deliberate test/example records stay searchable, clearly marked as invalid.
                doc.metadata["validation_status"] = "invalid"
                doc.metadata["validation_errors"] = ", ".join(errors)
                report["indexed_with_warnings"].append(item)
                valid.append(doc)
            else:
                report["invalid_documents"].append(item)
        else:
            valid.append(doc)

    # 7. duplicates (current documents first, so archived copies lose) ---------
    valid.sort(key=lambda d: not d.metadata["is_current"])
    unique, duplicates = cleaning.deduplicate(valid)
    report["duplicate_documents"] = [
        {"file": d.metadata["file_name"], "section": d.metadata.get("section_title")}
        for d in duplicates
    ]

    # 8. ids + enrichment ----------------------------------------------------------
    final_docs = []
    for doc in unique:
        meta = doc.metadata
        meta["document_id"] = _document_id(meta)
        meta["summary"] = enrichment.create_summary(doc.page_content)
        meta["keywords"] = enrichment.extract_keywords(doc.page_content)
        meta["tags"] = enrichment.create_tags(doc.page_content)
        meta["character_count"] = len(doc.page_content)
        meta["word_count"] = len(doc.page_content.split())
        final_docs.append(doc)

    # 9. chunk ---------------------------------------------------------------------
    chunks = chunk_documents(final_docs)
    # One complete roster chunk per department (+ a summary) next to the per-employee chunks, so a department's
    # full member list can be retrieved as a single chunk. Same access level as the directory it comes from.
    chunks = chunks + roster.build_roster_chunks(chunks)
    per_file = Counter(c.metadata["file_name"] for c in chunks)
    for name, count in per_file.items():
        report["files"][name]["chunks"] = count
    report["totals"] = {
        "files_indexed": len(per_file),
        "sections_loaded": len(loaded),
        "sections_invalid": len(report["invalid_documents"]),
        "sections_indexed_with_warnings": len(report["indexed_with_warnings"]),
        "sections_duplicate": len(duplicates),
        "chunks": len(chunks),
    }
    if len({c.metadata["chunk_id"] for c in chunks}) != len(chunks):
        raise RuntimeError("Chunk IDs are not unique; check document_id generation.")
    return chunks, report


def run_ingestion(cfg: Settings = default_settings, embeddings=None, persist_dir: Path | None = None,
                  reset: bool = True, verbose: bool = True) -> dict:
    """Prepare chunks, embed them and store them in Chroma. Returns the report."""
    chunks, report = prepare_chunks(cfg, verbose=verbose)
    if not chunks:
        raise ValueError("No chunks were produced. Check data/raw and manifest.yaml.")
    embeddings = embeddings or get_embeddings(cfg.embedding_model)
    build_vector_store(chunks, embeddings, persist_dir or cfg.chroma_dir, cfg.collection_name, reset=reset)
    report["embedding_model"] = cfg.embedding_model
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / "ingestion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
