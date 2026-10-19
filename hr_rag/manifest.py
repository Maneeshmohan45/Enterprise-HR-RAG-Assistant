"""Read data/manifest.yaml: per-file access level, version and status."""
from __future__ import annotations

from pathlib import Path

import yaml

FALLBACK_ACCESS = "confidential"   # unknown file => hidden from normal employees


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"defaults": {}, "files": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("defaults", {})
    data.setdefault("files", {})
    return data


def resolve_entry(manifest: dict, file_name: str) -> tuple[dict, bool]:
    """Return ``(settings_for_file, was_listed_in_manifest)``.

    Files missing from the manifest are NOT given the default access level.
    A new file nobody has classified yet is hidden from regular employees
    (fail closed) until someone adds it to the manifest.
    """
    listed = file_name in manifest["files"]
    entry = {**manifest["defaults"], **manifest["files"].get(file_name, {})}
    if not listed:
        entry["access_level"] = FALLBACK_ACCESS
        entry.setdefault("document_type", "unclassified")
    entry.setdefault("title", Path(file_name).stem.replace("_", " ").title())
    entry.setdefault("document_type", "hr_document")
    entry.setdefault("department", "Human Resources")
    entry.setdefault("access_level", "internal")
    entry.setdefault("is_current", True)
    entry.setdefault("version_number", 1)
    return entry, listed
