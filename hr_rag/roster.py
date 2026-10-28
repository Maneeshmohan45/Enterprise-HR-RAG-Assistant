"""Department rosters and headcounts that are complete by construction.

The problem this solves
    The employee directory is indexed one chunk per employee. "List everyone in Engineering" then needs six
    separate chunks to all rank in the top results at once; in testing the model often received none of them,
    answered with two names, and even claimed that employee IDs "are not included". The chunks that were missing
    were never retrieved, so no prompt or evaluation change could have fixed it.

What it does
    1. INGESTION (``build_roster_chunks``): besides the per-employee chunks, add ONE chunk per department that
       lists every member and the headcount, and one directory summary chunk. They inherit the directory's
       access level (confidential): an Employee can never retrieve them, exactly as before.
    2. ANSWERING (``plan_roster_answer``): when a question is only a roster or headcount request and the roster
       chunk of the department it names was retrieved, the answer is written from that chunk, so it cannot be
       partial. Anything more complicated still goes to the model.
    3. HONESTY (``RosterPlan.notice``): a roster-style question that is answered from individual rows only gets
       a visible note that the list may be incomplete.

Nothing here can see roles or access levels: it works on chunks and sources that retrieval has already filtered.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from . import enrichment

DIRECTORY_DOCUMENT_TYPE = "employee_directory"
ROSTER_HEADER = "[Department Roster |"
SUMMARY_HEADER = "[Directory Summary |"
ROSTER_NOTICE = ("Note: this list or count was not taken from the complete department roster, so it may be "
                 "incomplete. Please check the employee directory.")


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def _fields(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and " " not in key.strip():
            out[key.strip()] = value.strip()
    return out


def _roster_document(template: Document, text: str, section: str, header: str, department: str,
                     size: int, kind: str) -> Document:
    meta = dict(template.metadata)
    for stale in ("row_number", "employee_id"):
        meta.pop(stale, None)
    document_id = hashlib.sha256(f"{meta['file_name']}|{kind}|{department}".encode()).hexdigest()[:20]
    meta.update({
        "section_title": section, "content_kind": kind, "department": department, "roster_size": size,
        "chunk_header": header, "document_id": document_id, "chunk_index": 1, "total_chunks": 1,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "summary": enrichment.create_summary(text), "keywords": enrichment.extract_keywords(text),
        "tags": enrichment.create_tags(text), "character_count": len(text), "chunk_character_count": len(text),
        "word_count": len(text.split()),
        "chunk_id": hashlib.sha256(f"{document_id}|{text}".encode()).hexdigest()[:24],
    })
    return Document(page_content=text, metadata=meta)


def build_roster_chunks(chunks: list[Document]) -> list[Document]:
    """One complete roster chunk per department, plus one summary chunk, built from the directory's row chunks."""
    rows = [c for c in chunks
            if c.metadata.get("document_type") == DIRECTORY_DOCUMENT_TYPE and c.metadata.get("employee_id")
            and _fields(c.page_content).get("department")]
    if not rows:
        return []
    # The "department" in a row chunk's METADATA is the whole file's department from the manifest, the same for
    # every employee. The employee's own department is a field in the row text, so group by that.
    by_department: dict[str, list[Document]] = {}
    for row in rows:
        by_department.setdefault(_fields(row.page_content)["department"].strip(), []).append(row)
    built = []
    for department in sorted(by_department):
        members = sorted(by_department[department], key=lambda c: c.metadata["employee_id"])
        lines = []
        for number, row in enumerate(members, start=1):
            f = _fields(row.page_content)
            lines.append(f'{number}. {f.get("employee_name", row.metadata.get("section_title", ""))} | '
                         f'{f.get("designation", "")} | {row.metadata["employee_id"]} | '
                         f'{f.get("employment_type", "")} | {f.get("work_location", "")}')
        text = (f"{ROSTER_HEADER} {department}]\nDepartment: {department}\n"
                f"Headcount: {len(members)} employees (this is the complete list of employees in the "
                f"{department} department)\nEmployees (name | designation | employee ID | employment type | "
                f"location):\n" + "\n".join(lines) +
                f"\nThis roster answers: list of employees in {department}, who works in {department}, "
                f"how many employees are in {department}, {department} headcount and team members.")
        built.append(_roster_document(members[0], text, f"Roster: {department}",
                                      f"{ROSTER_HEADER} {department}]", department, len(members), "roster"))
    counts = {d: len(v) for d, v in sorted(by_department.items())}
    text = (f"{SUMMARY_HEADER} All Departments]\nTotal employees: {sum(counts.values())} across {len(counts)} "
            f"departments (this is the complete list of departments).\nHeadcount by department:\n"
            + "\n".join(f"- {d}: {n}" for d, n in counts.items())
            + "\nThis summary answers: how many employees does the company have, total number of employees, "
              "company headcount, headcount by department, how many employees are in each department, "
              "list of all departments.")
    built.append(_roster_document(rows[0], text, "Directory summary", f"{SUMMARY_HEADER} All Departments]",
                                  "All Departments", sum(counts.values()), "roster_summary"))
    return built


# ---------------------------------------------------------------------------
# answering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Roster:
    department: str
    count: int
    members: list[dict] = field(default_factory=list)


def parse_roster_text(text: str) -> Roster | None:
    """Read a roster chunk. The header line is NOT relied on: the context builder strips a chunk's header from
    the source text, so the ``Department:`` and ``Headcount:`` lines identify a roster."""
    department = re.search(r"^Department:\s*(.+?)\s*$", text, re.M)
    count = re.search(r"^Headcount:\s*(\d+)", text, re.M)
    if not (department and count):
        return None
    members = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\.\s*(.+?)\s*\|\s*(.*?)\s*\|\s*(\S+)\s*\|\s*(.*?)\s*\|\s*(.*)$", line)
        if m:
            members.append({"name": m.group(1), "designation": m.group(2), "employee_id": m.group(3),
                            "employment_type": m.group(4), "work_location": m.group(5)})
    return Roster(department.group(1).strip(), int(count.group(1)), members)


def parse_summary_text(text: str) -> tuple[int, int, list[tuple[str, int]]] | None:
    total = re.search(r"Total employees:\s*(\d+)\s+across\s+(\d+)", text)
    rows = [(m.group(1).strip(), int(m.group(2))) for m in re.finditer(r"^- (.+?):\s*(\d+)\s*$", text, re.M)]
    return (int(total.group(1)), int(total.group(2)), rows) if total else None


_ALIASES = {"human resources": ("hr",), "quality assurance": ("qa",), "administration": ("admin",),
            "devops": ("dev ops",), "data science": ("datascience",)}
_COUNT = re.compile(
    r"\bhow\s+many\s+(?:\w+\s+){0,2}?(?:employees|people|persons|staff|members|engineers|developers|interns|managers)\b|"
    r"\bheadcount\b|\bhead\s+count\b|\bnumber\s+of\s+(?:employees|people|staff|members)\b|\bteam\s+size\b", re.I)
_LIST = re.compile(
    r"\b(?:list|name|names|show|give|display|enumerate)\b[^.?!]{0,60}\b(?:employees|people|staff|members|team|everyone|"
    r"engineers)\b|\bwho\s+(?:are|is)\s+(?:the\s+)?(?:employees|members|staff|people)\b|"
    r"\bwho\s+(?:works?|belongs?)\s+(?:in|at|on|for)\b|\bmembers?\s+of\b|\b(?:employees|staff|people|members)\s+"
    r"(?:in|of|from|under)\b|\broster\b|\bteam\s+members\b|\beveryone\s+in\b", re.I)
_COMPANY_WIDE = re.compile(
    r"\bhow\s+many\s+(?:employees|people|staff)\b[^.?!]{0,50}\b(?:company|organi[sz]ation|total|overall|altogether|in\s+all|"
    r"do\s+we\s+have|are\s+there|does\s+\w+\s+have)\b|\btotal\s+(?:number\s+of\s+)?(?:employees|headcount|staff)\b|"
    r"\bheadcount\s+(?:by|per|across|of\s+each|for\s+each|of\s+all)\b|\blist\s+(?:all\s+)?(?:the\s+)?departments\b|"
    r"\bhow\s+many\s+departments\b|\bhow\s+many\s+employees\s+are\s+there\s+in\s+each\s+department\b", re.I)
# A roster question names a department, or uses a roster-specific phrase. "How many leave days do employees get?"
# and "Show me the leave policy for employees" are about policy, not about who works where.
KNOWN_DEPARTMENTS = ("engineering", "data science", "human resources", "hr", "quality assurance", "qa", "devops",
                      "finance", "marketing", "sales", "support", "administration", "admin", "management",
                      "operations", "legal", "product", "design")
_DEPARTMENT_WORD = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(d) for d in KNOWN_DEPARTMENTS) + r")(?![a-z])", re.I)
_ROSTER_MARKER = re.compile(
    r"\broster\b|\bheadcount\b|\bhead\s+count\b|\bemployee\s+directory\b|\ball\s+(?:the\s+)?employees\b|"
    r"\bevery(?:one|\s+employee)\b|\blist\s+of\s+employees\b|\bteam\s+members\b|\bmembers\s+of\b|"
    r"\bwho\s+works?\s+(?:in|at|for)\b", re.I)
# Words a pure roster request may contain; anything else (email, phone, leave, salary ...) is more than a roster.
_ROSTER_WORDS = frozenset("""please list name names designation designations designated id ids employee employees employee's
    staff people persons members member team teams roster headcount head count how many number total all every each
    show give display tell me who are is works work working in of the department departments dept according per from
    directory details full complete entire current company organization organisation here there under and also with
    their titles title job roles role positions position check exact only just what does do have has a an to for on at
    by location locations employment type types listed lists using use our us we hr's""".split())


def roster_intent(question: str) -> str | None:
    """"summary" (company-wide), "count", "list", or None when the question is not about a roster or headcount."""
    if _COMPANY_WIDE.search(question):
        return "summary"
    if not (_DEPARTMENT_WORD.search(question) or _ROSTER_MARKER.search(question)):
        return None
    if _LIST.search(question):
        return "list"
    if _COUNT.search(question):
        return "count"
    return None


def _names_for(department: str) -> list[str]:
    key = department.lower()
    return [key, *_ALIASES.get(key, ())]


def _mentions(question: str, department: str) -> bool:
    q = question.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", q) for name in _names_for(department))


def _roster_only(question: str, departments: list[str]) -> bool:
    known = {w for d in departments for name in _names_for(d) for w in re.findall(r"[a-z']+", name)}
    words = set(re.findall(r"[a-z']+", question.lower()))
    return not (words - _ROSTER_WORDS - known)


@dataclass(frozen=True)
class RosterPlan:
    text: str | None = None          # a complete answer written from the roster chunk, or None
    notice: bool = False             # the question looks like a roster question but is answered from rows only
    intent: str | None = None
    departments: tuple[str, ...] = ()
    source_indexes: tuple[int, ...] = ()


def _list_text(rosters: list[tuple[int, Roster]]) -> str:
    blocks = []
    for index, roster in rosters:
        head = f'{roster.department} has {roster.count} employee{"s" if roster.count != 1 else ""} [S{index}]:'
        blocks.append("\n".join([head] + [f'{n}. {m["name"]} \u2014 {m["designation"]} ({m["employee_id"]})'
                                          for n, m in enumerate(roster.members, start=1)]))
    return "\n\n".join(blocks)


def plan_roster_answer(question: str, sources: list[dict]) -> RosterPlan:
    """Decide how a roster or headcount question is answered. ``sources`` are the retrieved context chunks."""
    intent = roster_intent(question)
    if intent is None:
        return RosterPlan()
    rosters = [(i, r) for i, s in enumerate(sources, start=1) if (r := parse_roster_text(s["text"]))]
    from_directory = any(s.get("file_name") and s["file_name"].endswith("employee_directory.csv") for s in sources)
    if intent == "summary":
        for index, source in enumerate(sources, start=1):
            parsed = parse_summary_text(source["text"])
            if parsed and _roster_only(question, [r.department for _, r in rosters]):
                total, departments, rows = parsed
                text = (f"The employee directory lists {total} employees across {departments} departments [S{index}]:\n"
                        + "\n".join(f"- {name}: {count}" for name, count in rows))
                return RosterPlan(text, False, intent, (), (index,))
        return RosterPlan(None, from_directory, intent)
    mentioned = [(i, r) for i, r in rosters if _mentions(question, r.department)]
    if mentioned and _roster_only(question, [r.department for _, r in rosters]):
        return RosterPlan(_list_text(mentioned), False, intent, tuple(r.department for _, r in mentioned),
                          tuple(i for i, _ in mentioned))
    return RosterPlan(None, from_directory, intent, tuple(r.department for _, r in mentioned))
