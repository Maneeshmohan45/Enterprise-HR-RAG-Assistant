"""Pre-retrieval detection of clearly out-of-domain requests (not HR questions).

An HR assistant answers from the company's HR documents. A request to do arithmetic, write code, write
a poem or answer general-knowledge trivia can never be answered from those documents, so it must not
reach retrieval: unrelated HR chunks would be fetched and an answer (or a source card) could look
grounded when it is not.

What it is
    * Deterministic, high-precision rules for clearly non-HR *task types*. It declines the whole request
      before retrieval, so there is no LLM call and no source card.
What it is not
    * Not the security boundary (Phase 1 is) and not a complete topic classifier. Open-ended trivia and
      anything the rules do not name is handled by the relevance gate (an isolated, calibrated second
      layer in pipeline.py), and finally by the grounding rules.
    * It takes only the question text, so it cannot see roles, access levels or documents.

Compound requests ("What is the carry-forward limit, and also write a bash one-liner?") are split into
clauses. The HR clauses are still answered and the non-HR clause is declined with a notice; if no real HR
question remains, the whole request is declined.

Precision matters more than coverage: a genuine HR question must never be declined. General-knowledge
rules therefore never apply when the question also contains an HR word ("who won the employee of the
month award" stays answerable).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

OUT_OF_DOMAIN_MESSAGE = ("I can only help with questions about our HR policies and documents, so I can't "
                         "answer that one. Please ask an HR-related question.")
DOMAIN_PARTIAL_NOTICE = ("Note: part of your request is not about HR policies or documents (for example a "
                         "calculation, code, creative writing or general knowledge), so I have not answered that part.")

_NUMWORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
            r"hundred|thousand)")
_ASK = r"(?:write|create|generate|give|show|provide|make|need|want|draft|build|implement|code|fix|debug|refactor|suggest)"
_LANG = (r"(?:python|java(?:script)?|typescript|c\+\+|c#|golang|rust|php|ruby|sql|html|css|bash|shell|"
         r"react|node(?:\.js)?|kotlin|swift)")

# A question with any of these is about the workplace, so the general-knowledge rules do not apply to it.
_HR_ANCHOR = re.compile(
    r"\b(?:hr|human\s+resources|employees?|employers?|employment|staff|workplace|company|office|team|"
    r"managers?|departments?|colleagues?|leaves?|holidays?|vacation|attendance|payroll|salary|pay\s*slips?|"
    r"benefits?|insurance|probation|notice\s+period|resign\w*|onboarding|handbook|polic(?:y|ies)|appraisals?|"
    r"performance|work\s+from\s+home|wfh|remote|hybrid|shifts?|overtime|job|role|hiring|recruit\w*|"
    r"induction|training|reimburse\w*|expenses?|grievances?|harassment|code\s+of\s+conduct)\b")

DOMAIN_RULES: list[dict] = [
    {"name": "arithmetic", "patterns": [
        # a bare expression: "600 * 700 + 0", "what is 2+2", "calculate 15 x 4"
        r"^(?:(?:what\s+is|what's|whats|calculate|compute|solve|evaluate|find|simplify)\s*)?"
        r"(?:the\s+(?:value|result|answer)\s+of\s*)?\(?\s*\d[\d\s\.\+\-\*/x\u00d7\u00f7\^%\(\),=]*\d\s*\)?\s*=?\s*\??$",
        r"\b\d+(?:\.\d+)?\s*(?:plus|minus|times|multiplied\s+by|divided\s+by|to\s+the\s+power\s+of)\s*\d+",
        r"\bwhat\s+is\s+\d+(?:\.\d+)?\s*%\s*of\s*\d+",
        r"\b(?:square\s+root|cube\s+root|factorial|logarithm|derivative|integral|quadratic\s+equation|"
        r"prime\s+numbers?)\b",
        r"\bsolve\s+for\s+[a-z]\b",
        # an expression inside a longer sentence: "calculate my reimbursement if I spent (1250 * 3) + (450 / 2) - 75"
        r"\d\s*[*\u00d7\u00f7]\s*[\d(]",                      # 3 * 4, 3 x 4 written with a sign
        r"\)\s*[*\u00d7\u00f7/+\-]\s*[\d(]|[\d)]\s*[*\u00d7\u00f7/]\s*\(",   # (a) + (b), 3 * (b)
        r"\d\s+/\s+\d|\d\s+x\s+\d|\d\s*\+\s*\d",       # 450 / 2, 3 x 4, 2+2
        r"(?:\d\s+[-+]\s+){2,}\d",                           # 100 - 20 - 5
        rf"\b{_NUMWORD}\s+(?:plus|minus|times|multiplied\s+by|divided\s+by)\s+{_NUMWORD}\b",
        r"\b(?:sum|product|difference|quotient|average)\s+of\s+\d",
    ]},
    {"name": "programming", "patterns": [
        r"```",
        r"\bdef\s+\w+\s*\(",
        r"#include\s*<",
        r"\bpublic\s+static\s+void\b",
        r"\bconsole\.log\s*\(",
        rf"\b(?:write|create|generate|implement|debug|refactor|code|fix|build)\b[^.?!]{{0,40}}\b"
        rf"(?:function|snippet|algorithm|regex|{_LANG}\s+(?:code|script|program|function|class|snippet|query))\b",
        rf"\b(?:write|create|generate|implement|code|build)\b[^.?!]{{0,40}}\b(?:in|using|with)\s+{_LANG}\b",
        r"\bwrite\s+(?:me\s+)?(?:some\s+|a\s+|the\s+)?code\b(?!\s+of\s+(?:conduct|ethics))",
        rf"\b{_LANG}\s+(?:program|script|function|code)\s+(?:to|for|that)\b",
        rf"\b{_ASK}\b[^.?!]{{0,50}}\b(?:one-?liners?|{_LANG}\s+(?:one-?liners?|commands?|code|scripts?|programs?|"
        rf"functions?|classes|class|snippets?|quer(?:y|ies)))\b",
        r"\b(?:shell|terminal|command[- ]line|cmd|powershell)\s+(?:command|commands|script|one-?liner)\b",
        r"\b(?:command|script|one-?liner)\s+to\s+(?:delete|remove|list|find|kill|install|restart|copy|move|rename|"
        r"search|download|grep|sort)\b",
    ]},
    {"name": "creative_writing", "patterns": [
        r"\b(?:write|compose|create|generate|draft|make\s+up|pen)\b[^.?!]{0,30}\b(?:poems?|poetry|sonnets?|"
        r"haikus?|limericks?|lyrics|songs?|stor(?:y|ies)|fairy\s*tales?|jokes?|riddles?|raps?|ballads?|"
        r"novels?|screenplays?)\b",
        r"\btell\s+me\s+(?:a\s+|another\s+)?(?:joke|story|riddle)\b",
        r"\b(?:poem|poetry|haiku|limerick|sonnet)\s+(?:about|on)\b",
    ]},
    {"name": "general_knowledge_outcome", "category": "general_knowledge", "patterns": [   # asking for a sports / awards RESULT: never HR, whatever else the sentence says
        r"\b(?:who\s+won|who\s+wins|who\s+scored|final\s+score|what\s+was\s+the\s+score|match\s+result)\b[^.?!]{0,60}\b"
        r"(?:world\s*cup|fifa|olympics?|ipl|nba|nfl|wimbledon|premier\s+league|champions\s+league|grand\s+slam|"
        r"super\s+bowl|oscars?|grammys?|match|final|tournament|league)\b",
        r"\b(?:world\s*cup|fifa|olympics?|ipl|nba|nfl|wimbledon|premier\s+league|champions\s+league|super\s+bowl)\b"
        r"[^.?!]{0,60}\b(?:who\s+won|final\s+score|winner|the\s+score)\b",
    ]},
    {"name": "general_knowledge", "needs_no_hr_word": True, "patterns": [
        r"\b(?:world\s*cup|fifa|football|soccer|cricket|ipl|nba|nfl|olympics?|wimbledon|premier\s+league|"
        r"champions\s+league|grand\s+slam|oscars?|grammys?|box\s+office|bollywood|hollywood)\b",
        r"\bcapital\s+(?:city\s+)?of\b",
        r"\bwho\s+(?:won|invented|discovered|painted|composed|directed|founded)\b",
        r"\bwho\s+(?:is|was)\s+the\s+(?:current\s+)?(?:president|prime\s+minister|king|queen)\s+of\b",
        r"\b(?:weather|temperature)\b[^.?!]{0,20}\b(?:in|today|tomorrow|tonight)\b",
        r"\bstock\s+price\b|\bbitcoin\b|\bcrypto(?:currency)?\b|\bexchange\s+rate\b",
        r"\brecipes?\b|\bhow\s+to\s+(?:cook|bake|lose\s+weight)\b",
        r"\btranslate\b[^.?!]{0,60}\b(?:to|into)\s+(?:french|spanish|german|hindi|chinese|japanese|tamil|"
        r"telugu|arabic)\b",
        r"\bmeaning\s+of\s+life\b|\bspeed\s+of\s+light\b|\bsolar\s+system\b|\bphotosynthesis\b",
    ]},
]

for _rule in DOMAIN_RULES:   # compile once; a bad pattern fails at import, not at question time
    _rule["_compiled"] = [re.compile(p) for p in _rule["patterns"]]

_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*$")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad"), None)
_MARKDOWN = dict.fromkeys(map(ord, "*~#>|_"), " ")


@dataclass(frozen=True)
class DomainAnalysis:
    out_of_domain: bool               # the WHOLE request is not an HR question: decline it
    category: str | None = None       # arithmetic | programming | creative_writing | general_knowledge
    permitted_question: str | None = None   # compound request: the HR part that may still be answered


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = re.sub(r"(?<=[\d)])\s*\*\s*(?=[\d(])", " \u00d7 ", text)      # a * between numbers is a multiplication sign
    return " ".join(text.translate(_MARKDOWN).lower().split())


def _classify(text: str) -> str | None:
    """The out-of-domain category of one piece of normalised text, or None if it may be an HR question."""
    if _DATE_LIKE.match(text):
        return None
    has_hr_word = bool(_HR_ANCHOR.search(text))
    for rule in DOMAIN_RULES:
        if rule.get("needs_no_hr_word") and has_hr_word:
            continue
        if any(p.search(text) for p in rule["_compiled"]):
            return rule.get("category", rule["name"])
    return None


# Clauses of a compound request: sentences, ";", "and also / also / plus / as well as", and "and" before a command.
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.?!])\s+|\s*;\s*|\s+(?:and\s+also|also|as\s+well\s+as|plus)\s+|"
    r",?\s+and\s+(?=(?:please\s+)?(?:write|create|generate|compose|calculate|compute|translate|tell\s+me|give\s+me|"
    r"show\s+me|solve|draw|code|debug|find\s+me|work\s+out)\b)", re.I)
_REQUEST_START = re.compile(
    r"^(?:please\s+)?(?:what|how|why|when|where|who|which|is|are|can|could|does|do|should|will|would|list|give|show|"
    r"tell|explain|describe|compare|summari[sz]e|find)\b", re.I)
_QUESTION_WORD = re.compile(r"^(?:what|how|why|when|where|who|which|is|are|can|could|does|do|should|will|would)\b", re.I)


def _is_request(clause: str) -> bool:
    return "?" in clause or bool(_REQUEST_START.match(clause.strip()))


def _join(clauses: list[str]) -> str:
    return " ".join(c + ("?" if _QUESTION_WORD.match(c) else ".") for c in clauses)


def analyze_domain(question: str) -> DomainAnalysis:
    """Decide whether a request is clearly not an HR question. Takes ONLY the question text on purpose."""
    whole = _classify(_normalize(question))
    clauses = [c.strip(" ,;.?!") for c in _CLAUSE_SPLIT.split(question) if c and c.strip(" ,;.?!")]
    if len(clauses) <= 1:
        return DomainAnalysis(whole is not None, whole)
    verdicts = [(c, _classify(_normalize(c))) for c in clauses]
    out_of_domain = [category for _, category in verdicts if category]
    if not out_of_domain:
        return DomainAnalysis(whole is not None, whole)       # a pattern that spans clauses: decline whole
    keep = [c for c, category in verdicts if category is None and _is_request(c)]
    if not keep:
        return DomainAnalysis(True, out_of_domain[0])         # nothing real is left to answer
    return DomainAnalysis(False, out_of_domain[0], permitted_question=_join(keep))
