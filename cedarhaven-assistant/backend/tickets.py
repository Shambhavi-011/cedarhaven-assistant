"""
Conversation-to-Ticket (CH-SRM-01 style service request) for the Cedarhaven assistant.

Persistence: SQLite file (Python stdlib, no new dependency).
Identity: the project has no login (the assignment says accounts are NOT required), so
each browser sends a random secret in the X-Client-Token header. Only its SHA-256 hash is
stored. Tickets are only ever listed/read for the same token. This is "possession of a
secret" access control, NOT real authentication: whoever has the token sees the tickets.

The only status backed by real logic is "Submitted" (CH-SRM-01 SRM 6.1). There is no staff
side, so tickets never move to In Progress / Resolved, and none is shown.
Nothing is sent to any real municipality, email or third party.
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

DB_PATH = os.getenv("TICKET_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.db"))

CATEGORIES = [
    "Waste & Recycling", "Roads & Potholes", "Streetlights & Signs", "Drainage & Water",
    "Parks & Trees", "Sidewalks & Winter", "By-law Complaint", "Permits & Licensing",
    "General Inquiry",
]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
# CH-SRM-01 SRM 2.1: P1 urgent, P2 high, P3 standard, P4 information
PRIORITY_CODE = {"Urgent": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
ID_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

CATEGORY_RULES: List[Tuple[str, str]] = [
    ("Waste & Recycling", r"garbage|recycl|organics|missed (pickup|collection)|waste|bin\b"),
    ("Roads & Potholes", r"pothole|road\b|sinkhole|asphalt"),
    ("Streetlights & Signs", r"streetlight|street light|stop sign|traffic (sign|signal)|signal outage"),
    ("Drainage & Water", r"catch basin|drain|flood|water.?main|leak"),
    ("Parks & Trees", r"\bpark\b|tree|branch|playground|play equipment|graffiti"),
    ("Sidewalks & Winter", r"sidewalk|snow|ice\b"),
    ("By-law Complaint", r"noise|barking|fence|parking|grass|by-?law"),
    ("Permits & Licensing", r"permit|licen[cs]e|encroachment|patio"),
]
URGENT_RE = re.compile(r"sinkhole|downed|fallen overhead|exposed sharp|gas leak|imminent|collapse|live wire", re.I)
HIGH_RE = re.compile(r"stop sign|catch basin|flood|fallen (limb|branch|tree)|blocking (a |the )?sidewalk|"
                     r"play(ground)? equipment|water.?main|blocked drain", re.I)
LOW_RE = re.compile(r"\bquestion\b|\binquiry\b|how do i|follow.?up|general information", re.I)


# ---------------------------------------------------------------- storage
def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY, owner_hash TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT NOT NULL, category TEXT NOT NULL, priority TEXT NOT NULL,
            priority_reason TEXT, location TEXT NOT NULL, date_first_noticed TEXT,
            contact TEXT, source_pages TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'Submitted', created_at TEXT NOT NULL)"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_owner ON tickets(owner_hash, created_at)")
    return con


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d.pop("owner_hash", None)
    d["source_pages"] = json.loads(d.get("source_pages") or "[]")
    return d


def _new_id() -> str:
    suffix = "".join(secrets.choice(ID_ALPHABET) for _ in range(6))
    return f"CH-{datetime.now(timezone.utc):%y%m%d}-{suffix}"


def _norm_loc(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def find_duplicate(owner: str, category: str, location: str) -> Optional[Dict[str, Any]]:
    """CH-SRM-01 SRM 7.1: same location + category within 72 hours. Searches the caller's own tickets only."""
    since = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM tickets WHERE owner_hash=? AND category=? AND created_at>=? ORDER BY created_at DESC",
            (owner, category, since),
        ).fetchall()
    for r in rows:
        if _norm_loc(r["location"]) == _norm_loc(location):
            return {"id": r["id"], "title": r["title"], "created_at": r["created_at"]}
    return None


# ---------------------------------------------------------------- rules
def suggest_category(text: str) -> str:
    for name, pat in CATEGORY_RULES:
        if re.search(pat, text, re.I):
            return name
    return "General Inquiry"


def suggest_priority(text: str, category: str) -> Tuple[str, str]:
    if URGENT_RE.search(text):
        return "Urgent", "Matches a P1 trigger (e.g. sinkhole, downed wire, exposed sharp debris) in CH-SRM-01, Page 3 (SRM 2.1)."
    if HIGH_RE.search(text):
        return "High", "Matches a P2 example (e.g. damaged stop sign, drainage, obstructed sidewalk) in CH-SRM-01, Page 3 (SRM 2.1)."
    if category == "General Inquiry" or LOW_RE.search(text):
        return "Low", "Looks like a general question or follow-up (P4) per CH-SRM-01, Page 3 (SRM 2.1)."
    return "Medium", "Routine issue, treated as P3 standard per CH-SRM-01, Page 3 (SRM 2.1)."


LOC_RE = re.compile(
    r"\b(?:at|on|near|corner of|intersection of|outside)\s+((?:\d+\s+)?[A-Z][\w.'-]*(?:\s+(?:and|&|at)?\s*[A-Z0-9][\w.'-]*){0,5})"
)


def extract_location(user_text: str) -> Optional[str]:
    """Only extracts what the user wrote. Returns None rather than guessing."""
    m = LOC_RE.search(user_text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- API models
class Msg(BaseModel):
    role: str
    content: str


class CitedIn(BaseModel):
    doc_id: str
    page_number: int


class DraftReq(BaseModel):
    messages: List[Msg]
    cited_pages: Optional[List[CitedIn]] = None
    language: Optional[str] = "English"


class TicketIn(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    priority_reason: Optional[str] = None
    location: Optional[str] = None
    date_first_noticed: Optional[str] = None
    contact: Optional[str] = None
    source_pages: Optional[List[CitedIn]] = None
    confirm_duplicate: bool = False


def validate(t: TicketIn) -> Tuple[Dict[str, Any], Dict[str, str]]:
    errors: Dict[str, str] = {}
    title = (t.title or "").strip()
    desc = (t.description or "").strip()
    loc = (t.location or "").strip()
    contact = (t.contact or "").strip()
    reason = (t.priority_reason or "").strip()[:300]
    if not 5 <= len(title) <= 120:
        errors["title"] = "Title must be 5 to 120 characters."
    if not 10 <= len(desc) <= 2000:
        errors["description"] = "Description must be 10 to 2000 characters."
    if t.category not in CATEGORIES:
        errors["category"] = "Choose a valid category."
    if t.priority not in PRIORITIES:
        errors["priority"] = "Choose Low, Medium, High or Urgent."
    if not 3 <= len(loc) <= 200:
        errors["location"] = "Location or nearest intersection is required (CH-SRM-01, SRM 1.2)."
    if len(contact) > 120:
        errors["contact"] = "Contact must be 120 characters or fewer."
    noticed = None
    if t.date_first_noticed:
        try:
            d = date.fromisoformat(t.date_first_noticed.strip())
            if d > date.today():
                errors["date_first_noticed"] = "Date cannot be in the future."
            noticed = d.isoformat()
        except ValueError:
            errors["date_first_noticed"] = "Use the format YYYY-MM-DD."
    clean = {
        "title": title, "description": desc, "category": t.category, "priority": t.priority,
        "priority_reason": reason, "location": loc, "date_first_noticed": noticed,
        "contact": contact or None,
    }
    return clean, errors


# ---------------------------------------------------------------- router
router = APIRouter(prefix="/api/tickets")
_engine_getter: Callable[[], Any] = lambda: None


def set_engine_getter(fn: Callable[[], Any]) -> None:
    global _engine_getter
    _engine_getter = fn


def owner_hash(token: Optional[str]) -> str:
    if not token or not TOKEN_RE.fullmatch(token):
        raise HTTPException(status_code=401, detail="Missing or invalid client token.")
    return hashlib.sha256(token.encode()).hexdigest()


def _valid_pages(pages: Optional[List[CitedIn]]) -> List[Dict[str, Any]]:
    engine = _engine_getter()
    if not pages or engine is None:
        return []
    valid = {(c.doc_id.upper(), c.page_number) for c in engine.chunks}
    out, seen = [], set()
    for p in pages:
        k = (p.doc_id.upper(), p.page_number)
        if k in valid and k not in seen:
            seen.add(k)
            out.append({"doc_id": k[0], "page_number": k[1]})
    return out


def _polish_with_llm(user_text: str, language: str) -> Optional[Dict[str, str]]:
    """Optional: ask Gemini to tidy title/description. It may not add facts. Returns None on any problem."""
    engine = _engine_getter()
    if engine is None:
        return None
    key = engine._gemini_key()
    if not key:
        return None
    try:
        raw = engine._gemini_request(
            "Rewrite a resident's problem report as a ticket. Use ONLY facts stated in the text. "
            "Never invent locations, dates, names, numbers or contact details. "
            'Return JSON: {"title": "<=100 chars", "description": "<=600 chars"}. '
            f"Write in {language}.",
            user_text, key, json_mode=True, timeout=12,
        )
        d = engine._parse_json_loose(raw)
        if isinstance(d.get("title"), str) and isinstance(d.get("description"), str):
            return {"title": d["title"].strip()[:120], "description": d["description"].strip()[:2000]}
    except Exception:
        return None
    return None


@router.post("/draft")
def draft_ticket(req: DraftReq, x_client_token: Optional[str] = Header(None)):
    owner = owner_hash(x_client_token)
    users = [m.content.strip()[:1500] for m in req.messages[-12:] if m.role == "user" and m.content.strip()]
    if not users:
        raise HTTPException(status_code=400, detail="There is no resident message to build a ticket from.")
    user_text = "\n".join(users[-4:])
    last = users[-1]

    category = suggest_category(user_text)
    priority, reason = suggest_priority(user_text, category)
    location = extract_location(user_text)
    polished = _polish_with_llm(user_text, (req.language or "English")[:40])
    title = (polished or {}).get("title") or (last[:100] + ("…" if len(last) > 100 else ""))
    description = (polished or {}).get("description") or user_text[:2000]

    missing = []
    if not location:
        missing.append("location")
    if len(description) < 10:
        missing.append("description")

    dup = find_duplicate(owner, category, location) if location else None
    return {
        "title": title, "description": description, "category": category, "priority": priority,
        "priority_reason": reason, "location": location, "date_first_noticed": None, "contact": None,
        "source_pages": _valid_pages(req.cited_pages), "missing": missing, "possible_duplicate": dup,
        "emergency_notice": (
            "For immediate danger to life, fire, gas leaks or a fallen utility wire, contact emergency "
            "services first. A 311 case is not an emergency response (CH-SRM-01, Pages 2-3)."
            if priority == "Urgent" else None
        ),
        "drafted_by": "ai+rules" if polished else "rules",
    }


@router.post("")
def create_ticket(body: TicketIn, x_client_token: Optional[str] = Header(None)):
    owner = owner_hash(x_client_token)
    clean, errors = validate(body)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    dup = find_duplicate(owner, clean["category"], clean["location"])
    if dup and not body.confirm_duplicate:
        raise HTTPException(status_code=409, detail={"duplicate": dup,
                            "message": "A similar ticket was created in the last 72 hours."})
    pages = json.dumps(_valid_pages(body.source_pages))
    now = datetime.now(timezone.utc).isoformat()
    for _ in range(5):
        tid = _new_id()
        try:
            with _conn() as con:
                con.execute(
                    "INSERT INTO tickets (id, owner_hash, title, description, category, priority, priority_reason,"
                    " location, date_first_noticed, contact, source_pages, status, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, owner, clean["title"], clean["description"], clean["category"], clean["priority"],
                     clean["priority_reason"], clean["location"], clean["date_first_noticed"], clean["contact"],
                     pages, "Submitted", now),
                )
            break
        except sqlite3.IntegrityError:
            continue
    else:
        raise HTTPException(status_code=500, detail="Could not allocate a ticket ID. Please try again.")
    # Read back from the database: only report success for what was actually persisted.
    with _conn() as con:
        row = con.execute("SELECT * FROM tickets WHERE id=? AND owner_hash=?", (tid, owner)).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="The ticket could not be saved.")
    return _row(row)


@router.get("")
def list_tickets(x_client_token: Optional[str] = Header(None)):
    owner = owner_hash(x_client_token)
    with _conn() as con:
        rows = con.execute("SELECT * FROM tickets WHERE owner_hash=? ORDER BY created_at DESC LIMIT 50",
                           (owner,)).fetchall()
    return {"tickets": [_row(r) for r in rows]}


@router.get("/{ticket_id}")
def get_ticket(ticket_id: str, x_client_token: Optional[str] = Header(None)):
    owner = owner_hash(x_client_token)
    with _conn() as con:
        row = con.execute("SELECT * FROM tickets WHERE id=? AND owner_hash=?", (ticket_id, owner)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")  # same answer for "not yours"
    return _row(row)
