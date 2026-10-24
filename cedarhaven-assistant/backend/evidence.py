"""
Evidence Explorer backend helper.

Builds, for every VERIFIED cited page, the real page text with the passages that
relate to the question highlighted, plus a coarse evidence status.

The status is derived from retrieval/extraction results (how many of the question's
key terms are found in the highlighted passages of the cited pages). It is a
heuristic, not a calibrated probability, and it never uses the LLM's self-reported
confidence.
"""
from typing import Any, Dict, List

from rag_engine import GAP_RE, I18N

SUPPORTED_AT = 0.6
PARTIAL_AT = 0.3


def assess(engine, question: str, english_query: str, cited: List[Dict[str, Any]],
           answer_text: str) -> Dict[str, Any]:
    evidence: List[Dict[str, Any]] = []
    for c in cited:
        ev = engine.get_evidence(c["doc_id"], c["page_number"], question, english_query)
        if ev:
            evidence.append(ev)

    if not evidence:
        return {
            "evidence": [],
            "answer_status": "insufficient",
            "status_note": "No supporting passage was found in the Cedarhaven documents.",
        }

    q_terms = engine._query_terms(question) | engine._query_terms(english_query)
    hit_terms = set()
    has_gap_statement = False
    for ev in evidence:
        for h in ev["highlights"]:
            hit_terms |= engine._query_terms(h)
            if GAP_RE.search(h):
                has_gap_statement = True
    coverage = len(q_terms & hit_terms) / len(q_terms) if q_terms else 0.0

    says_not_specified = any(
        lab["not_specified"].lower()[:25] in answer_text.lower() for lab in I18N.values()
    )
    if has_gap_statement or says_not_specified:
        return {
            "evidence": evidence,
            "answer_status": "insufficient",
            "status_note": "The documents state that part of this detail is not published, "
                           "so it cannot be verified from the available material.",
        }
    if coverage >= SUPPORTED_AT:
        status, note = "supported", "Key terms of your question were found in the cited pages."
    elif coverage >= PARTIAL_AT:
        status, note = "partial", ("Only some parts of your question were found in the cited pages. "
                                   "Check the highlighted passages before relying on the rest.")
    else:
        status, note = "insufficient", "The cited pages match your question only weakly."
    return {"evidence": evidence, "answer_status": status, "status_note": note}


def empty_result(note: str) -> Dict[str, Any]:
    return {"evidence": [], "answer_status": "insufficient", "status_note": note}
