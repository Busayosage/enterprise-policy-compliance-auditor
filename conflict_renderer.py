"""Conflict card parsing and HTML rendering for the Streamlit presentation layer.

Isolated here so tests can import without triggering the Streamlit runtime.
No eval() is used anywhere in this module — ast.literal_eval is safe for
Python literals (it only parses the AST, never executes arbitrary code).
"""

import ast
import json
import html as html_module


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_conflict_item(item) -> dict:
    """Normalise a conflict item from the LangGraph pipeline to a display dict.

    Handles:
    - dict   — from Pydantic structured output stored directly in AgentState
    - str starting with '{' — JSON or Python-literal dict repr from the LLM
    - plain str — the current pipeline format; split on ':' heuristic
    - '[REVIEW RECOMMENDED]' prefix from graph.check_cross_doc_conflicts

    Returns a normalised dict with keys:
        topic, policy_a, page_a, excerpt_a,
        policy_b, page_b, excerpt_b,
        explanation, review_flag, structured
    """
    if isinstance(item, dict):
        return _normalise_conflict_dict(item)

    raw = str(item).strip()

    # Strip the sentinel prefix the graph adds
    review_flag = raw.startswith("[REVIEW RECOMMENDED]")
    if review_flag:
        raw = raw[len("[REVIEW RECOMMENDED]"):].strip()

    # Looks like a dict literal — try to parse it safely
    if raw.startswith("{"):
        parsed = _try_parse_dict(raw)
        if parsed is not None:
            result = _normalise_conflict_dict(parsed)
            result["review_flag"] = result.get("review_flag") or review_flag
            return result

    # Plain text — split on the first ':' if it looks like "Topic: body"
    if ":" in raw:
        topic, body = raw.split(":", 1)
        if len(topic.strip()) <= 120:
            return {
                "topic": topic.strip(),
                "explanation": body.strip(),
                "review_flag": review_flag,
                "structured": False,
                "policy_a": None, "page_a": None, "excerpt_a": None,
                "policy_b": None, "page_b": None, "excerpt_b": None,
            }

    return {
        "topic": "Policy Conflict",
        "explanation": raw,
        "review_flag": review_flag,
        "structured": False,
        "policy_a": None, "page_a": None, "excerpt_a": None,
        "policy_b": None, "page_b": None, "excerpt_b": None,
    }


def conflict_card_html(parsed: dict, index: int) -> str:
    """Build a self-contained HTML string for one conflict card.

    Structured conflicts get a two-column policy-comparison grid.
    Plain-text conflicts get a single explanatory paragraph.
    All user-supplied strings are HTML-escaped before insertion.
    """
    def _e(v):
        return html_module.escape(str(v)) if v is not None else ""

    topic = _e(parsed.get("topic") or f"Conflict {index}")
    explanation = _e(parsed.get("explanation") or "")
    review_flag = bool(parsed.get("review_flag"))

    structured = parsed.get("structured", False)
    policy_a = parsed.get("policy_a")
    policy_b = parsed.get("policy_b")

    compare_html = ""
    if structured and (policy_a or policy_b):
        compare_html = _two_col_html(
            policy_a=_e(policy_a),
            page_a=parsed.get("page_a"),
            excerpt_a=_e(parsed.get("excerpt_a") or ""),
            policy_b=_e(policy_b),
            page_b=parsed.get("page_b"),
            excerpt_b=_e(parsed.get("excerpt_b") or ""),
        )

    explanation_html = ""
    if explanation:
        label = "Disagreement" if structured else ""
        label_html = (
            f'<div class="conflict-explanation-label">{label}</div>' if label else ""
        )
        explanation_html = (
            f'<div class="conflict-explanation">{label_html}{explanation}</div>'
        )

    footer_text = (
        "⚠ Human review required before relying on either policy statement"
        if review_flag
        else "Human review recommended before acting on conflicting statements"
    )
    footer_html = f'<div class="conflict-footer">{footer_text}</div>'

    return (
        f'<div class="conflict-card">'
        f'<h4>⚠ {topic}</h4>'
        f'{compare_html}'
        f'{explanation_html}'
        f'{footer_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse_dict(raw: str):
    """Try JSON then ast.literal_eval to parse a dict-like string. Returns None on failure."""
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def _normalise_conflict_dict(d: dict) -> dict:
    """Extract known fields from a conflict dict into the standard display shape."""
    topic = (
        d.get("topic") or d.get("conflict_topic") or d.get("title") or "Policy Conflict"
    )
    docs = d.get("documents") or d.get("sources") or []
    policy_a = d.get("policy_a") or d.get("document_a") or (docs[0] if len(docs) > 0 else None)
    policy_b = d.get("policy_b") or d.get("document_b") or (docs[1] if len(docs) > 1 else None)
    page_a = d.get("page_a")
    page_b = d.get("page_b")
    excerpt_a = d.get("excerpt_a") or d.get("statement_a") or d.get("text_a")
    excerpt_b = d.get("excerpt_b") or d.get("statement_b") or d.get("text_b")
    explanation = (
        d.get("contradiction_details")
        or d.get("explanation")
        or d.get("description")
        or d.get("detail")
        or ""
    )
    review_flag = bool(
        d.get("review_recommended") or d.get("review_flag")
    )
    structured = bool(policy_a or policy_b or excerpt_a or excerpt_b)
    return {
        "topic": str(topic),
        "policy_a": str(policy_a) if policy_a else None,
        "page_a": page_a,
        "excerpt_a": str(excerpt_a) if excerpt_a else None,
        "policy_b": str(policy_b) if policy_b else None,
        "page_b": page_b,
        "excerpt_b": str(excerpt_b) if excerpt_b else None,
        "explanation": str(explanation),
        "review_flag": review_flag,
        "structured": structured,
    }


def _two_col_html(
    policy_a: str, page_a, excerpt_a: str,
    policy_b: str, page_b, excerpt_b: str,
) -> str:
    def _policy_box(label: str, name: str, page, excerpt: str) -> str:
        page_str = f" — Page {page}" if page is not None else ""
        name_html = (
            f'<div class="conflict-policy-name">📄 {name}{page_str}</div>'
            if name
            else ""
        )
        excerpt_html = (
            f'<div class="conflict-excerpt">&ldquo;{excerpt}&rdquo;</div>'
            if excerpt
            else ""
        )
        return (
            f'<div class="conflict-policy-box">'
            f'<div class="conflict-policy-label">{label}</div>'
            f'{name_html}'
            f'{excerpt_html}'
            f'</div>'
        )

    box_a = _policy_box("Policy A", policy_a, page_a, excerpt_a)
    box_b = _policy_box("Policy B", policy_b, page_b, excerpt_b)
    return f'<div class="conflict-compare">{box_a}{box_b}</div>'
