"""
LangGraph graph definition for the Evidence-Aware Agentic RAG pipeline.

Stage 4b: LLM-based classification, evidence verification, conflict
detection and grounded answer generation. All nodes fall back gracefully
when the LLM is unavailable (no API key, network failure, timeout).

Stage 4b additions vs 4a:
  - classify_query: LLM structured output (ClassificationOutput) + heuristic fallback.
    No longer labels questions unsupported based on topic keywords.
  - plan_retrieval: discovers actual indexed source_filename values from ChromaDB
    metadata rather than matching against hardcoded fixture filenames.
  - assess_evidence: LLM evaluates evidence quality → EvidenceStatus
    (SUPPORTED|PARTIALLY_SUPPORTED|INSUFFICIENT|CONFLICTING|UNAVAILABLE).
    UNAVAILABLE on LLM failure — triggers retry if budget remains.
  - check_cross_doc_conflicts: LLM detects genuine contradictions; returns
    UNKNOWN (not NO_CONFLICT) on failure — safety-first posture.
  - generate_grounded_answer: calls NVIDIA NIM LLM with retrieved context;
    strips fabricated citations (filename + page validated against retrieved docs);
    uses answer_type "grounded" | "partial_grounded" | "unverified_grounded" |
    "excerpt_fallback" | "refusal" | "error" to reflect actual verification status.

Graph topology (unchanged from Stage 4a):
--------------------------------------------------------------

    START
      │
      ▼
  classify_query
      │
      ▼
  plan_retrieval
      │
      ▼
  retrieve_policy_chunks
      │
      ▼
  assess_evidence ──────────────────────────────────────────┐
      │                                                      │
      │ (conditional edge — _route_after_assessment)        │
      ├─ needs_retry & retry<2 ──► expand_retrieval ────────┘
      │                              (loops back to assess)
      ├─ CONFLICTING evidence ──► check_cross_doc_conflicts
      ├─ requires_cross_doc ────► check_cross_doc_conflicts
      │                               │
      └─ otherwise ───────────────────┘
                                      │
                                      ▼
                              generate_grounded_answer
                                      │
                                      ▼
                              log_audit_record → END

Architecture safeguards (all enforced here):
  - At most 3 total retrieval calls: 1 initial + up to 2 expansions
    (expand_retrieval gated by retry_count < _MAX_RETRIES=2).
  - Distinct terminal answer_type values: grounded | refusal | error.
  - No fabricated citations: sources is the literal retrieved docs list;
    LLM-cited passages are validated against retrieved chunk text.
  - answer_type="error" and "refusal" are distinct from "grounded".
  - Evidence coverage ≠ answer correctness (assessed separately).
  - LLM failure in any node triggers fallback — never silent wrong answer.
  - No SharedSystemClient.clear_system_cache() inside graph nodes.
  - retriever_override and llm_override injection keep all tests offline.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from agent.state import AgentState, QueryClassification, RetrievalPlan, EvidenceAssessment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2          # max expansions; total retrieval calls = 1 initial + _MAX_RETRIES
_EVIDENCE_THRESHOLD = 0.3 # heuristic fallback minimum coverage score
_BASE_K = 3               # initial retrieval count
_EXPAND_K_INCREMENT = 2   # added to k on each retry

_REFUSAL_MESSAGE = (
    "I cannot find enough evidence in the uploaded policy documents to answer that."
)
_PARTIAL_ANSWER_NOTICE = (
    "\n\n⚠ Note: The retrieved evidence only partially addresses this question. "
    "Consult the cited policy documents directly to verify completeness."
)
_UNVERIFIED_ANSWER_NOTICE = (
    "\n\n⚠ Note: Evidence verification could not be completed (assessment model "
    "unavailable). This answer is drawn from retrieved excerpts and has not been "
    "independently verified."
)
_CITATION_STRIPPED_NOTICE = (
    "\n\n[Source note: One or more cited documents were removed because they could "
    "not be verified against the retrieved policy content. Treat source attribution "
    "in this answer as unverified.]"
)
_CITATION_METADATA_NOTICE = (
    "\n\n*Note: Citations verified by filename and page metadata only. "
    "AI evidence assessment does not independently confirm factual accuracy — "
    "human review is required before acting on compliance conclusions.*"
)

# Stage 4b evidence status values (used in evidence_status field)
_STATUS_SUPPORTED = "SUPPORTED"
_STATUS_PARTIAL = "PARTIALLY_SUPPORTED"
_STATUS_INSUFFICIENT = "INSUFFICIENT"
_STATUS_CONFLICTING = "CONFLICTING"
_STATUS_UNAVAILABLE = "UNAVAILABLE"  # LLM assessment itself failed

# Conflict status values
_CONFLICT_DETECTED = "CONFLICT_DETECTED"
_NO_CONFLICT = "NO_CONFLICT"
_CONFLICT_UNKNOWN = "UNKNOWN"        # LLM analysis failed — not NO_CONFLICT

# Keyword sets for heuristic fallback classifier (no longer used for unsupported detection)
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "does", "did", "do",
    "the", "a", "an", "and", "or", "but", "nor",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "that", "this", "these", "those",
    "our", "their", "its", "my", "your", "his", "her",
    "not", "no", "any", "all", "each", "both", "more",
    "than", "then", "there", "here", "can", "could", "will",
    "would", "should", "may", "might", "must", "shall",
    "according", "under", "about", "between", "policy",
})

_CROSS_DOC_PHRASES = (
    "both policies", "all policies", "it and hr", "hr and it",
    "it and finance", "finance and it", "hr and finance", "finance and hr",
    "across policies", "compare", "difference between",
)

# Keywords that suggest a given source document is relevant to the query
_SOURCE_KEYWORD_MAP: dict[str, list[str]] = {
    "it":      ["it", "security", "password", "network", "access", "cyber",
                 "data", "encryption", "vpn", "malware", "firewall", "breach"],
    "hr":      ["hr", "human", "leave", "holiday", "vacation", "employee",
                 "personnel", "staff", "recruitment", "onboarding", "sick",
                 "maternity", "paternity"],
    "finance": ["finance", "expense", "receipt", "reimbursement", "cost",
                 "budget", "claim", "invoice", "purchase", "travel", "meal"],
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classify_query_heuristic(state: AgentState) -> dict:
    """Keyword-based fallback classifier.

    Stage 4b: does NOT classify any query as 'unsupported'. Answerability is
    determined through retrieval and evidence assessment, not topic keywords.
    """
    query_lower = state["query"].lower()

    requires_cross_doc = any(phrase in query_lower for phrase in _CROSS_DOC_PHRASES)
    if requires_cross_doc:
        return {
            "classification": QueryClassification(
                category="cross_document",
                intent="Question that spans multiple policy documents.",
                requires_cross_doc=True,
            )
        }

    word_count = len(state["query"].split())
    has_conjunction = " and " in query_lower or ";" in query_lower
    if word_count > 20 and has_conjunction:
        return {
            "classification": QueryClassification(
                category="multi_section",
                intent="Question requiring multiple policy sections.",
                requires_cross_doc=False,
            )
        }

    return {
        "classification": QueryClassification(
            category="direct_factual",
            intent="Factual question for policy lookup.",
            requires_cross_doc=False,
        )
    }


def _assess_evidence_heuristic(state: AgentState, docs: list) -> dict:
    """Keyword-overlap heuristic fallback for evidence assessment."""
    query_words = {w.lower() for w in state["query"].split() if len(w) > 3}
    combined_text = " ".join(d.page_content for d in docs).lower()

    content_words = {w for w in query_words if w not in _STOPWORDS}
    effective = content_words if content_words else query_words
    matched = sum(1 for w in effective if w in combined_text)
    coverage = matched / len(effective) if effective else 1.0

    unique_sources = len({d.metadata.get("source_filename", "") for d in docs})
    confidence = min(1.0, len(docs) / _BASE_K) * min(1.0, unique_sources / 1)

    sufficient = coverage >= _EVIDENCE_THRESHOLD and len(docs) >= 1
    needs_retry = (not sufficient) and (state.get("retry_count", 0) < _MAX_RETRIES)

    assessment = EvidenceAssessment(
        coverage=round(coverage, 3),
        confidence=round(confidence, 3),
        sufficient=sufficient,
    )
    status = _STATUS_SUPPORTED if sufficient else _STATUS_INSUFFICIENT
    return {
        "evidence": assessment,
        "evidence_status": status,
        "supporting_passages": [],
        "needs_retry": needs_retry,
    }


def _discover_indexed_sources(state: AgentState) -> list:
    """Query ChromaDB metadata to discover currently indexed source filenames.

    Returns an empty list when:
      - retriever_override is set (test mode — no real ChromaDB).
      - No documents are indexed yet.
      - Any error occurs during discovery.

    Never raises.
    """
    if state.get("retriever_override") is not None:
        # Test mode: caller provides available_sources directly via make_initial_state()
        return []
    try:
        from database import load_retriever
        r = load_retriever(k=1)
        if r is None:
            return []
        vs = getattr(r, "vectorstore", None)
        if vs is None:
            return []
        result = vs.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return sorted({
            m.get("source_filename", "")
            for m in metadatas
            if m.get("source_filename")
        })
    except Exception:
        return []


def _source_hints_for_query(query_lower: str, available: list) -> list:
    """Return the subset of available sources likely relevant to the query."""
    hints: list[str] = []
    for fname in available:
        fname_lower = fname.lower()
        for key, keywords in _SOURCE_KEYWORD_MAP.items():
            if key in fname_lower and any(kw in query_lower for kw in keywords):
                hints.append(fname)
                break
    return hints


def _validate_passages(passages: list, docs: list) -> list:
    """Return only passages whose text actually appears in retrieved docs.

    Prevents the LLM from hallucinating passage citations that were not
    in the retrieved context. Checks the first 60 chars of each passage
    against the full text of each retrieved doc.
    """
    if not passages:
        return []
    doc_texts = [d.page_content for d in docs]
    validated: list = []
    for p in passages:
        text = p.get("text", "") if isinstance(p, dict) else ""
        probe = text[:60].strip()
        if probe and any(probe in dt for dt in doc_texts):
            validated.append(p)
    return validated


def _generate_excerpt_answer(state: AgentState, docs: list) -> dict:
    """Stage 4a-style excerpt fallback used when LLM generation fails."""
    lines: list[str] = []
    for doc in docs:
        fname = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        excerpt = doc.page_content[:300].strip()
        if len(doc.page_content) > 300:
            excerpt += "…"
        lines.append(f"[{fname} — Page {page}]\n{excerpt}")

    conflict_note = ""
    if state.get("conflict_status") in (_CONFLICT_DETECTED, _CONFLICT_UNKNOWN):
        conflict_note = (
            "\n\nNote: Evidence spans multiple policy documents. "
            "Review all cited sources to verify consistency."
        )

    return {
        "answer": (
            "⚠ Answer generation was unavailable. Raw policy excerpts are shown below "
            "and have NOT been verified or synthesised by a language model.\n\n"
            + "\n\n---\n\n".join(lines)
            + conflict_note
        ),
        "sources": docs,
        "answer_type": "excerpt_fallback",
    }


# ---------------------------------------------------------------------------
# Node: classify_query
# ---------------------------------------------------------------------------

def classify_query(state: AgentState) -> dict:
    """Stage 4b: LLM-based classifier with heuristic fallback.

    Uses structured output (ClassificationOutput) via llm.with_structured_output().
    On LLM failure, falls through to _classify_query_heuristic().

    Key change from Stage 4a: questions are NEVER labelled 'unsupported' based
    on topic keywords. Answerability is determined by retrieval + evidence.
    """
    from agent.llm import get_agent_llm, ClassificationOutput
    from langchain_core.messages import HumanMessage

    _VALID_CATEGORIES = {
        "direct_factual", "multi_section", "cross_document",
        "potential_conflict", "needs_clarification",
    }

    try:
        llm = get_agent_llm(state)
        structured_llm = llm.with_structured_output(ClassificationOutput)

        prompt = (
            "Classify the following compliance question into one of these categories:\n"
            "  direct_factual — single specific fact from one policy\n"
            "  multi_section  — requires multiple sections from one policy\n"
            "  cross_document — requires evidence from more than one policy document\n"
            "  potential_conflict — may reveal contradictions between documents\n"
            "  needs_clarification — question is ambiguous\n\n"
            f"Question: {state['query']}\n\n"
            "IMPORTANT: Do NOT mark a question as unanswerable based on its topic. "
            "A question about salary, tax, or grievance may or may not be in the indexed "
            "policies — that is determined by retrieval, not by the topic itself."
        )

        result = structured_llm.invoke([HumanMessage(content=prompt)])

        if hasattr(result, "category") and result.category in _VALID_CATEGORIES:
            return {
                "classification": QueryClassification(
                    category=result.category,
                    intent=result.intent,
                    requires_cross_doc=result.requires_cross_doc,
                )
            }
    except Exception:
        pass

    return _classify_query_heuristic(state)


# ---------------------------------------------------------------------------
# Node: plan_retrieval
# ---------------------------------------------------------------------------

def plan_retrieval(state: AgentState) -> dict:
    """Stage 4b: retrieval plan using actual indexed metadata.

    Discovers available source_filename values from ChromaDB if not
    pre-populated in state["available_sources"]. Document hints are built
    from actual indexed filenames rather than hardcoded fixture names.
    """
    category = (state["classification"] or {}).get("category", "direct_factual")

    k_map = {
        "direct_factual":    _BASE_K,
        "multi_section":     _BASE_K + 1,
        "cross_document":    _BASE_K + 2,
        "potential_conflict": _BASE_K + 2,
        "needs_clarification": _BASE_K,
        "unknown":           _BASE_K,
    }
    k = k_map.get(category, _BASE_K)

    # Discover indexed sources (from ChromaDB or pre-populated in state)
    available = state.get("available_sources") or []
    if not available:
        available = _discover_indexed_sources(state)

    query_lower = state["query"].lower()

    if category in ("cross_document", "potential_conflict") and available:
        # For cross-doc queries: include all available sources as hints
        document_hints = list(available)
    else:
        document_hints = _source_hints_for_query(query_lower, available)

    return {
        "available_sources": available,
        "retrieval_plan": RetrievalPlan(
            k=k,
            filter_hints=[],
            document_hints=document_hints,
        ),
    }


# ---------------------------------------------------------------------------
# Node: retrieve_policy_chunks
# ---------------------------------------------------------------------------

def retrieve_policy_chunks(state: AgentState) -> dict:
    """Fetch chunks from ChromaDB via load_retriever().

    Uses retriever_override when set (test injection point).
    On any retrieval error, sets error and returns empty docs so
    assess_evidence can route to a graceful refusal.
    """
    k = (state["retrieval_plan"] or {}).get("k", _BASE_K)

    retriever = state.get("retriever_override")
    if retriever is None:
        try:
            from database import load_retriever
            retriever = load_retriever(k=k)
        except Exception as exc:
            return {"retrieved_docs": [], "error": f"Retriever load failed: {exc}"}

    if retriever is None:
        return {"retrieved_docs": [], "error": "No indexed documents. Upload and index PDFs first."}

    try:
        docs = retriever.invoke(state["query"])
        return {"retrieved_docs": docs or []}
    except Exception as exc:
        return {"retrieved_docs": [], "error": f"Retrieval failed: {exc}"}


# ---------------------------------------------------------------------------
# Node: assess_evidence
# ---------------------------------------------------------------------------

def assess_evidence(state: AgentState) -> dict:
    """Stage 4b: LLM-based evidence assessment with 4 outcome states.

    Outcome mapping:
      SUPPORTED          → sufficient=True,  needs_retry=False
      PARTIALLY_SUPPORTED → sufficient=True,  needs_retry=False
      INSUFFICIENT       → sufficient=False, needs_retry=True (if budget)
      CONFLICTING        → sufficient=True,  needs_retry=False (→ conflict check)
      UNAVAILABLE        → heuristic fallback determines sufficient/needs_retry

    IMPORTANT: coverage_estimate from the LLM is stored in both evidence.coverage
    and evidence.confidence. It is a rough estimate, NOT a calibrated probability.
    """
    from agent.llm import get_agent_llm, EvidenceEvaluation
    from langchain_core.messages import HumanMessage

    docs = state.get("retrieved_docs") or []

    if not docs or state.get("error"):
        needs_retry = (not state.get("error")) and (state.get("retry_count", 0) < _MAX_RETRIES)
        return {
            "evidence": EvidenceAssessment(coverage=0.0, confidence=0.0, sufficient=False),
            "evidence_status": _STATUS_INSUFFICIENT,
            "supporting_passages": [],
            "needs_retry": needs_retry,
        }

    # Build context for the LLM evaluator
    context_parts: list[str] = []
    for i, doc in enumerate(docs):
        fname = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        context_parts.append(f"[Chunk {i + 1} — {fname}, Page {page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    _VALID_STATUSES = {_STATUS_SUPPORTED, _STATUS_PARTIAL, _STATUS_INSUFFICIENT, _STATUS_CONFLICTING}

    try:
        llm = get_agent_llm(state)
        structured_llm = llm.with_structured_output(EvidenceEvaluation)

        prompt = (
            f"Question: {state['query']}\n\n"
            f"Retrieved policy chunks:\n{context}\n\n"
            "Evaluate whether these chunks provide sufficient evidence to answer the question.\n\n"
            "Assign one status:\n"
            "  SUPPORTED           — chunks directly and specifically answer the question\n"
            "  PARTIALLY_SUPPORTED — chunks partially address the question\n"
            "  INSUFFICIENT        — chunks lack the specific information needed\n"
            "  CONFLICTING         — different source documents make contradictory claims\n\n"
            "IMPORTANT constraints:\n"
            "- Evidence coverage is NOT the same as answer correctness.\n"
            "- Mark SUPPORTED only if the chunks contain the specific requested information.\n"
            "- Only cite passages that literally appear in the provided chunks.\n"
            "- Do not paraphrase or invent passage text."
        )

        result = structured_llm.invoke([HumanMessage(content=prompt)])

        status = result.status if result.status in _VALID_STATUSES else _STATUS_INSUFFICIENT
        retry_budget = state.get("retry_count", 0) < _MAX_RETRIES

        if status == _STATUS_PARTIAL and retry_budget:
            # Attempt additional retrieval before settling for a partial answer.
            sufficient = False
            needs_retry = True
        elif status in (_STATUS_SUPPORTED, _STATUS_PARTIAL, _STATUS_CONFLICTING):
            # PARTIAL with no retry budget left: proceed to generate with qualification.
            sufficient = True
            needs_retry = False
        else:  # INSUFFICIENT
            sufficient = False
            needs_retry = retry_budget

        coverage = round(float(result.coverage_estimate), 3)

        validated_passages = _validate_passages(result.supporting_passages or [], docs)

        return {
            "evidence": EvidenceAssessment(
                coverage=coverage,
                confidence=coverage,  # same value; not a distinct calibrated score
                sufficient=sufficient,
            ),
            "evidence_status": status,
            "supporting_passages": validated_passages,
            "needs_retry": needs_retry,
        }

    except Exception:
        # LLM assessment unavailable — fall back to heuristic, report UNAVAILABLE
        heuristic = _assess_evidence_heuristic(state, docs)
        return {**heuristic, "evidence_status": _STATUS_UNAVAILABLE}


# ---------------------------------------------------------------------------
# Node: expand_retrieval
# ---------------------------------------------------------------------------

def expand_retrieval(state: AgentState) -> dict:
    """Increment k and re-fetch. Increments retry_count.

    After this node LangGraph routes back to assess_evidence.
    The retry_count cap is enforced in _route_after_assessment,
    not here — this node only runs when the router allowed it.
    """
    current_plan = state.get("retrieval_plan") or {}
    new_k = current_plan.get("k", _BASE_K) + _EXPAND_K_INCREMENT
    new_plan = RetrievalPlan(
        k=new_k,
        filter_hints=current_plan.get("filter_hints", []),
        document_hints=current_plan.get("document_hints", []),
    )

    retriever = state.get("retriever_override")
    if retriever is None:
        try:
            from database import load_retriever
            retriever = load_retriever(k=new_k)
        except Exception as exc:
            return {
                "retrieval_plan": new_plan,
                "retrieved_docs": [],
                "retry_count": state.get("retry_count", 0) + 1,
                "error": f"Retriever load failed on retry: {exc}",
            }

    if retriever is None:
        return {
            "retrieval_plan": new_plan,
            "retrieved_docs": [],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        docs = retriever.invoke(state["query"])
    except Exception as exc:
        return {
            "retrieval_plan": new_plan,
            "retrieved_docs": [],
            "retry_count": state.get("retry_count", 0) + 1,
            "error": f"Retrieval failed on retry: {exc}",
        }

    return {
        "retrieval_plan": new_plan,
        "retrieved_docs": docs or [],
        "retry_count": state.get("retry_count", 0) + 1,
        "needs_retry": False,  # reset; assess_evidence will set it again
    }


# ---------------------------------------------------------------------------
# Node: check_cross_doc_conflicts
# ---------------------------------------------------------------------------

def check_cross_doc_conflicts(state: AgentState) -> dict:
    """Stage 4b: LLM-based contradiction detection across policy documents.

    Only genuine logical contradictions are flagged — not merely the presence
    of multiple documents or documents covering different topics.

    Safety-first posture: returns UNKNOWN on LLM failure, never NO_CONFLICT.
    Consequential findings are flagged for human review.
    """
    from agent.llm import get_agent_llm, ConflictAnalysis
    from langchain_core.messages import HumanMessage

    docs = state.get("retrieved_docs") or []

    sources_by_file: dict[str, list[str]] = {}
    for doc in docs:
        fname = doc.metadata.get("source_filename", "unknown")
        sources_by_file.setdefault(fname, []).append(doc.page_content)

    if len(sources_by_file) <= 1:
        return {
            "conflicts": [],
            "conflict_status": _NO_CONFLICT,
        }

    context_parts: list[str] = []
    for fname, texts in sorted(sources_by_file.items()):
        # Limit to first 2 chunks per file to stay within LLM context limits
        combined = "\n".join(texts[:2])
        context_parts.append(f"Document: {fname}\n{combined}")
    context = "\n\n---\n\n".join(context_parts)

    try:
        llm = get_agent_llm(state)
        structured_llm = llm.with_structured_output(ConflictAnalysis)

        prompt = (
            f"Question: {state['query']}\n\n"
            f"Retrieved passages from multiple policy documents:\n{context}\n\n"
            "Determine if these passages contain GENUINE contradictions — statements that "
            "directly contradict each other on the SAME specific policy topic.\n\n"
            "Do NOT flag:\n"
            "  - Documents that simply cover different topics\n"
            "  - Minor differences in phrasing about different subjects\n"
            "  - Passages from the same document\n\n"
            "DO flag: explicit logical contradictions where two policies make incompatible "
            "statements about the same rule, limit, or requirement.\n\n"
            "Set review_recommended=True for any genuine contradiction found."
        )

        result = structured_llm.invoke([HumanMessage(content=prompt)])

        if result.has_genuine_conflict:
            descs = list(result.conflict_descriptions or [])
            if result.review_recommended:
                descs = [f"[REVIEW RECOMMENDED] {d}" for d in descs]
            return {
                "conflicts": descs,
                "conflict_status": _CONFLICT_DETECTED,
            }
        else:
            return {
                "conflicts": [],
                "conflict_status": _NO_CONFLICT,
            }

    except Exception:
        # LLM conflict analysis failed — return UNKNOWN, not NO_CONFLICT
        return {
            "conflicts": [
                "Cross-document conflict analysis could not be completed. "
                "Manual review of all retrieved policy sources is recommended."
            ],
            "conflict_status": _CONFLICT_UNKNOWN,
        }


# ---------------------------------------------------------------------------
# Node: generate_grounded_answer
# ---------------------------------------------------------------------------

def generate_grounded_answer(state: AgentState) -> dict:
    """Stage 4b: LLM answer generation with citation validation.

    Uses the NVIDIA NIM LLM (same model as main.ask_question) with the
    retrieved policy chunks as context. Validates inline citations against
    actual retrieved document metadata to detect fabricated references.

    Falls back to _generate_excerpt_answer() if the LLM is unavailable.
    Returns 'refusal' when evidence is insufficient after all retries.
    """
    # Hard error from retrieval (no docs at all)
    if state.get("error") and not state.get("retrieved_docs"):
        return {"answer": _REFUSAL_MESSAGE, "sources": [], "answer_type": "error"}

    evidence = state.get("evidence")
    docs = state.get("retrieved_docs") or []
    evidence_status = state.get("evidence_status", "")

    # Refuse when evidence is genuinely insufficient (not UNAVAILABLE/PARTIAL after retries).
    if not docs or (
        evidence
        and not evidence.get("sufficient")
        and evidence_status not in (_STATUS_UNAVAILABLE, _STATUS_PARTIAL)
    ):
        return {"answer": _REFUSAL_MESSAGE, "sources": [], "answer_type": "refusal"}

    # PARTIALLY_SUPPORTED after retries exhausted: low coverage → refuse.
    if (
        evidence_status == _STATUS_PARTIAL
        and evidence
        and evidence.get("coverage", 0) < _EVIDENCE_THRESHOLD
    ):
        return {"answer": _REFUSAL_MESSAGE, "sources": [], "answer_type": "refusal"}

    # LLM assessment failed (UNAVAILABLE) and heuristic also says insufficient → refuse.
    # Heuristic keyword-overlap is not strong enough evidence to generate an answer when
    # the LLM could not confirm relevance.  Only proceed to unverified_grounded when the
    # heuristic finds positive coverage (sufficient=True).
    if (
        evidence_status == _STATUS_UNAVAILABLE
        and evidence
        and not evidence.get("sufficient")
    ):
        return {"answer": _REFUSAL_MESSAGE, "sources": [], "answer_type": "refusal"}

    from agent.llm import get_agent_llm, build_agent_context, validate_answer_citations, AGENT_SYSTEM_PROMPT
    from langchain_core.prompts import ChatPromptTemplate
    import re as _re

    context = build_agent_context(docs)
    indexed_filenames = {d.metadata.get("source_filename", "") for d in docs}

    try:
        llm = get_agent_llm(state)

        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": state["query"]})
        answer_text = response.content

        # Citation validation: strip fabricated citations (filename + page checked).
        original_citation_count = len(_re.findall(r'\[[^\]]*?\.pdf[^\]]*?\]', answer_text))
        cleaned_text, citation_warnings = validate_answer_citations(
            answer_text, indexed_filenames, docs
        )
        remaining_citations = len(_re.findall(r'\[[^\]]*?\.pdf[^\]]*?\]', cleaned_text))

        if citation_warnings:
            if remaining_citations == 0 and original_citation_count > 0:
                # All citations were fabricated — fall back to excerpts, not a bare answer.
                return _generate_excerpt_answer(state, docs)
            cleaned_text += _CITATION_STRIPPED_NOTICE

        answer_text = cleaned_text

        # Determine answer_type based on evidence quality.
        if evidence_status == _STATUS_UNAVAILABLE:
            final_type = "unverified_grounded"
            answer_text += _UNVERIFIED_ANSWER_NOTICE
        elif evidence_status == _STATUS_PARTIAL:
            final_type = "partial_grounded"
            answer_text += _PARTIAL_ANSWER_NOTICE
        else:
            final_type = "grounded"
            # Citation validation checks filename/page metadata, not claim entailment.
            answer_text += _CITATION_METADATA_NOTICE

        # Append conflict / uncertainty notice.
        conflict_status = state.get("conflict_status", "")
        if conflict_status == _CONFLICT_DETECTED:
            answer_text += (
                "\n\n⚠ Note: Potentially contradictory statements were identified "
                "across policy documents. Please review the cited sources carefully."
            )
        elif conflict_status == _CONFLICT_UNKNOWN:
            answer_text += (
                "\n\nNote: Cross-document consistency could not be fully verified. "
                "Review all cited sources."
            )

        return {"answer": answer_text, "sources": docs, "answer_type": final_type}

    except Exception:
        # LLM unavailable — return excerpts explicitly marked as unverified.
        return _generate_excerpt_answer(state, docs)


# ---------------------------------------------------------------------------
# Node: log_audit_record
# ---------------------------------------------------------------------------

def log_audit_record(state: AgentState) -> dict:
    """Record the completed query for audit purposes.

    Stage 4b: records evidence_status and conflict_status in addition to
    Stage 4a fields. Snowflake integration is deferred to Stage 4c.
    This node must never raise.
    """
    evidence = state.get("evidence") or {}
    try:
        record = {
            "query": state.get("query", ""),
            "answer_type": state.get("answer_type", ""),
            "category": (state.get("classification") or {}).get("category", "unknown"),
            "evidence_status": state.get("evidence_status", ""),
            "conflict_status": state.get("conflict_status", ""),
            "retrieved_doc_count": len(state.get("retrieved_docs") or []),
            "source_files": sorted({
                d.metadata.get("source_filename", "")
                for d in (state.get("sources") or [])
            }),
            "evidence_coverage": evidence.get("coverage", 0.0),
            "evidence_confidence": evidence.get("confidence", 0.0),
            "supporting_passages_count": len(state.get("supporting_passages") or []),
            "retry_count": state.get("retry_count", 0),
            "conflicts_flagged": len(state.get("conflicts") or []),
            "error": state.get("error"),
        }
    except Exception as exc:
        record = {"logging_error": str(exc)}

    return {"audit_metadata": record}


# ---------------------------------------------------------------------------
# Conditional edge router
# ---------------------------------------------------------------------------

def _route_after_assessment(state: AgentState) -> str:
    """Select the next node after assess_evidence completes.

    Routing priority:
      1. Retry if evidence insufficient AND retry budget remains.
      2. Conflict check if evidence is CONFLICTING (explicit signal from LLM).
      3. Conflict check if classification says cross-document query.
      4. Generate answer (includes graceful refusal if insufficient after retries).
    """
    if state.get("needs_retry") and state.get("retry_count", 0) < _MAX_RETRIES:
        return "expand_retrieval"

    evidence_status = state.get("evidence_status", "")
    if evidence_status == _STATUS_CONFLICTING:
        return "check_cross_doc_conflicts"

    if (state.get("classification") or {}).get("requires_cross_doc"):
        return "check_cross_doc_conflicts"

    return "generate_grounded_answer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile and return the Evidence-Aware Agentic RAG graph.

    Call once per process and cache the result; compilation is not free.
    Returns a CompiledStateGraph whose .invoke() accepts an AgentState dict.
    """
    g = StateGraph(AgentState)

    g.add_node("classify_query", classify_query)
    g.add_node("plan_retrieval", plan_retrieval)
    g.add_node("retrieve_policy_chunks", retrieve_policy_chunks)
    g.add_node("assess_evidence", assess_evidence)
    g.add_node("expand_retrieval", expand_retrieval)
    g.add_node("check_cross_doc_conflicts", check_cross_doc_conflicts)
    g.add_node("generate_grounded_answer", generate_grounded_answer)
    g.add_node("log_audit_record", log_audit_record)

    g.set_entry_point("classify_query")
    g.add_edge("classify_query", "plan_retrieval")
    g.add_edge("plan_retrieval", "retrieve_policy_chunks")
    g.add_edge("retrieve_policy_chunks", "assess_evidence")

    g.add_conditional_edges(
        "assess_evidence",
        _route_after_assessment,
        {
            "expand_retrieval": "expand_retrieval",
            "check_cross_doc_conflicts": "check_cross_doc_conflicts",
            "generate_grounded_answer": "generate_grounded_answer",
        },
    )

    g.add_edge("expand_retrieval", "assess_evidence")
    g.add_edge("check_cross_doc_conflicts", "generate_grounded_answer")
    g.add_edge("generate_grounded_answer", "log_audit_record")
    g.add_edge("log_audit_record", END)

    return g.compile()
