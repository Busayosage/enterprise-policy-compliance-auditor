"""
LLM client factory and structured-output Pydantic schemas for Stage 4b.

All schemas are designed for use with llm.with_structured_output(Schema).
Each has documented fallback behaviour when the LLM call fails.

Pydantic v2 style models (pydantic==2.x installed).
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------

class ClassificationOutput(BaseModel):
    """Structured output for classify_query node (Stage 4b)."""
    category: str = Field(
        description=(
            "Query category: direct_factual | multi_section | cross_document | "
            "potential_conflict | needs_clarification"
        )
    )
    intent: str = Field(description="One sentence describing what the query asks.")
    requires_cross_doc: bool = Field(
        description=(
            "True if the answer likely requires evidence from multiple policy documents."
        )
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the question is ambiguous and would benefit from clarification.",
    )
    answerability: str = Field(
        default="possibly_answerable",
        description=(
            "answerable | possibly_answerable | unclear. "
            "Never set to 'unanswerable' based on topic keywords alone — "
            "answerability is determined through retrieval and evidence assessment."
        ),
    )


class EvidenceEvaluation(BaseModel):
    """Structured output for assess_evidence node (Stage 4b).

    IMPORTANT contract: supporting_passages must contain ONLY text that
    literally appears in the provided retrieved chunks — never paraphrased
    or invented content. coverage_estimate is a rough heuristic, NOT a
    calibrated probability or proof of answer correctness.
    """
    status: str = Field(
        description=(
            "SUPPORTED: passages directly and specifically answer the query. "
            "PARTIALLY_SUPPORTED: passages partially address the query. "
            "INSUFFICIENT: passages lack the needed specific information. "
            "CONFLICTING: passages from different sources make contradictory claims."
        )
    )
    reasoning: str = Field(
        description="One or two sentences explaining the assigned status."
    )
    supporting_passages: list = Field(
        default_factory=list,
        description=(
            "Each item: {text: str, source: str, page: int}. "
            "ONLY cite text that literally appears in the provided chunks. "
            "Never paraphrase or invent passage content."
        ),
    )
    coverage_estimate: float = Field(
        default=0.5,
        description=(
            "0.0–1.0 rough estimate of how much the retrieved chunks cover the query. "
            "This is NOT a calibrated probability and must not be presented as one."
        ),
        ge=0.0,
        le=1.0,
    )


class ConflictAnalysis(BaseModel):
    """Structured output for check_cross_doc_conflicts node (Stage 4b).

    Only genuine logical contradictions should be flagged — not mere topical
    differences between documents. On LLM failure the caller returns UNKNOWN,
    never NO_CONFLICT, to preserve the safety-first posture.
    """
    has_genuine_conflict: bool = Field(
        description=(
            "True ONLY if different documents make directly contradictory statements "
            "about the same specific policy topic. False if documents cover different "
            "topics or if the apparent difference is minor/contextual."
        )
    )
    conflict_descriptions: list = Field(
        default_factory=list,
        description="Human-readable description of each genuine contradiction (list[str]).",
    )
    supporting_passages: list = Field(
        default_factory=list,
        description=(
            "Passages from each side of the contradiction, with source metadata. "
            "Format: [{text: str, source: str, page: int, side: str}]."
        ),
    )
    review_recommended: bool = Field(
        default=False,
        description="True if a human reviewer should examine these findings.",
    )


# ---------------------------------------------------------------------------
# Agent-specific LLM configuration and context helpers
# ---------------------------------------------------------------------------

_AGENT_MODEL = "meta/llama-3.1-70b-instruct"
_AGENT_MAX_TOKENS = 500
_MAX_CONTEXT_CHARS = 8000

AGENT_SYSTEM_PROMPT = """You are a strict corporate compliance auditor. Your role is to answer
compliance questions using ONLY the policy document context provided below.

Rules:
- Base your answer strictly on the provided context.
- When citing a specific clause, include the source document name and page number in brackets,
  for example: [it_security_policy.pdf, Page 3].
- Only cite documents that actually appear in the context below.
- If the context does not contain enough information to answer, respond with:
  "I cannot find enough evidence in the uploaded policy documents to answer that."
- Do not speculate or use outside knowledge.
- Do not fabricate quotations or citations.

Context:
{context}"""


def build_agent_context(docs: list) -> str:
    """Build LLM context string from retrieved chunks.

    Mirrors main._build_context(): includes complete chunks up to
    _MAX_CONTEXT_CHARS; chunks are never truncated mid-sentence.
    Page numbers are 1-indexed (PyPDF stores 0-indexed).
    """
    chunk_texts: list[str] = []
    char_count = 0
    sep = "\n\n---\n\n"

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        header = f"[{doc.metadata.get('source_filename', 'Unknown')} — Page {page_num}]"
        chunk = f"{header}\n{doc.page_content}"
        sep_cost = len(sep) if chunk_texts else 0

        if chunk_texts and char_count + sep_cost + len(chunk) > _MAX_CONTEXT_CHARS:
            break

        chunk_texts.append(chunk)
        char_count += sep_cost + len(chunk)

    return sep.join(chunk_texts)


def validate_answer_citations(answer: str, valid_filenames: set, docs: list = None) -> tuple:
    """Strip PDF citations whose filename or page cannot be matched in the retrieved set.

    Returns (cleaned_answer, warnings) where:
    - cleaned_answer has unmatched citations physically removed from the text.
    - warnings lists each removed citation for audit purposes.

    Validation scope — metadata only:
      This function verifies that cited filenames exist in the retrieved document
      set and that cited page numbers match a retrieved chunk for that file.  It
      does NOT verify that the cited passage entails the generated claim, nor that
      the answer is factually accurate.  Passing validation means the citation
      refers to a document that was actually retrieved; it is not a guarantee of
      factual support.

    Validation layers (applied in order, strictest-first):
    1. Filename: must appear in valid_filenames — else strip.
    2. Page number: if parseable AND docs provided, the page must exist in a
       retrieved chunk from that file — else strip.

    Only [*.pdf ...] citation blocks are examined; other text is left untouched.
    """
    pattern = r'\[([^\]]*?\.pdf[^\]]*?)\]'

    # Build (filename, 1-indexed-page) pairs from retrieved docs for page validation.
    valid_file_pages: set = set()
    if docs:
        for doc in docs:
            fname = doc.metadata.get("source_filename", "")
            page = doc.metadata.get("page", 0) + 1  # stored 0-indexed → display 1-indexed
            if fname:
                valid_file_pages.add((fname, page))

    warnings: list[str] = []

    def _check(match: re.Match) -> str:
        block = match.group(1)
        fname_part = block.split(",")[0].strip()

        if not fname_part:
            return match.group(0)  # no filename found — leave unchanged

        # Layer 1: filename must be in the retrieved set.
        if fname_part not in valid_filenames:
            warnings.append(
                f"Removed citation '{fname_part}': document not in retrieved set"
            )
            return ""

        # Layer 2: page number, when present, must match a retrieved chunk.
        if valid_file_pages:
            page_m = re.search(r'[Pp]age\s+(\d+)', block)
            if page_m:
                page_num = int(page_m.group(1))
                if (fname_part, page_num) not in valid_file_pages:
                    warnings.append(
                        f"Removed citation '{fname_part}, Page {page_num}': "
                        "page not present in retrieved chunks"
                    )
                    return ""

        return match.group(0)  # citation verified — keep as-is

    cleaned = re.sub(pattern, _check, answer)
    # Collapse whitespace left behind by removed citation blocks.
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned, warnings


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_agent_llm(state: dict):
    """Return an LLM instance for use in agent nodes.

    Resolution order:
      1. state["llm_override"] — test injection point (mock LLM).
      2. ChatNVIDIA built from the NVIDIA_API_KEY environment variable.

    Raises ValueError if neither is available.
    Does NOT make any network calls on its own.
    """
    override = state.get("llm_override")
    if override is not None:
        return override

    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to .env or pass llm_override for testing."
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    return ChatNVIDIA(
        model=_AGENT_MODEL,
        nvidia_api_key=api_key,
        temperature=0,
        max_tokens=_AGENT_MAX_TOKENS,
        timeout=30,
    )
