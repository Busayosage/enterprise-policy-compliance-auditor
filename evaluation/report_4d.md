# Stage 4d: Comparative Evaluation Report

**Generated:** 2026-10-30T00:04:34.581977+00:00  
**Embedding model:** sentence-transformers/all-MiniLM-L6-v2  
**Fixture PDFs:** it_security_policy.pdf, hr_leave_policy.pdf, finance_expense_policy.pdf  
**Indexing time:** 3.27s  

## Evaluation Scope and Methodology

### What was measured (offline, deterministic)
- Source-document hit rate: retriever returns ≥1 chunk from the expected file
- Evidence-clause coverage: all expected key terms present in top-k chunks
- Cross-document completeness: both expected files retrieved for cross-doc questions
- Citation validation: fabricated citations detected and stripped
- Difficult scenarios: retry, conflict, credential failure, fallback paths
- Agentic refusal accuracy (mocked INSUFFICIENT LLM)

### What was NOT measured (requires live NVIDIA inference)
- Answer text correctness / factual accuracy
- Standard RAG refusal accuracy (D01–D06)
- Token usage and actual response latency
- Real NVIDIA API error rates

---

## Comparison Table

| Metric | Standard RAG | Agentic RAG | Notes |
|--------|-------------|-------------|-------|
| Source-doc hit rate | 100.0% | 100.0% | Retriever only |
| Key-term coverage | 100.0% | 100.0% | Retriever only |
| Cross-doc complete rate | 100.0% | 100.0% | Both files retrieved |
| Refusal accuracy | n/a (live only) | 100.0% (mocked) | D01–D06 |
| Citation validation | n/a | 100.0% | 5 cases |
| Difficult scenarios | n/a | 6/6 passed | See below |
| LLM calls per question (live) | 1 | 3–6 (classify + assess + [conflict] + generate) | Overhead |
| Dataset questions | 38 | 38 | Same corpus |

---

## Standard RAG — Retrieval Metrics

Questions: 38 total (32 answerable, 6 should-refuse)  
k = 3

### Per-category results

| Category | n | Hit rate | Key-term rate |
|----------|---|----------|---------------|
| direct_factual | 10 | 100.0% | 100.0% |
| multi_section | 8 | 100.0% | 100.0% |
| cross_document | 4 | 100.0% | 100.0% |
| unsupported | 6 | 0.0% | 0.0% |
| conflicting_or_edge | 10 | 100.0% | 100.0% |

### Failing questions (hit miss or key-term miss)

_No failures — all answerable questions retrieve from correct source and cover key terms._

---

## Agentic RAG — Retrieval and Pipeline Metrics

Questions: 38 total (32 answerable, 6 should-refuse)  
Base k = 3, max k after retry = 6  
LLM: mocked — answer text not evaluated

### Per-category results

| Category | n | Hit rate | Key-term rate | Avg retries | Answer types |
|----------|---|----------|---------------|-------------|--------------|
| direct_factual | 10 | 100.0% | 100.0% | 0.0 | grounded:10 |
| multi_section | 8 | 100.0% | 100.0% | 0.0 | grounded:8 |
| cross_document | 4 | 100.0% | 100.0% | 0.0 | grounded:4 |
| unsupported | 6 | 0.0% | 0.0% | 2.0 | refusal:6 |
| conflicting_or_edge | 10 | 100.0% | 100.0% | 0.0 | grounded:10 |

### Answer type distribution (mocked LLM)

| Answer type | Count | % |
|-------------|-------|---|
| grounded | 32 | 84.2% |
| refusal | 6 | 15.8% |

---

## Citation Validation

Pass rate: 5/5 (100.0%)  
Fully deterministic — no LLM required.

| ID | Description | Expected stripped | Actual stripped | Pass |
|----|-------------|------------------|-----------------|------|
| CIT01 | All citations valid | 0 | 0 | ✓ |
| CIT02 | One fabricated filename | 1 | 1 | ✓ |
| CIT03 | Wrong page number | 1 | 1 | ✓ |
| CIT04 | All citations fabricated | 2 | 2 | ✓ |
| CIT05 | No citations in answer | 0 | 0 | ✓ |

---

## Difficult Scenario Tests

Pass rate: 6/6 (100.0%)  

| Scenario | Passed | Notes |
|----------|--------|-------|
| partial_supported_triggers_retry | ✓ | PARTIALLY_SUPPORTED must trigger at least 1 retry |
| llm_assessment_failure_produces_unverified | ✓ | LLM failure during evidence assessment must not produce 'grounded' |
| missing_api_key_raises_value_error | ✓ | get_agent_llm() must raise ValueError when key absent; ChatNVIDIA never constructed |
| all_citations_fabricated_produces_fallback | ✓ | All fabricated citations must trigger excerpt_fallback, not 'grounded' |
| insufficient_evidence_produces_refusal | ✓ | INSUFFICIENT evidence must produce 'refusal' answer_type |
| conflicting_evidence_detected_and_reported | ✓ | CONFLICTING evidence must produce conflict_status=CONFLICT_DETECTED with descriptions |

---

## Limitations and Honest Caveats

1. **No live LLM calls were made.** All answer-type results are from mocked LLMs.
   Hit rates and coverage rates are retrieval metrics, not answer-accuracy metrics.
2. **Standard RAG refusal accuracy (D01–D06) is not measured offline.**
   The retriever always returns k results; refusal depends on the LLM following its
   system prompt.
3. **Agentic refusal accuracy is measured with a mocked INSUFFICIENT LLM.**
   Real accuracy may differ once the live NVIDIA model is used.
4. **Agentic LLM overhead:** ~3–6 LLM calls per question (classify + assess +
   [conflict check] + generate). Standard RAG uses 1 call. This trades latency
   and cost for structured evidence verification.
5. **Hit rates are identical between Standard RAG and Agentic RAG** when using the
   same retriever and k. The agentic pipeline adds retry logic (k expansion) that
   can improve coverage on PARTIALLY_SUPPORTED queries.
6. **Conflicting policy detection only fires for cross-document queries.**
   Single-document questions never reach the conflict-check node.

## Stage 4e Recommendation

Stage 4e should address:
- **Live evaluation run**: A single approved NVIDIA API call session to measure
  answer accuracy, refusal precision and recall on D01–D06, and actual latency.
- **Latency profiling**: Measure real wall-clock time for both modes including
  cold-start embedding model load.
- **Token accounting**: Count tokens per question for both modes to estimate
  production cost per query.
- **k-ablation study**: Evaluate whether k=5 or k=7 improves multi-section
  coverage for category B and C questions.
- **Threshold tuning**: Calibrate PARTIALLY_SUPPORTED retry trigger vs.
  coverage threshold to reduce unnecessary retries on category A questions.
