"""
LangGraph-based Evidence-Aware Agentic RAG — Stage 4a foundation.

Public surface:
    build_graph()  → CompiledStateGraph (import and compile once per process)
    run_agent()    → invoke the compiled graph for a single question
    AgentState     → typed state dict threaded through the graph
    make_initial_state() → helper to create a clean starting state
"""

from agent.state import AgentState, make_initial_state
from agent.graph import build_graph

__all__ = ["AgentState", "make_initial_state", "build_graph", "run_agent"]


def run_agent(
    question: str,
    *,
    retriever=None,
    graph=None,
) -> AgentState:
    """Run the agentic RAG pipeline for one question.

    Args:
        question:  The compliance question to answer.
        retriever: Optional pre-built LangChain retriever. When None the graph
                   will call database.load_retriever() at runtime (production).
                   Pass a stub retriever in tests to avoid live ChromaDB.
        graph:     Optional pre-compiled graph (avoids recompiling in tests).
                   When None a fresh graph is compiled for this call.

    Returns:
        The final AgentState after the graph has reached END.
    """
    if graph is None:
        graph = build_graph()

    initial = make_initial_state(question, retriever=retriever)
    result = graph.invoke(initial)
    return result
