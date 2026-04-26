"""
graph/edges.py

Conditional edge functions and filter nodes for the LangGraph pipeline.
"""

from graph.state import AgentState, ScoredCandidate


def filter_top_n_node(state: AgentState) -> dict:
    """
    LangGraph node wrapping the top-N filter logic.

    Filters scored candidates to the top N with match_score >= 40,
    ensuring the engagement agent only runs on qualified candidates.

    Reads:  state['scored_candidates'], state['top_n']
    Writes: state['top_candidates'], state['current_step']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    top_n: int = state.get("top_n", 5)
    scored: list[ScoredCandidate] = state["scored_candidates"]

    # Filter to minimum viable match score
    qualified = [c for c in scored if c.match_score >= 40]

    # Take top N
    top = qualified[:top_n]

    return {
        "top_candidates": top,
        "current_step": "filtered",
    }


def route_after_parsing(state: AgentState) -> str:
    """
    Conditional routing function after the jd_parser node.

    Returns:
        'candidate_matcher' if parsing succeeded.
        'end' if parsing failed (error state).
    """
    if state.get("error") or state.get("parsed_jd") is None:
        return "end"
    return "candidate_matcher"