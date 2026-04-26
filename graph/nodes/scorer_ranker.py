"""
graph/nodes/scorer_ranker.py

LangGraph node that computes composite scores and ranks all engaged candidates.
Pure Python — no LLM call needed here.
"""

from graph.state import AgentState, EngagedCandidate, RankedCandidate
from utils.helpers import compute_composite_score, determine_recommended_action


def scorer_ranker_node(state: AgentState) -> dict:
    """
    LangGraph node: rank engaged candidates by weighted composite score.

    composite_score = alpha * match_score + beta * interest_score

    Reads:  state['engaged_candidates'], state['alpha'], state['beta']
    Writes: state['ranked_list'], state['current_step']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    alpha = state.get("alpha", 0.6)
    beta = state.get("beta", 0.4)

    ranked: list[RankedCandidate] = []

    for candidate in state["engaged_candidates"]:
        composite = compute_composite_score(
            candidate.match_score, candidate.interest_score, alpha, beta
        )
        action = determine_recommended_action(composite)
        ranked.append(RankedCandidate(
            rank=0,  # assigned after sorting
            composite_score=composite,
            candidate=candidate,
            recruiter_summary="",  # filled by shortlist_formatter
            recommended_action=action,
        ))

    # Sort descending and assign ranks
    ranked.sort(key=lambda r: r.composite_score, reverse=True)
    for i, entry in enumerate(ranked):
        entry.rank = i + 1

    return {
        "ranked_list": ranked,
        "current_step": "scoring_complete",
    }
