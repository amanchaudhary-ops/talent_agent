"""
graph/nodes/shortlist_formatter.py

LangGraph node that generates a recruiter-facing summary for each ranked candidate.
Makes one LLM call per candidate to produce a 2-3 sentence briefing.
"""

import os
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.state import AgentState, RankedCandidate
from prompts.templates import SHORTLIST_SUMMARY_PROMPT
from utils.helpers import get_llm


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _generate_summary(llm: Any, entry: RankedCandidate, role_title: str) -> str:
    """
    Generate a recruiter summary for one ranked candidate.

    Args:
        llm: Configured LLM instance.
        entry: RankedCandidate with scores and signals.
        role_title: The role being hired for.

    Returns:
        2-3 sentence recruiter summary string.
    """
    c = entry.candidate
    chain = SHORTLIST_SUMMARY_PROMPT | llm
    response = chain.invoke({
        "role_title": role_title,
        "name": c.name,
        "current_title": c.profile.get("current_title", c.name),
        "years_experience": c.profile.get("years_experience", "N/A"),
        "match_score": round(c.match_score),
        "interest_score": round(c.interest_score),
        "composite_score": round(entry.composite_score),
        "recommended_action": entry.recommended_action,
        "match_reasons": "; ".join(c.match_reasons[:2]),
        "match_gaps": "; ".join(c.match_gaps[:2]),
        "interest_signals": "; ".join(c.interest_signals[:2]),
        "availability": c.availability or "Not specified",
        "comp_expectation": c.comp_expectation or "Not specified",
    })
    return response.content.strip()


def shortlist_formatter_node(state: AgentState) -> dict:
    """
    LangGraph node: generate recruiter summaries for all ranked candidates.

    Reads:  state['ranked_list'], state['parsed_jd']
    Writes: state['final_shortlist'], state['current_step']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    llm = get_llm(temperature=0.3)
    role_title = state["parsed_jd"].role_title
    final: list[RankedCandidate] = []

    for entry in state["ranked_list"]:
        try:
            summary = _generate_summary(llm, entry, role_title)
        except Exception:
            summary = (
                f"{entry.candidate.name} scored {round(entry.composite_score)}/100 overall. "
                f"Match: {round(entry.candidate.match_score)}, Interest: {round(entry.candidate.interest_score)}. "
                f"Recommended action: {entry.recommended_action}."
            )
        final.append(RankedCandidate(
            rank=entry.rank,
            composite_score=entry.composite_score,
            candidate=entry.candidate,
            recruiter_summary=summary,
            recommended_action=entry.recommended_action,
        ))

    return {
        "final_shortlist": final,
        "current_step": "complete",
    }