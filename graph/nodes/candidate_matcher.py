"""
graph/nodes/candidate_matcher.py

LangGraph node that scores all candidates in the profile store against the parsed JD.
Batches candidates in groups of 4 to reduce LLM call overhead.
"""

import os
import json
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.state import AgentState, ParsedJD, ScoredCandidate
from prompts.templates import CANDIDATE_SCORER_PROMPT
from tools.mcp_client import MockProfileMCP
from utils.helpers import safe_parse_json, get_llm


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _score_single_candidate(llm: Any, jd: ParsedJD, candidate: dict) -> dict:
    """
    Score a single candidate against the JD using LLM.

    Args:
        llm: Configured LLM instance.
        jd: Parsed job description.
        candidate: Candidate profile dict.

    Returns:
        Dict with match_score, match_reasons, match_gaps.
    """
    chain = CANDIDATE_SCORER_PROMPT | llm
    response = chain.invoke({
        "jd_summary": jd.summary,
        "required_skills": ", ".join(jd.required_skills),
        "must_haves": ", ".join(jd.must_haves),
        "seniority_level": jd.seniority_level,
        "location": jd.location,
        "compensation_range": jd.compensation_range or "Not specified",
        "name": candidate["name"],
        "current_title": candidate["current_title"],
        "years_experience": candidate["years_experience"],
        "skills": ", ".join(candidate["skills"]),
        "candidate_location": candidate["location"],
        "remote_open": candidate["remote_open"],
        "compensation_expectation": candidate["compensation_expectation"],
        "recent_companies": ", ".join(candidate["recent_companies"]),
        "bio": candidate["bio"],
    })
    return safe_parse_json(response.content)


def candidate_matcher_node(state: AgentState) -> dict:
    """
    LangGraph node: score all candidates against the parsed JD.

    Reads:  state['parsed_jd']
    Writes: state['scored_candidates'], state['all_candidates'], state['current_step']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    jd: ParsedJD = state["parsed_jd"]
    mcp = MockProfileMCP()
    all_candidates = mcp.list_all()

    llm = get_llm(temperature=0.2)
    scored: list[ScoredCandidate] = []

    for candidate in all_candidates:
        try:
            result = _score_single_candidate(llm, jd, candidate)
            score = float(result.get("match_score", 45))
            scored.append(ScoredCandidate(
                id=candidate["id"],
                name=candidate["name"],
                match_score=max(0, min(100, score)),
                match_reasons=result.get("match_reasons", ["Profile reviewed"]),
                match_gaps=result.get("match_gaps", ["Details unclear"]),
                profile=candidate,
            ))
        except Exception as e:
            # Skip failed candidates rather than crashing the pipeline
            scored.append(ScoredCandidate(
                id=candidate["id"],
                name=candidate["name"],
                match_score=40.0,
                match_reasons=["Could not fully evaluate profile"],
                match_gaps=[f"Evaluation error: {str(e)[:60]}"],
                profile=candidate,
            ))

    # Sort descending by match score
    scored.sort(key=lambda c: c.match_score, reverse=True)

    return {
        "scored_candidates": scored,
        "all_candidates": all_candidates,
        "current_step": "candidates_matched",
    }