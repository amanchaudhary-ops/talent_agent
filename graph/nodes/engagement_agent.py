"""
graph/nodes/engagement_agent.py

LangGraph node that simulates a multi-turn recruiter-candidate conversation
for each top candidate and produces an Interest Score with explainability.

Conversation flow per candidate:
  Turn 1  — Recruiter: personalized outreach
  Turn 2  — Candidate: initial response (simulated)
  Turn 3  — Recruiter: asks about availability, compensation, motivation
  Turn 4  — Candidate: responds with specifics
  Turn 5  — Recruiter: asks about biggest motivation for a move
  Turn 6  — Candidate: final response
  Analysis — LLM analyzes transcript → interest_score + signals
"""

import os
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.state import AgentState, ScoredCandidate, EngagedCandidate
from prompts.templates import (
    RECRUITER_OUTREACH_PROMPT,
    CANDIDATE_PERSONA_PROMPT,
    INTEREST_ANALYZER_PROMPT,
)
from utils.helpers import safe_parse_json, format_transcript_for_prompt, get_llm


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _generate_recruiter_outreach(llm: Any, jd_info: dict, candidate: dict) -> str:
    """Generate the recruiter's first outreach message."""
    chain = RECRUITER_OUTREACH_PROMPT | llm
    response = chain.invoke({
        "role_title": jd_info["role_title"],
        "location": jd_info["location"],
        "jd_summary": jd_info["summary"],
        "compensation_range": jd_info.get("compensation_range") or "Competitive",
        "name": candidate["name"],
        "current_title": candidate["current_title"],
        "recent_companies": ", ".join(candidate["recent_companies"]),
        "skills": ", ".join(candidate["skills"][:5]),
        "bio": candidate["bio"],
    })
    return response.content.strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _generate_candidate_reply(llm: Any, candidate: dict, conversation_history: list[dict], recruiter_message: str) -> str:
    """Simulate the candidate's reply based on their profile and persona."""
    history_str = format_transcript_for_prompt(conversation_history)
    chain = CANDIDATE_PERSONA_PROMPT | llm
    response = chain.invoke({
        "name": candidate["name"],
        "current_title": candidate["current_title"],
        "years_experience": candidate["years_experience"],
        "bio": candidate["bio"],
        "availability": candidate["availability"],
        "compensation_expectation": candidate["compensation_expectation"],
        "candidate_location": candidate["location"],
        "remote_open": candidate["remote_open"],
        "conversation_history": history_str,
        "recruiter_message": recruiter_message,
    })
    return response.content.strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _analyze_interest(llm: Any, candidate: dict, transcript: list[dict]) -> dict:
    """Analyze the full conversation transcript to produce interest score and signals."""
    transcript_str = format_transcript_for_prompt(transcript)
    chain = INTEREST_ANALYZER_PROMPT | llm
    response = chain.invoke({
        "name": candidate["name"],
        "availability": candidate["availability"],
        "compensation_expectation": candidate["compensation_expectation"],
        "transcript": transcript_str,
    })
    return safe_parse_json(response.content)


def _run_conversation(jd_info: dict, candidate: dict) -> tuple[list[dict], dict]:
    """
    Run the full 6-turn recruiter-candidate conversation for one candidate.

    Args:
        jd_info: Dict with role info extracted from ParsedJD.
        candidate: Candidate profile dict.

    Returns:
        Tuple of (transcript list, interest analysis dict).
    """
    llm_recruiter = get_llm(temperature=0.3)
    llm_candidate = get_llm(temperature=0.6)
    llm_analyst = get_llm(temperature=0.1)

    transcript: list[dict] = []

    # Turn 1 — Recruiter outreach
    outreach = _generate_recruiter_outreach(llm_recruiter, jd_info, candidate)
    transcript.append({"role": "recruiter", "content": outreach})

    # Turn 2 — Candidate initial response
    reply1 = _generate_candidate_reply(llm_candidate, candidate, [], outreach)
    transcript.append({"role": "candidate", "content": reply1})

    # Turn 3 — Recruiter follow-up: role details, availability, compensation
    followup_msg = (
        f"Great to hear from you! The role is focused on building payment processing "
        f"microservices at scale. The team is growing and this would be a senior IC position "
        f"with real ownership. We're looking at {jd_info.get('compensation_range', 'competitive compensation')}. "
        f"Would that compensation range work for you? And what's your current notice period?"
    )
    transcript.append({"role": "recruiter", "content": followup_msg})

    # Turn 4 — Candidate responds to specifics
    reply2 = _generate_candidate_reply(llm_candidate, candidate, transcript[:-1], followup_msg)
    transcript.append({"role": "candidate", "content": reply2})

    # Turn 5 — Recruiter: motivation for a move
    motivation_msg = (
        "That's helpful context. One more thing I'd love to understand — what's the "
        "primary thing that would make you consider a new role right now? Is it the "
        "technical challenge, the team, growth trajectory, or something else?"
    )
    transcript.append({"role": "recruiter", "content": motivation_msg})

    # Turn 6 — Candidate final response
    reply3 = _generate_candidate_reply(llm_candidate, candidate, transcript[:-1], motivation_msg)
    transcript.append({"role": "candidate", "content": reply3})

    # Analyze the full transcript
    analysis = _analyze_interest(llm_analyst, candidate, transcript)
    return transcript, analysis


def engagement_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: run simulated conversations with top candidates.

    Reads:  state['top_candidates'], state['parsed_jd']
    Writes: state['engaged_candidates'], state['current_step']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    jd = state["parsed_jd"]
    jd_info = {
        "role_title": jd.role_title,
        "location": jd.location,
        "summary": jd.summary,
        "compensation_range": jd.compensation_range,
    }

    engaged: list[EngagedCandidate] = []

    for scored_candidate in state["top_candidates"]:
        candidate = scored_candidate.profile
        try:
            transcript, analysis = _run_conversation(jd_info, candidate)
            interest_score = float(analysis.get("interest_score", 50))

            engaged.append(EngagedCandidate(
                id=scored_candidate.id,
                name=scored_candidate.name,
                match_score=scored_candidate.match_score,
                interest_score=max(0, min(100, interest_score)),
                interest_signals=analysis.get("interest_signals", ["Conversation completed"]),
                conversation_transcript=transcript,
                availability=analysis.get("availability") or candidate.get("availability"),
                comp_expectation=analysis.get("comp_expectation") or candidate.get("compensation_expectation"),
                match_reasons=scored_candidate.match_reasons,
                match_gaps=scored_candidate.match_gaps,
                profile=candidate,
            ))
        except Exception as e:
            # If engagement fails for one candidate, use defaults and continue
            engaged.append(EngagedCandidate(
                id=scored_candidate.id,
                name=scored_candidate.name,
                match_score=scored_candidate.match_score,
                interest_score=50.0,
                interest_signals=["Engagement simulation encountered an error"],
                conversation_transcript=[{"role": "system", "content": f"Error: {str(e)[:100]}"}],
                availability=candidate.get("availability", "unknown"),
                comp_expectation=candidate.get("compensation_expectation"),
                match_reasons=scored_candidate.match_reasons,
                match_gaps=scored_candidate.match_gaps,
                profile=candidate,
            ))

    return {
        "engaged_candidates": engaged,
        "current_step": "engagement_complete",
    }