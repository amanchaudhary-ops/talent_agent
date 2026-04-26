"""
graph/state.py

Defines the central AgentState TypedDict that flows through every LangGraph node,
along with all Pydantic models used for structured LLM outputs.
"""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field


class ParsedJD(BaseModel):
    """Structured representation of a parsed job description."""

    role_title: str = Field(description="The job title being hired for")
    required_skills: list[str] = Field(description="Non-negotiable technical skills")
    preferred_skills: list[str] = Field(description="Nice-to-have skills")
    seniority_level: str = Field(description="junior / mid / senior / lead / principal")
    role_type: str = Field(description="full-time / contract / remote / hybrid")
    location: str = Field(description="City or remote")
    compensation_range: Optional[str] = Field(default=None, description="Salary range if mentioned")
    must_haves: list[str] = Field(description="Hard requirements beyond skills")
    nice_to_haves: list[str] = Field(description="Soft or bonus requirements")
    summary: str = Field(description="One-paragraph plain-English summary of the role")


class ScoredCandidate(BaseModel):
    """A candidate profile with a match score against the JD."""

    id: str
    name: str
    match_score: float = Field(ge=0, le=100, description="Match quality 0-100")
    match_reasons: list[str] = Field(description="2-4 specific reasons why they match")
    match_gaps: list[str] = Field(description="1-3 specific gaps or concerns")
    profile: dict = Field(description="Original candidate profile dict")


class EngagedCandidate(BaseModel):
    """A candidate who has gone through the conversational engagement simulation."""

    id: str
    name: str
    match_score: float
    interest_score: float = Field(ge=0, le=100, description="Interest level 0-100")
    interest_signals: list[str] = Field(description="3-5 signals extracted from the conversation")
    conversation_transcript: list[dict] = Field(description="List of {role, content} turn dicts")
    availability: Optional[str] = Field(default=None)
    comp_expectation: Optional[str] = Field(default=None)
    match_reasons: list[str]
    match_gaps: list[str]
    profile: dict


class RankedCandidate(BaseModel):
    """Final ranked output entry, ready for the recruiter UI."""

    rank: int
    composite_score: float = Field(description="Weighted combination of match + interest")
    candidate: EngagedCandidate
    recruiter_summary: str = Field(description="2-3 sentence summary for the recruiter")
    recommended_action: str = Field(description="Schedule interview / Follow up / Pass")


class AgentState(TypedDict):
    """
    Central state object that flows through every node in the LangGraph pipeline.
    Each node reads the fields it needs and writes back its output fields.
    """

    raw_jd: str
    parsed_jd: Optional[ParsedJD]
    all_candidates: list[dict]
    scored_candidates: list[ScoredCandidate]
    top_candidates: list[ScoredCandidate]
    engaged_candidates: list[EngagedCandidate]
    ranked_list: list[RankedCandidate]
    final_shortlist: list[RankedCandidate]
    alpha: float          # match score weight, default 0.6
    beta: float           # interest score weight, default 0.4
    top_n: int            # how many candidates to engage
    error: Optional[str]
    current_step: str