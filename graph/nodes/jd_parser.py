"""
graph/nodes/jd_parser.py

LangGraph node that parses a raw job description into a structured ParsedJD object.
Uses LLM with JSON output for compatibility with all providers (Claude, OpenAI, Gemini).
"""

import os
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.state import AgentState, ParsedJD
from prompts.templates import JD_PARSER_PROMPT
from utils.helpers import get_llm


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def _parse_jd(raw_jd: str) -> ParsedJD:
    """
    Call LLM to parse the JD with JSON output (compatible with all providers).

    Args:
        raw_jd: Raw job description text.

    Returns:
        ParsedJD Pydantic model instance.
    """
    llm = get_llm(temperature=0.1)
    
    # Create a JSON-focused prompt for Gemini compatibility
    json_prompt = f"""Parse this job description and return ONLY a valid JSON object with these exact fields:

{{
    "role_title": "string",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "seniority_level": "junior|mid|senior|lead|principal",
    "role_type": "full-time|contract|remote|hybrid",
    "location": "city or remote",
    "compensation_range": "range or null",
    "must_haves": ["requirement1", "requirement2"],
    "nice_to_haves": ["bonus1", "bonus2"],
    "summary": "one paragraph description"
}}

Job Description:
{raw_jd}

Return ONLY the JSON object, no other text:"""
    
    try:
        # Try structured output first (works with Claude/OpenAI)
        if hasattr(llm, 'with_structured_output'):
            structured_llm = llm.with_structured_output(ParsedJD)
            chain = JD_PARSER_PROMPT | structured_llm
            return chain.invoke({"raw_jd": raw_jd})
        else:
            # Fallback for Gemini: use regular text generation + JSON parsing
            response = llm.invoke(json_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the JSON response
            parsed_data = safe_parse_json(response_text)
            
            if not parsed_data:
                raise ValueError(f"Failed to parse JSON from LLM response: {response_text[:200]}")
            
            # Validate required fields and provide defaults
            validated_data = {
                "role_title": parsed_data.get("role_title", "Software Engineer"),
                "required_skills": parsed_data.get("required_skills", []),
                "preferred_skills": parsed_data.get("preferred_skills", []),
                "seniority_level": parsed_data.get("seniority_level", "mid"),
                "role_type": parsed_data.get("role_type", "full-time"),
                "location": parsed_data.get("location", "remote"),
                "compensation_range": parsed_data.get("compensation_range"),
                "must_haves": parsed_data.get("must_haves", []),
                "nice_to_haves": parsed_data.get("nice_to_haves", []),
                "summary": parsed_data.get("summary", "A software engineering position")
            }
            
            return ParsedJD(**validated_data)
            
    except Exception as e:
        if "authentication" in str(e).lower() or "auth" in str(e).lower():
            raise ValueError(
                f"Authentication failed with your API key. "
                f"Please check that your API key is valid and not expired. "
                f"Error: {str(e)}"
            )
        raise


def jd_parser_node(state: AgentState) -> dict:
    """
    LangGraph node: parse raw JD text into a structured ParsedJD.

    Reads:  state['raw_jd']
    Writes: state['parsed_jd'], state['current_step'], state['error']

    Args:
        state: Current AgentState.

    Returns:
        Partial state update dict.
    """
    try:
        parsed = _parse_jd(state["raw_jd"])
        return {
            "parsed_jd": parsed,
            "current_step": "jd_parsed",
            "error": None,
        }
    except Exception as e:
        return {
            "parsed_jd": None,
            "current_step": "error",
            "error": f"JD parsing failed: {str(e)}",
        }