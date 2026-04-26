"""
prompts/templates.py

All LLM prompt templates used across the agent pipeline.
Centralised here so nodes stay clean and prompts can be tuned independently.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Stage 1 — JD Parser
# ---------------------------------------------------------------------------

JD_PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical recruiter and JD analyst.
Extract structured information from the job description provided.
Be precise — if a field is not mentioned, use a sensible default or empty list.
For seniority_level use exactly one of: junior / mid / senior / lead / principal.
For role_type use: full-time / contract / remote / hybrid.
Return only the structured data requested, no additional commentary."""),
    ("human", """Parse this job description and extract all structured fields:

{raw_jd}

Extract:
- role_title
- required_skills (list)
- preferred_skills (list)
- seniority_level (junior/mid/senior/lead/principal)
- role_type (full-time/contract/remote/hybrid)
- location (city or "remote")
- compensation_range (or null if not mentioned)
- must_haves (non-skill hard requirements)
- nice_to_haves (bonus qualifications)
- summary (one clear paragraph describing the role and ideal candidate)"""),
])

# ---------------------------------------------------------------------------
# Stage 2 — Candidate Scorer
# ---------------------------------------------------------------------------

CANDIDATE_SCORER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior technical recruiter scoring candidate profiles against a job description.
Score each candidate honestly and produce a meaningful spread — do NOT cluster scores around 70.
Use the full 40-95 range based on actual fit.

Scoring rubric:
- 85-100: Exceptional match — hits all required skills, right seniority, domain experience
- 70-84: Strong match — hits most required skills, minor gaps only
- 55-69: Partial match — has core skills but notable gaps (wrong domain, missing key tech)
- 40-54: Weak match — has some transferable skills but significant misalignment
- Below 40: Poor match — do not score below 40 (those get filtered out pre-screening)

Be specific in match_reasons and match_gaps — reference actual skills and experience from the profile."""),
    ("human", """Score this candidate against the job description.

JOB DESCRIPTION SUMMARY:
{jd_summary}

REQUIRED SKILLS: {required_skills}
MUST HAVES: {must_haves}
SENIORITY NEEDED: {seniority_level}
LOCATION: {location}
COMPENSATION RANGE: {compensation_range}

CANDIDATE PROFILE:
Name: {name}
Title: {current_title}
Years Experience: {years_experience}
Skills: {skills}
Location: {candidate_location}
Remote Open: {remote_open}
Comp Expectation: {compensation_expectation}
Recent Companies: {recent_companies}
Bio: {bio}

Return a JSON object with exactly these fields:
{{
  "match_score": <integer 40-100>,
  "match_reasons": ["<specific reason 1>", "<specific reason 2>", "<specific reason 3>"],
  "match_gaps": ["<specific gap 1>", "<specific gap 2>"]
}}"""),
])

# ---------------------------------------------------------------------------
# Stage 3a — Recruiter Outreach (first message)
# ---------------------------------------------------------------------------

RECRUITER_OUTREACH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a thoughtful, personable technical recruiter writing the first outreach message to a candidate.
Write naturally — not a template blast. Reference specifics from their background.
Keep it concise (3-4 sentences), warm, and clear about the opportunity.
Do NOT use phrases like "I hope this message finds you well" or "exciting opportunity"."""),
    ("human", """Write a personalized outreach message to this candidate for this role.

ROLE: {role_title} at a fintech company in {location}
JD SUMMARY: {jd_summary}
COMPENSATION: {compensation_range}

CANDIDATE:
Name: {name}
Current Title: {current_title}
Recent Companies: {recent_companies}
Key Skills: {skills}
Bio: {bio}

Write only the message text, no subject line or signature."""),
])

# ---------------------------------------------------------------------------
# Stage 3b — Candidate Persona (simulates candidate reply)
# ---------------------------------------------------------------------------

CANDIDATE_PERSONA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are simulating how a real candidate would respond to a recruiter message.
Reply authentically based on the candidate's profile, availability, and compensation expectations.

Availability signal guide:
- "immediate" → enthusiastic, proactive, asks follow-up questions
- "2 weeks" → interested but measured, asks about the role details
- "1 month" → cautious, asks about team and culture before committing
- "not looking" → polite but clearly not interested, gives short non-committal answers

Match the candidate's seniority in writing style. Senior engineers write concisely; juniors are more eager.
Keep responses realistic — 2-5 sentences typically."""),
    ("human", """You are {name}, a {current_title} with {years_experience} years of experience.
Your background: {bio}
Your availability: {availability}
Your compensation expectation: {compensation_expectation}
Your location: {candidate_location}, Remote open: {remote_open}

Conversation so far:
{conversation_history}

Recruiter's latest message:
{recruiter_message}

Reply as {name}. Be authentic to your availability and interest level."""),
])

# ---------------------------------------------------------------------------
# Stage 3c — Interest Analyzer
# ---------------------------------------------------------------------------

INTEREST_ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are analyzing a recruiter-candidate conversation to assess the candidate's genuine interest level.
Look for: enthusiasm in language, asking follow-up questions, sharing specific details, mentioning timeline eagerness vs hedging.

Interest score guide:
- 80-100: Highly interested — proactive, asks questions, clear excitement, flexible on details
- 60-79: Moderately interested — engaged but cautious, wants more info, non-committal
- 40-59: Low interest — polite but passive, short answers, deflecting
- 20-39: Not interested — clearly not looking, dismissive or very short
- 0-19: Disinterested — explicitly said no or unresponsive

Extract concrete signals from the actual conversation text, not generic statements."""),
    ("human", """Analyze this recruiter-candidate conversation and score the candidate's interest.

CANDIDATE PROFILE:
Name: {name}
Availability: {availability}
Compensation expectation: {compensation_expectation}

FULL CONVERSATION TRANSCRIPT:
{transcript}

Return a JSON object with exactly these fields:
{{
  "interest_score": <integer 0-100>,
  "interest_signals": ["<specific signal from conversation 1>", "<signal 2>", "<signal 3>"],
  "availability": "<extracted availability string>",
  "comp_expectation": "<extracted compensation expectation or null>"
}}"""),
])

# ---------------------------------------------------------------------------
# Stage 5 — Shortlist Summary
# ---------------------------------------------------------------------------

SHORTLIST_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are writing a concise recruiter briefing for each shortlisted candidate.
Write 2-3 sentences. Be direct and specific — mention their strongest match point,
their interest level, and the recommended action with a brief reason.
Write for a busy recruiter who will scan this in 5 seconds."""),
    ("human", """Write a recruiter summary for this candidate.

ROLE: {role_title}
CANDIDATE: {name} — {current_title}, {years_experience} yrs exp
MATCH SCORE: {match_score}/100
INTEREST SCORE: {interest_score}/100
COMPOSITE SCORE: {composite_score}/100
RECOMMENDED ACTION: {recommended_action}

TOP MATCH REASONS: {match_reasons}
KEY GAPS: {match_gaps}
INTEREST SIGNALS: {interest_signals}
AVAILABILITY: {availability}
COMP EXPECTATION: {comp_expectation}

Write only the summary paragraph, 2-3 sentences."""),
])