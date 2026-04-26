"""
app.py

Streamlit frontend for the AI Talent Scouting & Engagement Agent.
Provides JD input, live step progress, candidate shortlist cards,
conversation transcripts, and a Match vs Interest scatter plot.

Run with:  streamlit run app.py
"""

import os
import time
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.builder import talent_graph
from graph.state import AgentState, RankedCandidate

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Talent Scout",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.candidate-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    background: #fafafa;
}
.rank-badge {
    display: inline-block;
    background: #1f2937;
    color: white;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    line-height: 36px;
    text-align: center;
    font-weight: bold;
    font-size: 16px;
}
.action-schedule { background:#16a34a; color:white; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.action-followup { background:#d97706; color:white; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.action-pass     { background:#dc2626; color:white; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.skill-chip {
    display:inline-block;
    background:#ede9fe;
    color:#5b21b6;
    padding:2px 10px;
    border-radius:12px;
    font-size:12px;
    margin:2px;
}
.metric-big { font-size:28px; font-weight:700; color:#1f2937; }
.transcript-turn { padding:8px 12px; border-radius:8px; margin:6px 0; font-size:14px; color: white; }
.turn-recruiter { background:#1e3a8a; border-left:3px solid #3b82f6; color: white; }
.turn-candidate { background:#14532d; border-left:3px solid #16a34a; color: white; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Input panel
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_JD = """We are looking for a Senior Backend Engineer to join our fintech platform team in Hyderabad (hybrid). You will design and own microservices handling payment processing at scale.

Required: 5+ years Python, strong experience with FastAPI or Django, PostgreSQL, Redis, Docker/Kubernetes. Experience with AWS (ECS, RDS, SQS) is a must.

Nice to have: Kafka, Golang, prior fintech or payments domain experience.

We offer 25-35 LPA depending on experience. Immediate joiners preferred."""

with st.sidebar:
    st.title("🎯 AI Talent Scout")
    st.caption("LangGraph-powered candidate discovery & engagement")
    st.divider()

    jd_input = st.text_area(
        "Paste Job Description",
        value=SAMPLE_JD,
        height=280,
        placeholder="Paste the full job description here...",
    )

    st.subheader("Scoring weights")
    alpha = st.slider("Match weight (α)", min_value=0.3, max_value=0.9, value=0.6, step=0.05,
                      help="How much the technical match score counts")
    beta = round(1.0 - alpha, 2)
    st.caption(f"Interest weight (β) = **{beta}** (auto-calculated as 1 - α)")

    top_n = st.number_input("Candidates to engage", min_value=3, max_value=10, value=5,
                             help="How many top-matched candidates to run the conversation with")

    st.divider()
    run_button = st.button("🔍 Find Candidates", type="primary", use_container_width=True)

    st.divider()
    st.caption("Built with LangGraph + AI (Claude/OpenAI/Gemini)")

# ──────────────────────────────────────────────────────────────────────────────
# Main content area
# ──────────────────────────────────────────────────────────────────────────────

st.title("AI Talent Scouting & Engagement Agent")

if not run_button:
    st.info("👈 Paste a job description in the sidebar and click **Find Candidates** to start.")
    st.markdown("""
    **How it works:**
    1. 📄 **JD Parser** — AI extracts structured requirements from your JD
    2. 🔍 **Candidate Matcher** — Scores all 20 profiles with explainability
    3. 💬 **Engagement Agent** — Simulates multi-turn conversations with top candidates
    4. 📊 **Scorer & Ranker** — Combines Match + Interest into a composite rank
    5. 🏆 **Shortlist** — Recruiter-ready cards with summaries and transcripts
    """)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Run the LangGraph pipeline
# ──────────────────────────────────────────────────────────────────────────────

if not jd_input.strip():
    st.error("Please paste a job description before running.")
    st.stop()

# Check for any valid API key using our multi-LLM logic
try:
    from utils.helpers import get_llm
    llm = get_llm()  # This will raise an error if no valid key is found
    st.success(f"✅ LLM initialized: {type(llm).__name__}")
except ValueError as e:
    st.error(f"API Key Error: {str(e)}")
    st.info("💡 **Quick Fix:** Set `DEMO_MODE=true` in your `.env` file to test the interface with mock data")
    st.stop()

initial_state: AgentState = {
    "raw_jd": jd_input.strip(),
    "parsed_jd": None,
    "all_candidates": [],
    "scored_candidates": [],
    "top_candidates": [],
    "engaged_candidates": [],
    "ranked_list": [],
    "final_shortlist": [],
    "alpha": alpha,
    "beta": beta,
    "top_n": int(top_n),
    "error": None,
    "current_step": "starting",
}

start_time = time.time()
final_state = None

with st.status("Running AI Talent Scout pipeline...", expanded=True) as status_box:
    st.write("⏳ Parsing job description...")
    try:
        for step_state in talent_graph.stream(initial_state):
            node_name = list(step_state.keys())[0]
            node_output = step_state[node_name]
            current_step = node_output.get("current_step", "")

            if current_step == "jd_parsed":
                st.write("✅ Job description parsed")
            elif current_step == "candidates_matched":
                n = len(node_output.get("scored_candidates", []))
                st.write(f"✅ Matched & scored {n} candidates")
            elif current_step == "filtered":
                n = len(node_output.get("top_candidates", []))
                st.write(f"✅ Selected top {n} candidates for engagement")
            elif current_step == "engagement_complete":
                st.write("✅ Conversations simulated")
            elif current_step == "scoring_complete":
                st.write("✅ Composite scores calculated")
            elif current_step == "complete":
                st.write("✅ Shortlist formatted")
                final_state = node_output

        elapsed = round(time.time() - start_time, 1)
        status_box.update(label=f"✅ Pipeline complete in {elapsed}s", state="complete", expanded=False)

    except Exception as e:
        status_box.update(label="❌ Pipeline error", state="error")
        st.error(f"Pipeline failed: {str(e)}")
        st.stop()

# Rebuild full final state from stream
@st.cache_data(show_spinner=False)
def run_pipeline_cached(jd: str, _alpha: float, _beta: float, _top_n: int):
    """Cached full pipeline run — avoids re-running on UI interaction."""
    state: AgentState = {
        "raw_jd": jd,
        "parsed_jd": None,
        "all_candidates": [],
        "scored_candidates": [],
        "top_candidates": [],
        "engaged_candidates": [],
        "ranked_list": [],
        "final_shortlist": [],
        "alpha": _alpha,
        "beta": _beta,
        "top_n": _top_n,
        "error": None,
        "current_step": "starting",
    }
    return talent_graph.invoke(state)

# Use the streamed output we already have, but if final_state is incomplete, invoke fully
if final_state is None or not final_state.get("final_shortlist"):
    with st.spinner("Finalizing results..."):
        full_result = run_pipeline_cached(jd_input.strip(), alpha, beta, int(top_n))
else:
    full_result = run_pipeline_cached(jd_input.strip(), alpha, beta, int(top_n))

if full_result.get("error"):
    st.error(f"Error: {full_result['error']}")
    st.stop()

parsed_jd = full_result.get("parsed_jd")
shortlist: list[RankedCandidate] = full_result.get("final_shortlist", [])
scored_all = full_result.get("scored_candidates", [])

if not shortlist:
    st.warning("No candidates were shortlisted. Try adjusting the JD or top-N setting.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Parsed JD Summary
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📄 Parsed Job Requirements")

if parsed_jd:
    col1, col2, col3 = st.columns(3)
    col1.metric("Role", parsed_jd.role_title)
    col2.metric("Seniority", parsed_jd.seniority_level.capitalize())
    col3.metric("Location", parsed_jd.location)

    st.markdown("**Required skills**")
    skills_html = " ".join([f'<span class="skill-chip">{s}</span>' for s in parsed_jd.required_skills])
    st.markdown(skills_html, unsafe_allow_html=True)

    if parsed_jd.preferred_skills:
        st.markdown("**Preferred skills**")
        pref_html = " ".join([f'<span class="skill-chip" style="background:#fef9c3;color:#92400e">{s}</span>'
                               for s in parsed_jd.preferred_skills])
        st.markdown(pref_html, unsafe_allow_html=True)

    with st.expander("Full parsed JD details"):
        st.json(parsed_jd.model_dump())

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Pipeline stats
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
s1, s2, s3, s4 = st.columns(4)
s1.metric("Profiles scanned", len(scored_all))
s2.metric("Candidates engaged", len(shortlist))
s3.metric("Top composite score", f"{shortlist[0].composite_score:.1f}" if shortlist else "—")
s4.metric("Scoring weights", f"α={alpha} / β={beta}")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Scatter plot
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Match vs Interest — Candidate Map")

action_colors = {
    "Schedule interview": "#16a34a",
    "Follow up": "#d97706",
    "Pass": "#dc2626",
}

fig = go.Figure()

# Quadrant shading
fig.add_hrect(y0=60, y1=100, fillcolor="rgba(22,163,74,0.05)", line_width=0)
fig.add_vrect(x0=60, x1=100, fillcolor="rgba(22,163,74,0.05)", line_width=0)

# Quadrant lines
fig.add_hline(y=60, line_dash="dash", line_color="#d1d5db", line_width=1)
fig.add_vline(x=60, line_dash="dash", line_color="#d1d5db", line_width=1)

# Quadrant labels
for (x, y, label) in [(80, 90, "🏆 Top picks"), (35, 90, "💎 Hidden gems"),
                       (80, 30, "📋 Cold matches"), (35, 30, "⬇ Low priority")]:
    fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                       font=dict(size=11, color="#9ca3af"), xanchor="center")

for entry in shortlist:
    c = entry.candidate
    color = action_colors.get(entry.recommended_action, "#6b7280")
    fig.add_trace(go.Scatter(
        x=[c.match_score],
        y=[c.interest_score],
        mode="markers+text",
        marker=dict(size=18, color=color, line=dict(width=2, color="white")),
        text=[f"#{entry.rank}"],
        textposition="middle center",
        textfont=dict(color="white", size=10, family="Arial Black"),
        name=entry.recommended_action,
        hovertemplate=(
            f"<b>{c.name}</b><br>"
            f"Match: {c.match_score:.1f}<br>"
            f"Interest: {c.interest_score:.1f}<br>"
            f"Composite: {entry.composite_score:.1f}<br>"
            f"Action: {entry.recommended_action}<extra></extra>"
        ),
        showlegend=False,
    ))

# Legend traces
for action, color in action_colors.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=color),
        name=action, showlegend=True,
    ))

fig.update_layout(
    xaxis_title="Match Score →",
    yaxis_title="Interest Score →",
    xaxis=dict(range=[30, 105], gridcolor="#f3f4f6"),
    yaxis=dict(range=[10, 105], gridcolor="#f3f4f6"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(l=40, r=40, t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Candidate shortlist cards
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader(f"🏆 Ranked Shortlist — Top {len(shortlist)} Candidates")

for entry in shortlist:
    c = entry.candidate
    action = entry.recommended_action
    action_class = {
        "Schedule interview": "action-schedule",
        "Follow up": "action-followup",
        "Pass": "action-pass",
    }.get(action, "action-pass")

    with st.container():
        # Header row
        hcol1, hcol2, hcol3 = st.columns([0.5, 4, 2])
        with hcol1:
            st.markdown(f'<div class="rank-badge">#{entry.rank}</div>', unsafe_allow_html=True)
        with hcol2:
            st.markdown(f"**{c.name}** &nbsp; `{c.profile.get('current_title', '')}` &nbsp; · &nbsp; {c.profile.get('years_experience', '')} yrs &nbsp; · &nbsp; {c.profile.get('location', '')}")
            recent = ", ".join(c.profile.get("recent_companies", []))
            st.caption(f"📍 Recent: {recent} &nbsp;|&nbsp; 💰 {c.comp_expectation or 'Not specified'} &nbsp;|&nbsp; ⏱ {c.availability or 'Unknown'}")
        with hcol3:
            st.markdown(f'<span class="{action_class}">{action}</span>', unsafe_allow_html=True)
            st.markdown(f"**Composite: {entry.composite_score:.1f}**")

        # Score bars
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown(f"Match Score: **{c.match_score:.1f}**")
            st.progress(int(c.match_score) / 100)
        with scol2:
            st.markdown(f"Interest Score: **{c.interest_score:.1f}**")
            st.progress(int(c.interest_score) / 100)

        # Summary
        st.markdown(f"_{entry.recruiter_summary}_")

        # Reasons & gaps
        rcol, gcol = st.columns(2)
        with rcol:
            st.markdown("**✅ Match reasons**")
            for r in c.match_reasons:
                st.markdown(f"- {r}")
        with gcol:
            st.markdown("**⚠️ Gaps**")
            for g in c.match_gaps:
                st.markdown(f"- {g}")

        # Conversation transcript (expandable)
        with st.expander(f"💬 View conversation transcript — {c.name}"):
            for turn in c.conversation_transcript:
                role = turn.get("role", "")
                content = turn.get("content", "")
                css_class = "turn-recruiter" if role == "recruiter" else "turn-candidate"
                label = "🧑‍💼 Recruiter" if role == "recruiter" else f"👤 {c.name}"
                st.markdown(
                    f'<div class="transcript-turn {css_class}"><strong>{label}</strong><br>{content}</div>',
                    unsafe_allow_html=True,
                )

        # Interest signals
        with st.expander(f"📈 Interest signals — {c.name}"):
            for sig in c.interest_signals:
                st.markdown(f"- {sig}")

        st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — All scored candidates (expandable table)
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("📋 View all 20 scored candidates (match scores)"):
    rows = []
    for sc in scored_all:
        rows.append({
            "Name": sc.name,
            "Title": sc.profile.get("current_title", ""),
            "Match Score": round(sc.match_score, 1),
            "Location": sc.profile.get("location", ""),
            "Availability": sc.profile.get("availability", ""),
            "Comp Expectation": sc.profile.get("compensation_expectation", ""),
        })
    st.dataframe(rows, use_container_width=True)

st.caption(f"Pipeline completed · {len(scored_all)} profiles evaluated · Top {len(shortlist)} engaged · Weights α={alpha} / β={beta}")