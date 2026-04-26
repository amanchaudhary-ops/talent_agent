"""
tools/mcp_client.py

Mock MCP (Model Context Protocol) client that simulates what a real
LinkedIn MCP or ATS MCP server would provide.

HOW TO SWAP IN A REAL MCP SERVER:
----------------------------------
1. Install the MCP SDK:  pip install mcp
2. Replace MockProfileMCP with a real MCP client connection:

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@your-org/ats-mcp-server"],
        env={"ATS_API_KEY": os.getenv("ATS_API_KEY")}
    )

3. Call tools via:  session.call_tool("search_candidates", {...})
4. Map the JSON response to the same dict schema used here.

The node code (candidate_matcher.py) calls MockProfileMCP and does not
need to change — only this file needs updating for production.
"""

from tools.profile_store import get_all_candidates, get_candidate_by_id


class MockProfileMCP:
    """
    Simulates an MCP server for candidate profile retrieval.
    Wraps the local profile store with a tool-call-style API.

    In production, replace the method bodies with real MCP tool calls
    to a LinkedIn, Greenhouse, Lever, or custom ATS MCP server.
    """

    def search_candidates(
        self,
        skills: list[str] | None = None,
        location: str | None = None,
        seniority: str | None = None,
        min_years: int = 0,
    ) -> list[dict]:
        """
        Search candidates by filters. Simulates an MCP 'search_candidates' tool call.

        Args:
            skills: List of skill keywords to filter by (any match).
            location: Preferred city; also returns remote-open candidates.
            seniority: One of 'junior', 'mid', 'senior', 'lead'.
            min_years: Minimum years of experience.

        Returns:
            Filtered list of candidate profile dicts.
        """
        candidates = get_all_candidates()

        if min_years:
            candidates = [c for c in candidates if c["years_experience"] >= min_years]

        if location:
            candidates = [
                c for c in candidates
                if location.lower() in c["location"].lower() or c["remote_open"]
            ]

        if skills:
            skills_lower = [s.lower() for s in skills]
            candidates = [
                c for c in candidates
                if any(s.lower() in [sk.lower() for sk in c["skills"]] for s in skills_lower)
            ]

        if seniority:
            seniority_map = {
                "junior": (0, 2),
                "mid": (2, 5),
                "senior": (5, 9),
                "lead": (7, 99),
                "principal": (10, 99),
            }
            yr_range = seniority_map.get(seniority.lower(), (0, 99))
            candidates = [
                c for c in candidates
                if yr_range[0] <= c["years_experience"] <= yr_range[1]
            ]

        return candidates

    def get_candidate_profile(self, candidate_id: str) -> dict | None:
        """
        Retrieve a full candidate profile by ID. Simulates MCP 'get_profile' tool.

        Args:
            candidate_id: The candidate's unique ID string.

        Returns:
            Full profile dict or None if not found.
        """
        return get_candidate_by_id(candidate_id)

    def list_all(self) -> list[dict]:
        """
        Returns the full candidate pool. Simulates MCP 'list_candidates' tool.

        Returns:
            All candidate profiles.
        """
        return get_all_candidates()