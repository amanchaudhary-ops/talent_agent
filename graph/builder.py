"""
graph/builder.py

Assembles and compiles the LangGraph StateGraph for the talent scouting pipeline.

Graph topology:
  START
    └─► jd_parser
          └─► [conditional: route_after_parsing]
                ├─► candidate_matcher
                │     └─► filter_top_n
                │           └─► engagement_agent
                │                 └─► scorer_ranker
                │                       └─► shortlist_formatter
                │                             └─► END
                └─► END  (on parse error)
"""

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes.jd_parser import jd_parser_node
from graph.nodes.candidate_matcher import candidate_matcher_node
from graph.nodes.engagement_agent import engagement_agent_node
from graph.nodes.scorer_ranker import scorer_ranker_node
from graph.nodes.shortlist_formatter import shortlist_formatter_node
from graph.edges import filter_top_n_node, route_after_parsing


def build_graph():
    """
    Build and compile the talent scouting StateGraph.

    Returns:
        Compiled LangGraph runnable ready for .invoke() or .stream().
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("jd_parser", jd_parser_node)
    graph.add_node("candidate_matcher", candidate_matcher_node)
    graph.add_node("filter_top_n", filter_top_n_node)
    graph.add_node("engagement_agent", engagement_agent_node)
    graph.add_node("scorer_ranker", scorer_ranker_node)
    graph.add_node("shortlist_formatter", shortlist_formatter_node)

    # Entry point
    graph.set_entry_point("jd_parser")

    # Conditional edge after JD parsing
    graph.add_conditional_edges(
        "jd_parser",
        route_after_parsing,
        {
            "candidate_matcher": "candidate_matcher",
            "end": END,
        },
    )

    # Linear pipeline after matching
    graph.add_edge("candidate_matcher", "filter_top_n")
    graph.add_edge("filter_top_n", "engagement_agent")
    graph.add_edge("engagement_agent", "scorer_ranker")
    graph.add_edge("scorer_ranker", "shortlist_formatter")
    graph.add_edge("shortlist_formatter", END)

    return graph.compile()


# Singleton compiled graph — import this in app.py
talent_graph = build_graph()