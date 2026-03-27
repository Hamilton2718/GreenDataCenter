"""
LangGraph Studio entrypoint for GreenDataCenter.

Exports a compiled graph object named `graph` so LangGraph Studio/CLI
can discover and run the workflow.
"""

from graph import create_datacenter_agent_system
from nodes.requirement_analysis_node import requirement_analysis_node
from nodes.energy_planner_node import energy_planner_node
from nodes.cooling_specialist_node import cooling_specialist_node
from nodes.simulation_node import simulation_node
from nodes.review_node import review_node
from nodes.financial_consultant_node import financial_consultant_node
from nodes.final_report_node import final_report_node


def build_graph():
    """Build and return the compiled LangGraph application."""
    return create_datacenter_agent_system(
        requirement_analysis_node=requirement_analysis_node,
        energy_planner_node=energy_planner_node,
        cooling_specialist_node=cooling_specialist_node,
        simulation_node=simulation_node,
        review_node=review_node,
        financial_consultant_node=financial_consultant_node,
        final_report_node=final_report_node,
    )


# LangGraph Studio looks for this export.
graph = build_graph()
