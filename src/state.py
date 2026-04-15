from typing import Dict, List, TypedDict


class FleetState(TypedDict, total=False):
    fleet_context: str
    maintenance_query: str
    vehicles: List[Dict]
    risk_analysis: List[Dict]
    retrieved_guidelines: List[str]
    reasoning_trace: List[str]
    final_recommendation: Dict
    disclaimer: str
    llm_status: str
    llm_error: str
