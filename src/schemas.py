from typing import List

from pydantic import BaseModel, Field


class HealthSummary(BaseModel):
    vehicle_id: str = Field(description="Unique vehicle identifier")
    risk_level: str = Field(description="Low, Medium, High, or Critical")
    key_issues: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ActionItem(BaseModel):
    vehicle_id: str
    service_action: str
    timeline: str
    priority: str
    rationale: str


class FleetRecommendation(BaseModel):
    problem_understanding: str
    health_summary: List[HealthSummary]
    action_plan: List[ActionItem]
    sources: List[str]
    safety_disclaimer: str
