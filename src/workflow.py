import os
import json
from typing import Dict, List

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.rag import GuidelineRetriever
from src.risk import analyze_risks
from src.schemas import FleetRecommendation
from src.state import FleetState


def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    if provider != "groq":
        return None

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return None
    return ChatGroq(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
    )


def analyze_node(state: FleetState) -> FleetState:
    vehicles = state.get("vehicles", [])
    risks = analyze_risks(vehicles)
    trace = state.get("reasoning_trace", []) + ["Completed deterministic risk scoring."]
    return {"risk_analysis": risks, "reasoning_trace": trace}


def retrieve_node(state: FleetState) -> FleetState:
    query = state.get("maintenance_query", "General preventive maintenance.")
    retriever = GuidelineRetriever()
    guidelines = retriever.retrieve(query)
    trace = state.get("reasoning_trace", []) + [f"Retrieved {len(guidelines)} guideline chunks."]
    return {"retrieved_guidelines": guidelines, "reasoning_trace": trace}


def _fallback_recommendation(risks: List[Dict], guidelines: List[str], query: str) -> Dict:
    actions: List[Dict] = []
    for r in risks:
        timeline = "Within 7 days"
        if r["risk_level"] == "Critical":
            timeline = "Immediate (24 hours)"
        elif r["risk_level"] == "High":
            timeline = "Within 72 hours"
        elif r["risk_level"] == "Medium":
            timeline = "Within 14 days"

        actions.append(
            {
                "vehicle_id": r["vehicle_id"],
                "service_action": "Perform targeted maintenance for identified issues",
                "timeline": timeline,
                "priority": r["risk_level"],
                "rationale": "; ".join(r["key_issues"]),
            }
        )

    return {
        "problem_understanding": f"Addressing fleet maintenance request: {query}",
        "health_summary": risks,
        "action_plan": actions,
        "sources": guidelines[:4],
        "safety_disclaimer": "Recommendations are AI-assisted. Final maintenance decisions must be validated by certified technicians.",
    }


def _ensure_all_vehicles_present(risks: List[Dict], recommendation: Dict, guidelines: List[str], query: str) -> Dict:
    risk_by_id = {str(r["vehicle_id"]): r for r in risks}
    expected_ids = [str(r["vehicle_id"]) for r in risks]

    health_summary = recommendation.get("health_summary", []) or []
    action_plan = recommendation.get("action_plan", []) or []
    health_by_id = {str(h.get("vehicle_id", "")): h for h in health_summary}
    action_by_id = {str(a.get("vehicle_id", "")): a for a in action_plan}

    for vehicle_id in expected_ids:
        if vehicle_id not in health_by_id:
            health_by_id[vehicle_id] = risk_by_id[vehicle_id]
        if vehicle_id not in action_by_id:
            risk_level = risk_by_id[vehicle_id]["risk_level"]
            timeline = "Within 7 days"
            if risk_level == "Critical":
                timeline = "Immediate (24 hours)"
            elif risk_level == "High":
                timeline = "Within 72 hours"
            elif risk_level == "Medium":
                timeline = "Within 14 days"

            action_by_id[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "service_action": "Perform targeted maintenance for identified issues",
                "timeline": timeline,
                "priority": risk_level,
                "rationale": "; ".join(risk_by_id[vehicle_id].get("key_issues", [])),
            }

    recommendation["problem_understanding"] = recommendation.get(
        "problem_understanding", f"Addressing fleet maintenance request: {query}"
    )
    recommendation["health_summary"] = [health_by_id[vid] for vid in expected_ids]
    recommendation["action_plan"] = [action_by_id[vid] for vid in expected_ids]
    recommendation["sources"] = recommendation.get("sources", guidelines[:4])[:4]
    recommendation["safety_disclaimer"] = recommendation.get(
        "safety_disclaimer",
        "Recommendations are AI-assisted. Final maintenance decisions must be validated by certified technicians.",
    )
    return recommendation


def recommend_node(state: FleetState) -> FleetState:
    risks = state.get("risk_analysis", [])
    guidelines = state.get("retrieved_guidelines", [])
    query = state.get("maintenance_query", "")

    llm = _get_llm()
    if llm is None:
        recommendation = _fallback_recommendation(risks, guidelines, query)
        trace = state.get("reasoning_trace", []) + ["Used fallback recommendation path (no valid Groq API key)."]
        return {
            "final_recommendation": recommendation,
            "reasoning_trace": trace,
            "llm_status": "fallback_no_valid_groq_key",
        }

    prompt = f"""
You are a Fleet Management Agent.
Use only evidence from risk analysis and retrieved guidelines.
If evidence is weak, provide conservative service actions and say uncertainty clearly.
Return ONLY valid JSON (no markdown fences) matching this schema:
{{
  "problem_understanding": "string",
  "health_summary": [
    {{"vehicle_id": "string", "risk_level": "Low|Medium|High|Critical", "key_issues": ["string"], "confidence": 0.0}}
  ],
  "action_plan": [
    {{"vehicle_id": "string", "service_action": "string", "timeline": "string", "priority": "string", "rationale": "string"}}
  ],
  "sources": ["string"],
  "safety_disclaimer": "string"
}}

Fleet Query:
{query}

Risk Analysis:
{risks}

Retrieved Guidelines:
{guidelines}
"""

    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        output = json_llm.invoke(prompt)
        text = output.content if hasattr(output, "content") else str(output)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        recommendation = FleetRecommendation.model_validate(json.loads(cleaned)).model_dump()
        recommendation = _ensure_all_vehicles_present(risks, recommendation, guidelines, query)
        trace = state.get("reasoning_trace", []) + ["Generated structured recommendation with Groq LLM."]
        return {
            "final_recommendation": recommendation,
            "reasoning_trace": trace,
            "llm_status": "groq_call_success",
            "llm_error": "",
        }
    except Exception as exc:
        recommendation = _fallback_recommendation(risks, guidelines, query)
        err = f"{type(exc).__name__}: {str(exc)}"
        trace = state.get("reasoning_trace", []) + [
            f"Groq request failed ({err[:220]}); returned deterministic fallback recommendation.",
        ]
        return {
            "final_recommendation": recommendation,
            "reasoning_trace": trace,
            "llm_status": "fallback_groq_request_failed",
            "llm_error": err[:500],
        }


def build_graph():
    graph = StateGraph(FleetState)
    graph.add_node("analyze_risks", analyze_node)
    graph.add_node("retrieve_guidelines", retrieve_node)
    graph.add_node("generate_recommendation", recommend_node)

    graph.set_entry_point("analyze_risks")
    graph.add_edge("analyze_risks", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "generate_recommendation")
    graph.add_edge("generate_recommendation", END)

    return graph.compile()
