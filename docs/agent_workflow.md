# Agent Workflow Documentation

## Input -> Output Specification

### Input
- Fleet CSV with fields:
  - `vehicle_id`
  - `mileage`
  - `engine_temperature`
  - `oil_quality`
  - `brake_wear`
  - `battery_health`
  - `days_since_last_service`
- Maintenance query from user.

### Output (Structured)
- `problem_understanding`
- `health_summary` (risk level, key issues, confidence per vehicle)
- `action_plan` (service action, timeline, priority, rationale)
- `sources` (retrieved guideline snippets)
- `safety_disclaimer`

## LangGraph State

The workflow uses explicit `FleetState`:
- `vehicles`
- `maintenance_query`
- `risk_analysis`
- `retrieved_guidelines`
- `reasoning_trace`
- `final_recommendation`

## Workflow Steps

1. **Risk Analyzer Node**
   - Deterministic scoring for reliability and reproducibility.
2. **RAG Retrieval Node**
   - Retrieves relevant maintenance references from Chroma vector DB.
3. **Recommendation Node**
   - Uses structured output with Pydantic schema.
   - Fallback mode works without API key.

## Hallucination Mitigation

- Retrieval-grounded responses.
- Explicit safety disclaimers.
- Conservative fallback policy when confidence is limited.
- Structured output to reduce free-form unsupported claims.
