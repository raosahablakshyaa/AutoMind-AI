import json

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.workflow import build_graph

load_dotenv()

st.set_page_config(page_title="FleetAI Assistant", page_icon="🚚", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #0f1117 0%, #161b22 100%); color: #e6edf3; }
        .main-card {
            background: rgba(33, 38, 45, 0.7); border: 1px solid #30363d; border-radius: 16px;
            padding: 1.2rem; margin-bottom: 1rem;
        }
        .title-glow { color: #58a6ff; font-weight: 700; letter-spacing: 0.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title-glow">FleetAI Agentic Maintenance Assistant</h1>', unsafe_allow_html=True)
st.caption("Autonomous fleet risk reasoning + RAG-grounded service planning")

with st.sidebar:
    st.header("Input Actions")
    num_rows = st.slider("Number of vehicles", min_value=1, max_value=20, value=4)
    query = st.text_area(
        "Maintenance query",
        value="Prioritize upcoming service actions for safety-critical vehicles.",
        height=120,
    )
    run = st.button("Run Agent Workflow", width="stretch")

base_rows = [
    {
        "vehicle_id": "TRK-101",
        "mileage": 120000,
        "engine_temperature": 102,
        "oil_quality": 40,
        "brake_wear": 72,
        "battery_health": 68,
        "days_since_last_service": 165,
    },
    {
        "vehicle_id": "TRK-203",
        "mileage": 175000,
        "engine_temperature": 108,
        "oil_quality": 30,
        "brake_wear": 88,
        "battery_health": 42,
        "days_since_last_service": 240,
    },
    {
        "vehicle_id": "VAN-008",
        "mileage": 62000,
        "engine_temperature": 91,
        "oil_quality": 76,
        "brake_wear": 35,
        "battery_health": 81,
        "days_since_last_service": 90,
    },
    {
        "vehicle_id": "CAR-550",
        "mileage": 98000,
        "engine_temperature": 97,
        "oil_quality": 52,
        "brake_wear": 67,
        "battery_health": 47,
        "days_since_last_service": 181,
    },
]

while len(base_rows) < num_rows:
    idx = len(base_rows) + 1
    base_rows.append(
        {
            "vehicle_id": f"VH-{idx:03}",
            "mileage": 0,
            "engine_temperature": 90,
            "oil_quality": 70,
            "brake_wear": 20,
            "battery_health": 85,
            "days_since_last_service": 30,
        }
    )

fleet_df = pd.DataFrame(base_rows[:num_rows])

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Fleet Dataset (Entry-Based Input)")
fleet_df = st.data_editor(
    fleet_df,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "vehicle_id": st.column_config.TextColumn("Vehicle ID", required=True),
        "mileage": st.column_config.NumberColumn("Mileage", min_value=0),
        "engine_temperature": st.column_config.NumberColumn("Engine Temp (C)", min_value=0),
        "oil_quality": st.column_config.NumberColumn("Oil Quality (%)", min_value=0, max_value=100),
        "brake_wear": st.column_config.NumberColumn("Brake Wear (%)", min_value=0, max_value=100),
        "battery_health": st.column_config.NumberColumn("Battery Health (%)", min_value=0, max_value=100),
        "days_since_last_service": st.column_config.NumberColumn("Days Since Last Service", min_value=0),
    },
)
st.markdown("</div>", unsafe_allow_html=True)

if run:
    graph = build_graph()
    vehicles = fleet_df.to_dict(orient="records")
    result = graph.invoke(
        {
            "fleet_context": "Mixed urban logistics fleet",
            "maintenance_query": query,
            "vehicles": vehicles,
            "reasoning_trace": [],
        }
    )

    rec = result.get("final_recommendation", {})
    llm_status = result.get("llm_status", "unknown")
    llm_error = result.get("llm_error", "")
    health_df = pd.DataFrame(rec.get("health_summary", []))
    action_df = pd.DataFrame(rec.get("action_plan", []))

    status_labels = {
        "groq_call_success": "Groq API call: Success",
        "fallback_no_valid_groq_key": "Groq API call: Skipped (missing/placeholder key)",
        "fallback_groq_request_failed": "Groq API call: Failed (fallback used)",
    }
    status_text = status_labels.get(llm_status, f"Groq API call: {llm_status}")
    if llm_status == "groq_call_success":
        st.success(status_text)
    elif llm_status == "fallback_no_valid_groq_key":
        st.warning(status_text)
    else:
        st.error(status_text)
        if llm_error:
            st.caption(f"Groq error detail: {llm_error}")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Health Summary")
        st.dataframe(health_df, width="stretch")
        if not health_df.empty:
            fig = px.histogram(health_df, x="risk_level", color="risk_level", title="Risk Level Distribution")
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Action Plan")
        st.dataframe(action_df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Problem Understanding")
    st.write(rec.get("problem_understanding", "No summary produced"))
    st.subheader("Retrieved Sources")
    for source in rec.get("sources", []):
        st.markdown(f"- {source}")
    st.subheader("Operational Disclaimer")
    st.warning(
        rec.get(
            "safety_disclaimer",
            "AI output is advisory only. Certified maintenance professionals must approve final actions.",
        )
    )
    with st.expander("Agent Reasoning Trace"):
        for step in result.get("reasoning_trace", []):
            st.markdown(f"- {step}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download Structured Report (JSON)",
        data=json.dumps(rec, indent=2),
        file_name="fleet_recommendation_report.json",
        mime="application/json",
        width="stretch",
    )
