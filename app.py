import json

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.workflow import build_graph

load_dotenv()

st.set_page_config(page_title="Agentic Data Analysis", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background: #0b0f19; color: #e6edf3; }
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        .main-card {
            background: rgba(19, 27, 44, 0.86); border: 1px solid #27334d; border-radius: 16px;
            padding: 1.1rem 1.2rem; margin-bottom: 1rem;
        }
        .title-glow { color: #dbe8ff; font-weight: 700; letter-spacing: 0.4px; margin-bottom: 0.2rem; }
        .subtitle { color: #8ca4d6; margin-bottom: 1rem; font-size: 0.95rem; }
        .center-wrap { max-width: 920px; margin: 0 auto; }
        .small-note { color: #9bb0d8; font-size: 0.84rem; margin-top: 0.35rem; }
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #4b6bff 0%, #3954d6 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
st.markdown('<h2 class="title-glow">Agentic Data Analysis</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze fleet telemetry and generate AI-guided maintenance recommendations.</div>',
    unsafe_allow_html=True,
)

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

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("#### Input Parameters")
num_rows = st.slider("Number of Vehicles", min_value=1, max_value=20, value=4)
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
fleet_df = st.data_editor(
    fleet_df,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "vehicle_id": st.column_config.TextColumn("Vehicle ID", required=True),
        "mileage": st.column_config.NumberColumn("Mileage", min_value=0, step=500),
        "engine_temperature": st.column_config.NumberColumn("Engine Temp", min_value=0, step=1),
        "oil_quality": st.column_config.NumberColumn("Oil Quality", min_value=0, max_value=100, step=1),
        "brake_wear": st.column_config.NumberColumn("Brake Wear", min_value=0, max_value=100, step=1),
        "battery_health": st.column_config.NumberColumn("Battery Health", min_value=0, max_value=100, step=1),
        "days_since_last_service": st.column_config.NumberColumn("Days Since Last Service", min_value=0, step=1),
    },
)
query = st.text_area(
    "Analysis Goal",
    value="Prioritize upcoming service actions for safety-critical vehicles.",
    height=80,
)
st.markdown('<div class="small-note">Tip: edit rows directly and click Analyze to run the full workflow.</div>', unsafe_allow_html=True)
run = st.button("Analyze Fleet Data")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if run:
    vehicles = fleet_df.to_dict(orient="records")
    if not vehicles:
        st.error("Please add at least one vehicle row before running analysis.")
        st.stop()

    with st.spinner("Running agent workflow..."):
        graph = build_graph()
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
        if not health_df.empty and "risk_level" in health_df.columns:
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
