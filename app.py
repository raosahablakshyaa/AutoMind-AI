"""
Fleet Maintenance Assistant - Streamlit App

A fleet maintenance dashboard for ML-driven predictive maintenance,
AI recommendations, editable fleet data, and RAG-style guideline retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Fleet Maintenance Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #080d18;
        color: #e2e8f0;
    }
    .st-bz {
        background-color: #080d18;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 15px;
        border-radius: 10px;
        color: #e2e8f0;
    }
    .risk-critical {
        background: #FF6B6B;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-high {
        background: #FFA500;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-medium {
        background: #FFC75F;
        color: #1f2937;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-low {
        background: #6BCB77;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .rec-card {
        background-color: #111827;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #1f2937;
        border-left: 4px solid #F59E0B;
    }
    .rec-card.custom-critical { border-left-color: #B91C1C; }
    .rec-card.custom-high { border-left-color: #EF4444; }
    .rec-card.custom-medium { border-left-color: #F59E0B; }
    .rec-card.custom-low { border-left-color: #10B981; }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
        color: white;
    }
    .badge.custom-critical { background-color: #7f1d1d; border: 1px solid #ef4444; }
    .badge.custom-high { background-color: #991b1b; border: 1px solid #f87171; }
    .badge.custom-medium { background-color: #78350f; border: 1px solid #fbbf24; }
    .badge.custom-low { background-color: #064e3b; border: 1px solid #34d399; }
</style>
""", unsafe_allow_html=True)

# Import predictor
try:
    from predictor import get_predictor, predict_machine_failure
except ImportError:
    st.error("Could not import predictor module. Make sure predictor.py is available.")
    st.stop()


# ====== HELPERS ======

def load_dataset() -> pd.DataFrame:
    data_path = Path(__file__).parent / "Raw" / "ai4i2020.csv"
    if not data_path.exists():
        st.error(f"Dataset not found at {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        "UDI": "Vehicle ID",
        "Rotational speed [rpm]": "Rotational Speed",
        "Air temperature [K]": "Air Temp",
        "Process temperature [K]": "Engine Temp",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool Wear",
        "Machine failure": "Failure",
        "Product ID": "Product ID",
    })
    df["Vehicle ID"] = df["Vehicle ID"].astype(str)
    return df


def load_guidelines() -> str:
    guidelines_path = Path(__file__).parent / "docs" / "maintenance_guidelines.txt"
    if not guidelines_path.exists():
        return "No guidelines available"
    return guidelines_path.read_text()


def get_risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "High"
    if prob >= 0.6:
        return "Medium"
    return "Low"


def get_risk_badge(risk: str) -> str:
    mapping = {
        "High": "🔴 High",
        "Medium": "🟠 Medium",
        "Low": "🟢 Low",
    }
    return mapping.get(risk, "⚪ Unknown")


def get_risk_color(risk: str) -> str:
    return {
        "High": "#FF6B6B",
        "Medium": "#FFC75F",
        "Low": "#6BCB77",
    }.get(risk, "#B0B0B0")


def summarize_risk(predictions: List[Dict]) -> Dict[str, int]:
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for p in predictions:
        counts[p["risk_level"]] += 1
    return counts


def predict_fleet(df: pd.DataFrame) -> pd.DataFrame:
    predictor = get_predictor()
    if not predictor.is_available():
        st.error("ML model not available. Run python train.py first.")
        st.stop()

    results = []
    for _, row in df.iterrows():
        row_data = {
            "Type": str(row.get("Type", "M")),
            "Air_temperature": float(row.get("Air Temp", 25.0)),
            "Process_temperature": float(row.get("Engine Temp", 80.0)),
            "Rotational_speed": float(row.get("Rotational Speed", 0)),
            "Torque": float(row.get("Torque", 0)),
            "Tool_wear": float(row.get("Tool Wear", 0)),
        }
        prediction = predict_machine_failure(row_data)
        if prediction is None:
            result = {
                "Risk Level": "Unknown",
                "Probability": 0.0,
                "Priority Rank": 0,
                "Action": "Model unavailable",
            }
        else:
            risk = prediction["risk_level"]
            prob = prediction["failure_probability"]
            action = (
                "Urgent service" if risk in ["High", "Critical"] else
                "Inspect soon" if risk == "Medium" else
                "Continue monitoring"
            )
            result = {
                "Risk Level": risk,
                "Probability": prob * 100,
                "Priority Rank": 1 if risk in ["High", "Critical"] else 2 if risk == "Medium" else 3,
                "Action": action,
            }
        results.append(result)
    return pd.DataFrame(results)


import os
def get_recommendations(fleet: pd.DataFrame) -> List[Dict]:
    recs = []
    high = fleet[fleet["Risk Level"].isin(["High", "Critical"])]
    medium = fleet[fleet["Risk Level"] == "Medium"]

    # Use API call for analysis if GROQ_API_KEY is available
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            llm = ChatGroq(temperature=0.3, model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"))
            
            for _, row in high.head(3).iterrows():
                prompt = f"You are a predictive maintenance expert. Vehicle {row['Vehicle ID']} has a {row['Probability']:.0f}% failure probability. Its parameters are: Engine Temp {row.get('Engine Temp')}C, Tool Wear {row.get('Tool Wear')}min, Torque {row.get('Torque')}Nm. Provide a 1-sentence urgent and specific maintenance recommendation. Do not use quotes or prefixes."
                resp = llm.invoke([HumanMessage(content=prompt)])
                recs.append({"vehicle_id": row["Vehicle ID"], "text": resp.content})
                
            for _, row in medium.head(2).iterrows():
                prompt = f"You are a predictive maintenance expert. Vehicle {row['Vehicle ID']} has a {row['Probability']:.0f}% failure probability (Medium Risk). Its parameters are: Engine Temp {row.get('Engine Temp')}C, Tool Wear {row.get('Tool Wear')}min, Torque {row.get('Torque')}Nm. Provide a 1-sentence preventive maintenance recommendation. Do not use quotes or prefixes."
                resp = llm.invoke([HumanMessage(content=prompt)])
                recs.append({"vehicle_id": row["Vehicle ID"], "text": resp.content})
                
            if not recs:
                recs.append({"vehicle_id": "System", "text": "All vehicles are operating normally. No API recommendations needed."})
            return recs
        except Exception as e:
            st.warning(f"API call failed: {e}. Falling back to default recommendations.")

    for _, row in high.head(3).iterrows():
        recs.append({
            "vehicle_id": row["Vehicle ID"],
            "text": f"{row['Vehicle ID']} should receive inspection within 3 days due to high failure probability ({row['Probability']:.0f}%).",
        })
    for _, row in medium.head(2).iterrows():
        recs.append({
            "vehicle_id": row["Vehicle ID"],
            "text": f"{row['Vehicle ID']} should be scheduled for preventive maintenance soon with {row['Probability']:.0f}% risk.",
        })
    return recs


def extract_guidelines(goal: str, guidelines: str) -> List[Dict[str, str]]:
    snippets = []
    lines = [line.strip() for line in guidelines.splitlines() if line.strip()]
    keywords = [word.lower() for word in goal.split()[:6]]
    for line in lines:
        text = line.lower()
        if any(keyword in text for keyword in keywords) and len(snippets) < 4:
            snippets.append({"text": line, "source": "Maintenance Docs"})
    if not snippets:
        snippets = [{"text": lines[i], "source": "Maintenance Docs"} for i in range(min(3, len(lines)))]
    return snippets


def build_recommendation(risk: str, probability: float, query: str) -> Dict[str, str]:
    """Build a final recommendation and timeline from the risk profile."""
    probability_pct = f"{probability * 100:.1f}%"
    health_summary = f"{risk} risk with {probability_pct} failure probability."

    if risk in ["Critical", "High"]:
        action_plan = "Immediate inspection and service planning."
        timeline = "Inspect within 24 hours and schedule repairs immediately."
    elif risk == "Medium":
        action_plan = "Schedule preventive maintenance and monitor closely."
        timeline = "Service within 7 days."
    else:
        action_plan = "Keep monitoring and follow routine maintenance checks."
        timeline = "Review again in 14 days."

    # Add query-aware guidance when possible
    if "urgent" in query.lower() or "risk" in query.lower():
        health_summary += " Query indicates urgency, so prioritize this vehicle."

    return {
        "vehicle_id": "",
        "health_summary": health_summary,
        "risk_level": risk,
        "action_plan": action_plan,
        "timeline": timeline,
    }


# ====== UI ======

def main():
    st.markdown("""
    <style>
    .hero-title {font-size:45px; font-weight:700; color:#ffffff;}
    .hero-subtitle {font-size:18px; color:#d0d7e2; margin-top:-10px;}
    .card {padding:20px; border-radius:16px; background:#111827; border:1px solid #24303f;}
    .small-card {padding:18px; border-radius:14px; background:#141b28; border:1px solid #27334d;}
    .badge {padding:6px 12px; border-radius:999px; color:white; font-weight:600;}
    .high {background:#ff6b6b;}
    .medium {background:#ffc75f; color:#1f2937;}
    .low {background:#6bcb77;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hero-title'>🚗 Fleet Maintenance Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>AI-powered predictive maintenance and service planning for fleet operations.</div>", unsafe_allow_html=True)
    st.write("---")

    guidelines = load_guidelines()

    if "fleet_data" not in st.session_state:
        st.session_state.fleet_data = []
        st.session_state.analysis_goal = "Prioritize urgent vehicles"
        st.session_state.analysis_results = None
        st.session_state.selected_vehicle = None

    with st.sidebar:
        st.markdown("##  Enter Vehicle Details")
        with st.form("vehicle_input_form"):
            vehicle_id = st.text_input("Vehicle ID", value=f"VEH-{len(st.session_state.fleet_data) + 1}")
            vehicle_type = st.selectbox("Vehicle Type", ["M", "L", "H"], index=0)
            engine_temp = st.slider("Engine Temp (°C)", min_value=0.0, max_value=150.0, value=80.0, step=0.1)
            air_temp = st.slider("Air Temp (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.1)
            rotational_speed = st.slider("Rotational Speed (rpm)", min_value=0, max_value=3000, value=1500, step=1)
            torque = st.slider("Torque (Nm)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
            tool_wear = st.slider("Tool Wear (min)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
            mileage = st.slider("Mileage", min_value=0, max_value=500000, value=100000, step=1000)
            oil_quality = st.slider("Oil Quality", min_value=0, max_value=100, value=80)
            brake_wear = st.slider("Brake Wear", min_value=0, max_value=100, value=20)
            battery_health = st.slider("Battery Health", min_value=0, max_value=100, value=90)
            days_since_last_service = st.slider("Days Since Last Service", min_value=0, max_value=1000, value=30, step=1)
            tire_health = st.slider("Tire Health", min_value=0, max_value=100, value=90)
            downtime_count = st.slider("Downtime Count", min_value=0, max_value=50, value=0, step=1)
            add_vehicle = st.form_submit_button("Add vehicle")

        if add_vehicle:
            new_vehicle = {
                "Vehicle ID": vehicle_id or f"VEH-{len(st.session_state.fleet_data) + 1}",
                "Type": vehicle_type,
                "Engine Temp": engine_temp,
                "Air Temp": air_temp,
                "Rotational Speed": rotational_speed,
                "Torque": torque,
                "Tool Wear": tool_wear,
                "Mileage": mileage,
                "Oil Quality": oil_quality,
                "Brake Wear": brake_wear,
                "Battery Health": battery_health,
                "Days Since Last Service": days_since_last_service,
                "Tire Health": tire_health,
                "Downtime Count": downtime_count,
            }
            st.session_state.fleet_data.append(new_vehicle)
            st.session_state.analysis_results = None
            st.success("Vehicle added to the fleet.")
            st.rerun()

        if st.button("Clear fleet"):
            st.session_state.fleet_data = []
            st.session_state.analysis_results = None
            st.rerun()

        if st.button("Load sample fleet"):
            st.session_state.analysis_results = None
            st.session_state.fleet_data = [
                {
                    "Vehicle ID": "TRK-101",
                    "Type": "H",
                    "Engine Temp": 110.0,
                    "Air Temp": 35.0,
                    "Rotational Speed": 1300,
                    "Torque": 65.0,
                    "Tool Wear": 220.0,
                    "Mileage": 150000,
                    "Oil Quality": 25,
                    "Brake Wear": 85,
                    "Battery Health": 45,
                    "Days Since Last Service": 210,
                    "Tire Health": 60,
                    "Downtime Count": 4,
                },
                {
                    "Vehicle ID": "VAN-008",
                    "Type": "M",
                    "Engine Temp": 91.0,
                    "Air Temp": 22.0,
                    "Rotational Speed": 1350,
                    "Torque": 18.0,
                    "Tool Wear": 8.0,
                    "Mileage": 62000,
                    "Oil Quality": 76,
                    "Brake Wear": 35,
                    "Battery Health": 81,
                    "Days Since Last Service": 90,
                    "Tire Health": 91,
                    "Downtime Count": 1,
                },
                {
                    "Vehicle ID": "ENG-404",
                    "Type": "L",
                    "Engine Temp": 102.0,
                    "Air Temp": 30.0,
                    "Rotational Speed": 1359,
                    "Torque": 55.0,
                    "Tool Wear": 206.0,
                    "Mileage": 115000,
                    "Oil Quality": 45,
                    "Brake Wear": 62,
                    "Battery Health": 72,
                    "Days Since Last Service": 140,
                    "Tire Health": 68,
                    "Downtime Count": 2,
                },
                {
                    "Vehicle ID": "SUV-999",
                    "Type": "M",
                    "Engine Temp": 102.0,
                    "Air Temp": 30.0,
                    "Rotational Speed": 1350,
                    "Torque": 55.0,
                    "Tool Wear": 206.0,
                    "Mileage": 132000,
                    "Oil Quality": 50,
                    "Brake Wear": 58,
                    "Battery Health": 68,
                    "Days Since Last Service": 135,
                    "Tire Health": 74,
                    "Downtime Count": 1,
                },
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("### Analysis Goal")
        st.session_state.analysis_goal = st.text_area(
            "Enter a fleet analysis goal",
            value=st.session_state.analysis_goal,
            placeholder="Prioritize urgent vehicles for safety-critical service...",
            height=120,
        )
        st.markdown("---")
        analyze = st.button("Analyze Fleet", width="stretch")
        if analyze:
            st.session_state.run_analysis = True

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    total = len(st.session_state.fleet_data)
    results = st.session_state.get("analysis_results")
    if results is not None:
        high_count = results[results["Risk Level"].isin(["High", "Critical"])].shape[0]
        medium_count = results[results["Risk Level"] == "Medium"].shape[0]
        low_count = results[results["Risk Level"] == "Low"].shape[0]
        avg_health_val = 100 - (high_count * 25 + medium_count * 10)
        avg_health = f"{max(0, min(100, avg_health_val))}%"
    else:
        high_count = "-"
        medium_count = "-"
        low_count = "-"
        avg_health = "N/A"

    c1.markdown(f"**Total Vehicles**\n### {total}")
    c2.markdown(f"**High Risk Vehicles**\n### {high_count}")
    c3.markdown(f"**Medium Risk Vehicles**\n### {medium_count}")
    c4.markdown(f"**Low Risk Vehicles**\n### {low_count}")
    c5.markdown(f"**Avg Health Score**\n### {avg_health}")
    st.write("---")

    st.markdown("## 🚚 Fleet Data Table")
    fleet_df = pd.DataFrame(st.session_state.fleet_data)
    if fleet_df.empty:
        st.info("Add vehicle details in the side panel, then click Analyze Fleet.")
    else:
        if fleet_df.columns.duplicated().any():
            st.error("The fleet table contains duplicate column names, which prevents the editor from rendering correctly.")
            st.write("Duplicate columns:", fleet_df.columns[fleet_df.columns.duplicated()].tolist())
            st.dataframe(fleet_df, width="stretch")
            return

        if hasattr(st, "data_editor"):
            fleet_df = st.data_editor(
                fleet_df,
                key="fleet_editor",
                width="stretch",
            )
        elif hasattr(st, "experimental_data_editor"):
            fleet_df = st.experimental_data_editor(
                fleet_df,
                key="fleet_editor",
                width="stretch",
            )
        else:
            st.warning("Editable table is not supported in this Streamlit version. Showing read-only fleet data.")
            st.dataframe(fleet_df, width="stretch")

        if not fleet_df.empty:
            st.session_state.fleet_data = fleet_df.to_dict(orient="records")
            fleet_df = pd.DataFrame(st.session_state.fleet_data)

    if st.session_state.get("run_analysis"):
        if len(st.session_state.fleet_data) > 0:
            analysis_df = pd.DataFrame(st.session_state.fleet_data)
            analysis_df["Vehicle ID"] = analysis_df["Vehicle ID"].astype(str)
            predictions_df = predict_fleet(analysis_df)
            merged = pd.concat([analysis_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
            merged = merged.sort_values(by=["Priority Rank", "Probability"], ascending=[True, False])
            st.session_state.analysis_results = merged
            st.session_state.selected_vehicle = merged.iloc[0]["Vehicle ID"] if not merged.empty else None
        
        # Only rerun if we actually triggered a new analysis from the button
        if st.session_state.get("run_analysis"):
            st.session_state.run_analysis = False
            st.rerun()
        st.session_state.run_analysis = False

    if st.session_state.analysis_results is not None:
        res = st.session_state.analysis_results
        
        st.markdown("## 💡 AI Recommendation Section")
        recs = get_recommendations(res)
        
        # Display recommendations as cards
        if recs and recs[0]["vehicle_id"] != "System":
            for rec in recs:
                vid = rec["vehicle_id"]
                v_data = res[res["Vehicle ID"] == vid].iloc[0]
                r_level = v_data["Risk Level"]
                prob = v_data["Probability"]
                action = v_data["Action"]
                
                css_class = "custom-" + r_level.lower()
                badge_text = f"ACTION: {action.upper()}"
                
                st.markdown(f"""
                <div class="rec-card {css_class}">
                    <div class="badge {css_class}">{badge_text}</div>
                    <div style="font-size: 1.1em; margin-bottom: 8px;">
                        <strong>Vehicle: {vid}</strong> &nbsp;|&nbsp; 
                        Risk Level: {r_level} ({prob:.1f}%)
                    </div>
                    <div style="color: #d1d5db; line-height: 1.5;">
                        <strong>**RECOMMENDATION:**</strong> {rec['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All vehicles are operating normally. No API recommendations needed.")
            
        st.markdown("---")
        st.markdown("## 📈 Risk Visualizations")
        chart1, chart2 = st.columns(2)
        with chart1:
            risk_counts = res["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig1 = px.pie(
                risk_counts,
                names="Risk Level",
                values="Count",
                title="Risk Distribution",
                color_discrete_map={"Critical": "#D32F2F", "High": "#FF6B6B", "Medium": "#FFC75F", "Low": "#6BCB77"}
            )
            st.plotly_chart(fig1, width="stretch")
        with chart2:
            fig2 = px.bar(
                res,
                x="Vehicle ID",
                y="Probability",
                color="Risk Level",
                title="Failure Probability by Vehicle",
                color_discrete_map={"Critical": "#D32F2F", "High": "#FF6B6B", "Medium": "#FFC75F", "Low": "#6BCB77"}
            )
            st.plotly_chart(fig2, width="stretch")
            
        st.markdown("### Feature Risk Analysis")
        fig3 = px.scatter(
            res,
            x="Engine Temp",
            y="Tool Wear",
            color="Risk Level",
            title="Engine Temperature vs Tool Wear Risk Profile",
            hover_data=["Vehicle ID"],
            color_discrete_map={"Critical": "#D32F2F", "High": "#FF6B6B", "Medium": "#FFC75F", "Low": "#6BCB77"}
        )
        fig3.update_traces(marker=dict(size=12))
        st.plotly_chart(fig3, width="stretch")

    else:
        st.info("Fill in fleet details and click Analyze Fleet to generate results.")

if __name__ == "__main__":
    main()
