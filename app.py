import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AutoMind AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# CUSTOM STYLING
# ==============================
st.markdown("""
<style>
    .stApp {
        background: #0B1120;
        color: #E5E7EB;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 800;
        color: white;
        margin-bottom: 0;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #CBD5E1;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .dashboard-card {
        background: #111827;
        border: 1px solid #1F2937;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
        color: white;
    }

    .high {
        color: #EF4444;
        font-weight: bold;
    }

    .medium {
        color: #F59E0B;
        font-weight: bold;
    }

    .low {
        color: #10B981;
        font-weight: bold;
    }

    .recommendation-box {
        background: #111827;
        border-left: 5px solid #3B82F6;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# RISK PREDICTION LOGIC
# ==============================
def predict_risk(vehicle):
    score = 0

    if vehicle["Engine Temp"] > 100:
        score += 30

    if vehicle["Tool Wear"] > 150:
        score += 30

    if vehicle["Brake Wear"] > 70:
        score += 20

    if vehicle["Battery Health"] < 50:
        score += 20

    if score >= 70:
        risk = "High"
        action = "Urgent Service Required"
    elif score >= 40:
        risk = "Medium"
        action = "Schedule Preventive Maintenance"
    else:
        risk = "Low"
        action = "Continue Monitoring"

    return {
        "Risk Level": risk,
        "Failure Probability": min(score, 100),
        "Recommended Action": action
    }


# ==============================
# SAMPLE DATA
# ==============================
def load_demo_data():
    return [
        {
            "Vehicle ID": "TRK-101",
            "Type": "H",
            "Engine Temp": 112.0,
            "Tool Wear": 220.0,
            "Brake Wear": 85,
            "Battery Health": 42
        },
        {
            "Vehicle ID": "VAN-008",
            "Type": "M",
            "Engine Temp": 90.0,
            "Tool Wear": 35.0,
            "Brake Wear": 30,
            "Battery Health": 80
        }
    ]


# ==============================
# MAIN APP
# ==============================
def main():
    st.markdown(
        "<div class='hero-title'>🚀 AutoMind AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='hero-subtitle'>AI-powered predictive maintenance system for intelligent fleet operations.</div>",
        unsafe_allow_html=True
    )

    st.write("---")

    if "fleet_data" not in st.session_state:
        st.session_state.fleet_data = []

    # ==============================
    # SIDEBAR
    # ==============================
    with st.sidebar:
        st.markdown("## AutoMind Vehicle Control Panel")

        with st.form("vehicle_form"):
            vehicle_id = st.text_input("Vehicle ID", "VEH-001")
            vehicle_type = st.selectbox(
                "Vehicle Type",
                ["L", "M", "H"]
            )

            engine_temp = st.slider(
                "Engine Temperature (°C)",
                0.0, 150.0, 80.0
            )

            tool_wear = st.slider(
                "Tool Wear (min)",
                0.0, 300.0, 20.0
            )

            brake_wear = st.slider(
                "Brake Wear (%)",
                0, 100, 20
            )

            battery_health = st.slider(
                "Battery Health (%)",
                0, 100, 90
            )

            submit = st.form_submit_button("Add Vehicle")

        if submit:
            st.session_state.fleet_data.append({
                "Vehicle ID": vehicle_id,
                "Type": vehicle_type,
                "Engine Temp": engine_temp,
                "Tool Wear": tool_wear,
                "Brake Wear": brake_wear,
                "Battery Health": battery_health
            })

            st.success("Vehicle added successfully")
            st.rerun()

        if st.button("Load Demo Data"):
            st.session_state.fleet_data = load_demo_data()
            st.rerun()

        if st.button("Reset Dashboard"):
            st.session_state.fleet_data = []
            st.rerun()

    # ==============================
    # DASHBOARD
    # ==============================
    st.markdown(
        "<div class='section-title'>AutoMind Fleet Intelligence Dashboard</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.fleet_data:
        st.info("Add vehicles or load demo data to begin analysis.")
        return

    df = pd.DataFrame(st.session_state.fleet_data)

    predictions = []
    for _, row in df.iterrows():
        predictions.append(predict_risk(row.to_dict()))

    result_df = pd.concat(
        [df, pd.DataFrame(predictions)],
        axis=1
    )

    # ==============================
    # METRICS
    # ==============================
    high_count = result_df[
        result_df["Risk Level"] == "High"
    ].shape[0]

    medium_count = result_df[
        result_df["Risk Level"] == "Medium"
    ].shape[0]

    low_count = result_df[
        result_df["Risk Level"] == "Low"
    ].shape[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("🔴 High Risk", high_count)
    c2.metric("🟠 Medium Risk", medium_count)
    c3.metric("🟢 Low Risk", low_count)

    st.write("")

    # ==============================
    # DATA TABLE
    # ==============================
    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.write("---")

    # ==============================
    # CHARTS
    # ==============================
    st.markdown(
        "<div class='section-title'>Risk Analysis & Recommendations</div>",
        unsafe_allow_html=True
    )

    fig = px.bar(
        result_df,
        x="Vehicle ID",
        y="Failure Probability",
        color="Risk Level",
        title="Vehicle Failure Probability Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==============================
    # RECOMMENDATIONS
    # ==============================
    high_risk_df = result_df[
        result_df["Risk Level"] == "High"
    ]

    if not high_risk_df.empty:
        st.warning(
            "Immediate attention required for high-risk vehicles."
        )

        for _, row in high_risk_df.iterrows():
            st.markdown(f"""
            <div class="recommendation-box">
                <b>{row['Vehicle ID']}</b><br>
                Risk Level: <span class="high">{row['Risk Level']}</span><br>
                Failure Probability: {row['Failure Probability']}%<br>
                Recommended Action: {row['Recommended Action']}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.success(
            "All vehicles are operating normally. No urgent maintenance required."
        )


if __name__ == "__main__":
    main()