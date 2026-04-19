# 🚀 AutoMind AI

AutoMind AI is a smart AI-powered predictive maintenance platform built for intelligent fleet operations.

It helps businesses monitor vehicle health, predict failures early, reduce downtime, and optimize maintenance planning using Machine Learning + Streamlit Dashboard + AI Recommendations.

---

# ✨ Features

## ✅ Smart Vehicle Monitoring

- Add vehicle details using an interactive control panel
- Track Engine Temperature, Tool Wear, Brake Wear, Battery Health, and more
- Real-time fleet monitoring dashboard

---

## ✅ Predictive Maintenance System

- Detects possible machine failures before breakdown
- Calculates Failure Probability (0–100%)
- Automatically classifies vehicles into:

🔴 High Risk  
🟠 Medium Risk  
🟢 Low Risk

---

## ✅ AI-Powered Recommendations

- Smart maintenance suggestions for risky vehicles
- Priority-based service planning
- Preventive maintenance workflow

---

## ✅ Interactive Analytics Dashboard

- Premium dark UI with modern dashboard design
- Live Data Table for fleet monitoring
- Risk Visualization using interactive charts
- Failure Probability Analysis with Plotly

---

# 📁 Project Structure

```bash
AutoMind-AI/
│
├── app.py                  # Main Streamlit Dashboard
├── predictor.py            # Prediction logic
├── train.py                # ML model training
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
│
├── Models/                 # Saved trained models
├── Raw/                    # Raw datasets
├── docs/                   # Maintenance guidelines
├── chroma_db/              # Vector database
├── src/                    # Supporting source files
└── .streamlit/             # Streamlit config