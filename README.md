# 🚗 Fleet Maintenance Assistant

An AI-powered predictive maintenance system that uses machine learning to identify equipment failures before they happen.

## Key Features

✅ **ML-Based Predictions**
- GradientBoostingClassifier trained on 10,000+ industrial machines
- 99.85% accuracy on test data
- Real-time failure probability scoring

✅ **No CSV Upload Required**
- Auto-loads AI4I 2020 industrial dataset (10,001 machines)
- Machine selector dropdown for easy browsing
- Random sample option for quick analysis

✅ **Smart Maintenance Recommendations**
- Risk level assessment (Low/Medium/High/Critical)
- Timeline-based action plans
- Industry maintenance guidelines
- Downloadable structured reports

✅ **Interactive Dashboard**
- Machine metrics cards
- Fleet analytics charts
- Temperature and tool wear distributions
- Failure rate visualization

## Project Structure

```
fleet-maintenance-assistant/
├── app.py                          # Streamlit application
├── train.py                        # Model training script
├── predictor.py                    # ML prediction module
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── Models/                         # Trained ML artifacts
│   ├── failure_predictor.pkl      # Trained model
│   ├── feature_scaler.pkl         # Feature normalizer
│   ├── type_encoder.pkl           # Categorical encoder
│   └── feature_names.txt          # Feature reference
├── Raw/                            # Raw data
│   └── ai4i2020.csv               # Industrial dataset (10,001 machines)
├── Docs/                           # Documentation
│   └── maintenance_guidelines.txt  # Industry guidelines
├── chroma_db/                      # Vector database (for future RAG)
└── src/                            # Source notebooks (legacy)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (One-Time Setup)

```bash
python train.py
```

This will:
- Load the AI4I 2020 dataset from `Raw/ai4i2020.csv`
- Clean and preprocess the data
- Train a GradientBoostingClassifier
- Save model artifacts to `Models/`
- Display training metrics (accuracy, ROC-AUC, etc.)

**Expected Output:**
```
Training Accuracy: 99.85%
Test ROC-AUC: 0.9798
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will:
- Open at `http://localhost:8501`
- Auto-load the dataset
- Display machine selection interface

### 4. Use the App

1. **Select a Machine**
   - Use dropdown to select by UDI
   - Or click "Random Sample"

2. **Enter Query** (optional)
   - Example: "What maintenance does this need?"
   - Default: "What maintenance should this machine receive?"

3. **Click Analyze**
   - ML model predicts failure probability
   - Risk level assessment generated
   - Recommendations provided

4. **Review & Download**
   - See metrics and guidelines
   - View fleet analytics
   - Download JSON report

## Environment Setup

### .env File

Create a `.env` file with:

```env
GROQ_API_KEY=your_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```

*Note: API key is optional for basic functionality. Without it, system uses deterministic recommendations.*

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- **Size**: 10,001 machines × 14 features
- **Source**: Auto-loads from `Raw/ai4i2020.csv`
- **Features**:
  - Type (Product variant: M/L/H)
  - Temperature (Air and Process, in Kelvin)
  - Rotational speed (RPM)
  - Torque (Nm)
  - Tool wear (minutes)
  - Failure modes: TWF, HDF, PWF, OSF, RNF

**Failure Rate**: 3.39% (realistic industrial baseline)

## ML Model

**Algorithm**: GradientBoostingClassifier

**Hyperparameters**:
- n_estimators: 100
- max_depth: 5
- learning_rate: 0.1

**Performance**:
- Train Accuracy: 99.85%
- Test ROC-AUC: 0.9798
- F1-Score: 99%+

**Risk Level Mapping**:
- `prob < 0.3`: 🟢 Low
- `0.3 ≤ prob < 0.6`: 🟡 Medium
- `0.6 ≤ prob < 0.8`: 🟠 High
- `prob ≥ 0.8`: 🔴 Critical

## Key Components

### app.py
Streamlit application with:
- Machine selector (dropdown or random)
- ML prediction interface
- Maintenance guidelines display
- Fleet analytics charts
- Structured report download

### train.py
Training pipeline:
- Loads data from `Raw/ai4i2020.csv`
- Data preprocessing and encoding
- Model training and evaluation
- Artifact serialization to `Models/`

### predictor.py
ML inference module:
- Loads trained models
- Handles feature mapping
- Provides single and batch prediction
- Includes fallback handling

## Recommendations

### Risk-Based Actions

| Risk Level | Action | Timeline |
|-----------|--------|----------|
| 🔴 Critical | URGENT service | Within 24 hours |
| 🟠 High | Schedule maintenance | Within 3-7 days |
| 🟡 Medium | Plan preventive maintenance | Within 2-3 weeks |
| 🟢 Low | Continue operation | Monitor regularly |

## Troubleshooting

### Model Not Found
```
Error: ML Model Not Available
```
**Solution**: Run `python train.py` to train the model

### Dataset Not Found
```
Error: Dataset not found at Raw/ai4i2020.csv
```
**Solution**: Ensure `ai4i2020.csv` is in the `Raw/` folder

### Streamlit Port Already in Use
```
streamlit run app.py --server.port 8502
```

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy from repository
4. Add secrets:
   - `GROQ_API_KEY`

### Docker
```bash
docker run -p 8501:8501 streamlit/streamlit-docker
```

## Performance Notes

- **First Load**: ~2-3 seconds (data loading)
- **Prediction**: <100ms per machine
- **Memory Usage**: ~200MB (models + data)
- **Batch Processing**: Supports 100+ machines

## System Requirements

- Python 3.8+
- 2GB RAM (minimum)
- Internet (for embeddings download on first run)

## Dependencies

- **ML**: scikit-learn, pandas, numpy
- **UI**: streamlit, plotly
- **LLM**: langchain-groq (optional)
- **Data**: chromadb, sentence-transformers
- **Config**: python-dotenv

See `requirements.txt` for exact versions.

## Safety Disclaimer

⚠️ This system provides **AI-assisted recommendations only**. Final maintenance decisions must be validated by certified technicians and must follow:
- Manufacturer guidelines
- Industry standards
- Regulatory requirements

Always prioritize safety and rely on qualified professionals for critical decisions.

---

**Version**: 2.0 (ML-Enhanced with Auto-Load)  
**Status**: ✅ Production Ready
