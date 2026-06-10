# PredictPulse — Architecture

## System Overview

PredictPulse is a 6-block data science pipeline built on the Zerve platform. Each block is a self-contained Python module that feeds into the next.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZERVE PLATFORM                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Block 1  │→ │ Block 2  │→ │ Block 3  │→ │ Block 4  │       │
│  │ Data     │  │ Feature  │  │ Model    │  │ Visual-  │       │
│  │ Collect  │  │ Engineer │  │ Training │  │ ization  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       ↓                            ↓                            │
│  ┌──────────┐              ┌──────────┐                        │
│  │ External │              │ Block 5  │→ ┌──────────┐          │
│  │ APIs     │              │ Claude   │  │ Block 6  │          │
│  └──────────┘              │ Analysis │  │ API      │→ Users   │
│                            └──────────┘  │ Deploy   │          │
│                                          └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Block 1: Data Collection
- **Input:** API endpoints (Metaculus, FRED, Google Trends)
- **Output:** `resolved_df` (500+ questions), `open_df` (100 active), `fred_df`, `trends_df`
- **Dependencies:** `requests`, `pandas`

### Block 2: Feature Engineering
- **Input:** Raw DataFrames from Block 1
- **Output:** `df_features` (ML-ready), `feature_cols` (feature list)
- **Features (20+):**
  - Time: question_lifespan_days, time_to_close, resolve_month/dayofweek
  - Participation: log_prediction_count, log_comments, engagement_ratio, prediction_density
  - Complexity: log_description_length, title_length, title_word_count
  - Signal: has_number_in_title, is_yes_no_question
  - Confidence: community_pred, pred_confidence, pred_extreme
  - Category: one-hot encoded top 10 categories
  - Economic: FRED indicator z-scores (when available)
- **Target:** `prediction_accurate` (binary: error < 0.3)

### Block 3: Model Training
- **Input:** Feature matrix and target from Block 2
- **Output:** Trained `model`, `evaluation` metrics, `importance_df`
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation:** 5-fold stratified cross-validation, AUC-ROC, F1, accuracy
- **Selection:** Highest AUC-ROC wins

### Block 4: Visualization
- **Input:** Feature DataFrame, importance DataFrame, scored questions
- **Output:** 4 interactive Plotly charts
- **Charts:**
  1. Accuracy Overview Dashboard (4-panel)
  2. Feature Importance Bar Chart
  3. Reliability Scorecard (open questions)
  4. Category Analysis Bubble Chart

### Block 5: Claude AI Analysis
- **Input:** Scored questions from Block 3
- **Output:** Natural language reliability reports
- **API:** Anthropic Claude (claude-sonnet-4-20250514)
- **Fallback:** Template-based analysis when API unavailable

### Block 6: API Deployment
- **Input:** Request JSON with question metadata
- **Output:** Reliability score, tier, analysis, top factors
- **Deployment:** Zerve API endpoint

## External Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| `requests` | HTTP client for APIs | Yes |
| `pandas` | Data manipulation | Yes |
| `numpy` | Numerical computation | Yes |
| `scikit-learn` | ML models and evaluation | Yes |
| `plotly` | Interactive visualization | Yes |
| `anthropic` | Claude API client | Optional |
| `pytrends` | Google Trends data | Optional |

## API Schema

### Request
```json
{
  "title": "string (prediction question text)",
  "community_prediction": "float (0.0-1.0)",
  "prediction_count": "int",
  "description_length": "int",
  "num_comments": "int",
  "question_age_days": "float",
  "category": "string"
}
```

### Response
```json
{
  "reliability_score": "float (0.0-1.0)",
  "reliability_tier": "Very Low | Low | Medium | High | Very High",
  "community_prediction": "float",
  "analysis": "string (natural language)",
  "top_factors": [
    {
      "feature": "string",
      "value": "float",
      "importance": "float",
      "direction": "positive | neutral"
    }
  ],
  "metadata": {
    "model": "string",
    "training_samples": "int",
    "features_used": "int",
    "version": "string"
  }
}
```
