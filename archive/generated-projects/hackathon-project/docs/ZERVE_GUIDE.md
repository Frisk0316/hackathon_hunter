# PredictPulse — Zerve Platform Setup Guide

## Step-by-Step: Setting Up the Project on Zerve

### 1. Create Zerve Account
- Go to [zerve.ai](https://www.zerve.ai/)
- Sign up for a free account (free tier includes unlimited public projects)

### 2. Create New Canvas
- Click "Create Canvas" from the dashboard
- Name it "PredictPulse"
- Set visibility to **Public** (required for ZerveHack submission)

### 3. Add Python Blocks
Create 6 Python blocks in your canvas. Copy the code from each file:

| Block | Source File | Name in Canvas |
|-------|-----------|----------------|
| 1 | `src/01_data_collection.py` | Data Collection |
| 2 | `src/02_feature_engineering.py` | Feature Engineering |
| 3 | `src/03_model_training.py` | Model Training |
| 4 | `src/04_visualization.py` | Visualization |
| 5 | `src/05_claude_analysis.py` | Claude AI Analysis |
| 6 | `src/06_deploy_api.py` | API Deployment |

### 4. Set Environment Variables
In Zerve settings, add:
- `ANTHROPIC_API_KEY` — your Claude API key
- `FRED_API_KEY` — your FRED API key (optional)

### 5. Install Dependencies
Zerve should have most packages pre-installed. If needed, add a pip install block:
```python
!pip install anthropic pytrends
```

### 6. Run Pipeline
Execute blocks 1 through 6 in order. Each block depends on variables from previous blocks.

### 7. Deploy API
- Select Block 6
- Use Zerve's "Deploy" feature to create an API endpoint
- The `predict_reliability` function becomes your API handler

### 8. Publish and Share
- Ensure canvas is set to Public
- Copy the shareable URL
- This is your primary Zerve project URL for the Devpost submission

## Submission Checklist

- [ ] Zerve project is public and shareable
- [ ] All 6 blocks run without errors
- [ ] Visualizations render correctly
- [ ] API endpoint is deployed and responding
- [ ] Demo video recorded (under 3 minutes)
- [ ] Social post published with @Zerve_AI tag
- [ ] Devpost submission form completed
- [ ] Description is under 300 words

## Troubleshooting

**"Module not found" errors:**
Add `!pip install <package>` at the top of the relevant block.

**Metaculus API rate limiting:**
The code includes 0.5s delays between requests. If you still hit limits, reduce `limit` parameter in `fetch_metaculus_questions()`.

**Claude API errors:**
The system falls back to template-based analysis automatically. Check your API key is set correctly in Zerve environment variables.

**FRED API unavailable:**
Set `FRED_API_KEY = None` in Block 1. The pipeline works without FRED data (uses Metaculus features only).
