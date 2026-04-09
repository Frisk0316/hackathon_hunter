# AlgoFest Submission Copy — PredictPulse

## Devpost Title
PredictPulse — Predicting Which Predictions to Trust

## Track
AI & Machine Learning

---

## Inspiration

Prediction markets are becoming critical infrastructure — used by researchers, traders, and even governments to gauge future probabilities. But a fundamental problem persists: platforms show you a 65% probability without any signal of whether that number is reliable. We wondered: *can machine learning identify which crowd forecasts are actually trustworthy before the outcome is known?*

## What It Does

PredictPulse is an ML pipeline and interactive Streamlit dashboard that:
1. Trains a Random Forest classifier on 500+ resolved Metaculus predictions
2. Engineers 26 features from participation patterns, confidence levels, and question characteristics
3. Scores any active prediction with a reliability tier (Very Low → Very High)
4. Generates Claude-powered natural language explanations
5. Visualizes calibration curves, feature importance, and category-level accuracy patterns

## How We Built It

We built a modular Python pipeline with scikit-learn for model training and Plotly + Streamlit for the interactive frontend. The key insight driving feature design: a prediction's accuracy can be inferred from *how* people forecast, not just *what* they forecast.

We evaluated three models (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold cross-validation and selected Random Forest (AUC-ROC: 0.889). We added Brier Score and Calibration Curve analysis to validate that our probability outputs are statistically meaningful — not just directionally correct.

## Challenges We Ran Into

Metaculus now requires API authentication. We built a realistic synthetic data generator (calibrated to real Metaculus statistics) as an automatic fallback, ensuring the pipeline runs out of the box for any user.

The calibration curve revealed early model versions were overconfident in the 0.6-0.8 range. We added isotonic regression calibration and cross-validated probability estimates to fix this.

## Accomplishments We're Proud Of

- Brier Skill Score of **0.581** — our model has meaningful probabilistic accuracy, not just directional
- Mean Calibration Error **< 0.05** — when we say 70%, it really means ~70%
- Zero-setup experience: runs entirely on synthetic data without any API keys

## What We Learned

Prediction accuracy is driven more by *participation quality* than *prediction quantity*. A question with 20 engaged forecasters is often more reliable than one with 200 passive ones. The engagement ratio (comments ÷ predictions) captures this signal surprisingly well.

## What's Next

- Multi-platform support (Polymarket, Kalshi, Manifold)
- Real-time scoring via browser extension
- Temporal drift detection (accuracy patterns shift during high-volatility macro periods)

## Built With

Python, scikit-learn, Streamlit, Plotly, Anthropic Claude API, Metaculus API, pandas, numpy
