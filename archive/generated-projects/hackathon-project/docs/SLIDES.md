---
title: PredictPulse
tags: hackathon, ZerveHack, AI, prediction-markets
slideOptions:
  theme: white
  transition: slide
---

# PredictPulse
### AI-Powered Prediction Market Intelligence
ZerveHack 2026 | Data Science Track

---

## The Problem

- Prediction markets: **$1B+ annual volume** across Polymarket, Kalshi, Metaculus
- **68% of participants** have no tools to assess forecast reliability
- All predictions treated equally — no signal from noise

> *"Not all predictions are created equal. Can AI tell us which ones to trust?"*

---

## Our Solution

**PredictPulse** builds an ML pipeline on Zerve that:

1. Analyzes **500+ resolved predictions** for accuracy patterns
2. Cross-references with **economic indicators** and **social signals**
3. Deploys a **live API** that scores any prediction's reliability

![Architecture Overview](architecture.png)

---

## Live Demo

**Input:** "Will global temperature exceed 1.5°C before 2030?"
- Community prediction: 62%
- 245 forecasters

**PredictPulse Output:**
- Reliability Score: **82% (High)**
- Top factor: High participation density
- Claude analysis: detailed explanation of why this forecast is trustworthy

---

## How It Works

```
Metaculus API ──→ Feature Engineering ──→ ML Model ──→ API
FRED Data    ──→ (20+ features)      ──→ (GBM)    ──→ + Claude AI
Trends       ──→                     ──→           ──→ Analysis
```

**20+ engineered features** including:
- Participation density & engagement ratio
- Confidence extremity & question complexity
- Economic context (GDP, VIX, CPI z-scores)
- Category-specific accuracy baselines

---

## Key Findings

| Insight | Impact |
|---------|--------|
| Participation density is #1 predictor | More forecasters/day → more accurate |
| Extreme predictions are less reliable | 90%+ or 10%- confidence = red flag |
| Longer questions = better accuracy | Complexity attracts expert forecasters |
| Economic volatility hurts accuracy | High VIX periods → worse predictions |

---

## Zerve in Action

PredictPulse showcases Zerve's **full platform:**

- **AI Agent** — automated data discovery and pipeline building
- **Multi-language Notebooks** — Python + SQL in one canvas
- **Plotly Visualizations** — interactive accuracy dashboards
- **API Deployment** — live scoring endpoint from notebook to production

---

## Impact

- **For Traders:** Know which predictions to trust before placing bets
- **For Researchers:** Quantified accuracy patterns across domains
- **For Platforms:** Better calibration tools for their forecasters

> Potential to improve prediction accuracy by **15-25%** through reliability-aware decision making

---

## Roadmap

| Phase | Timeline | Goal |
|-------|----------|------|
| MVP | Now | Single-platform scoring with ML + Claude |
| v1.0 | 3 months | Multi-platform (Polymarket, Kalshi) + real-time |
| Scale | 6 months | Browser extension + trading platform integrations |

---

## Thank You

**PredictPulse** — Turning crowd wisdom into actionable intelligence

[Zerve Project](ZERVE_URL) | [GitHub](GITHUB_URL)

Built with Zerve AI + Claude API + scikit-learn
