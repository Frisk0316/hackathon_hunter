# Devpost Submission Copy — PredictPulse

## Title
PredictPulse — AI-Powered Prediction Market Intelligence

## Tagline
Turning crowd wisdom into actionable accuracy scores with ML and Claude AI

---

## Description (~300 words for ZerveHack)

**What question did we ask?**

Can we predict which prediction market forecasts will be most accurate by analyzing metadata signals that most traders ignore?

**What did we find?**

Using 500+ resolved predictions from Metaculus, we discovered that not all crowd wisdom is equal. Participation density — the rate of new forecasts per day — is the single strongest predictor of accuracy. Questions attracting sustained daily engagement are significantly more reliable than those with burst-then-fade participation patterns.

We also found that extreme consensus (>90% or <10%) is a reliability red flag: predictions where the crowd is "certain" have higher error rates than those in the 30-70% range. Economic volatility (measured by VIX and consumer sentiment) suppresses accuracy across all categories, while longer question descriptions correlate with better outcomes — likely because complexity attracts more expert forecasters.

**Why does it matter?**

Prediction markets are becoming critical infrastructure for decision-making in finance, policy, and research. Our Gradient Boosting classifier, trained on 20+ engineered features, can distinguish high-reliability predictions from unreliable ones, enabling a potential 15-25% improvement in prediction-based decisions.

PredictPulse deploys as a live Zerve API that accepts any prediction market question and returns a reliability score, confidence tier, top contributing factors, and a Claude-powered natural language explanation — turning raw market data into actionable intelligence.

---

## Built With

Zerve AI, Python, scikit-learn, Plotly, Anthropic Claude API, Metaculus API, FRED API, pandas, numpy

---

## Social Post (for LinkedIn/X)

"Can AI predict which prediction markets to trust? 🎯 Built PredictPulse for @ZerveHack — an ML pipeline that scores forecast reliability using 500+ data points and Claude AI analysis. Deployed as a live API on @Zerve_AI. Check it out: [ZERVE_PROJECT_URL] #ZerveHack #DataScience #AI"
