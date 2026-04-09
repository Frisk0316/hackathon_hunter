# Demo Video Script (Under 3 Minutes)

## [0:00 - 0:15] Hook

**Visual:** PredictPulse logo animation → prediction market screenshots (Metaculus, Polymarket)

**Narration:** "Every day, thousands of predictions are made on markets like Metaculus and Polymarket. But how do you know which ones to trust? PredictPulse uses AI and machine learning to answer that question."

---

## [0:15 - 0:30] The Problem

**Visual:** Side-by-side comparison — two predictions both showing ~65%, but one is based on 12 forecasters and the other on 300

**Narration:** "These two predictions look identical — both around 65%. But one has 12 forecasters and the other 300. One is in a category where crowds are historically accurate, and the other isn't. Current platforms show you the number, not the reliability."

---

## [0:30 - 1:00] Zerve Workflow — Data Collection

**Visual:** Screen recording of Zerve canvas. Show Block 1 running — API calls fetching Metaculus data, FRED economic indicators loading.

**Narration:** "PredictPulse starts by collecting data. Block 1 pulls 500+ resolved predictions from Metaculus, economic indicators from FRED — GDP, unemployment, VIX — and social trend signals. All automated within Zerve's notebook environment."

**Action:** Show data preview output — "Fetched 500 Metaculus questions"

---

## [1:00 - 1:30] Feature Engineering + Model Training

**Visual:** Block 2 running → feature engineering output. Block 3 running → cross-validation results table, feature importance chart.

**Narration:** "Block 2 engineers 20+ features — participation density, confidence levels, question complexity, economic context. Block 3 trains three ML models and picks the best one. Our Gradient Boosting classifier identifies that prediction density — how many forecasters per day — is the strongest accuracy signal."

**Action:** Show the feature importance bar chart (Plotly)

---

## [1:30 - 2:00] Visualization + AI Analysis

**Visual:** Block 4 → interactive Plotly dashboards appear (accuracy landscape, category analysis). Block 5 → Claude-generated report text streaming.

**Narration:** "Block 4 creates interactive dashboards showing accuracy patterns across categories and time. Block 5 brings in Claude AI — for each scored prediction, Claude generates a plain-English analysis explaining why a prediction is or isn't reliable, including caveats and confidence levels."

**Action:** Show the reliability scorecard chart, then scroll through Claude's analysis text

---

## [2:00 - 2:30] API Deployment + Live Demo

**Visual:** Block 6 → test API call → JSON response appearing

**Narration:** "Finally, Block 6 deploys everything as a live API on Zerve. Watch — I send a question: 'Will global temperature exceed 1.5°C before 2030?' Community says 62%. PredictPulse scores it at 82% reliability — High tier — and explains exactly why."

**Action:** Show the full API JSON response with reliability_score, tier, analysis, and top_factors

---

## [2:30 - 2:50] Key Insights + Impact

**Visual:** Summary slide with key findings (bullet points with icons)

**Narration:** "Our key finding: not all crowds are equal. Predictions with high participation density and moderate confidence are significantly more reliable than those with extreme consensus or low engagement. This pipeline could improve prediction-based decisions by 15 to 25 percent."

---

## [2:50 - 3:00] Close

**Visual:** PredictPulse logo + "Built with Zerve AI + Claude API"

**Narration:** "PredictPulse — turning crowd wisdom into actionable intelligence. Built entirely on the Zerve platform."

**Text on screen:** Zerve Project URL | GitHub URL
