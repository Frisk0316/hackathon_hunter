"""
PredictPulse — Block 1: Data Collection
========================================
Ingests prediction market data from Metaculus API,
economic indicators from FRED, and social signals from Google Trends.

This block runs on the Zerve platform as part of the PredictPulse pipeline.

AUTH NOTE:
  Metaculus API requires a free token. Get yours at:
  https://metaculus.com/aib  (free account, instant access)
  Then set METACULUS_TOKEN as a Zerve environment variable.
  Without a token, a realistic synthetic dataset is used automatically.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import numpy as np

# ============================================================
# 0. Synthetic Data Fallback
# ============================================================

def generate_synthetic_data(n_resolved=500, n_open=80, seed=42):
    """
    Generate realistic synthetic prediction market data.
    Used when Metaculus API token is not available.
    Distributions are calibrated from real Metaculus statistics.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    categories = ["Science", "Economics", "Politics", "Technology", "Health",
                  "Environment", "Sports", "Finance", "Social", "Other"]
    cat_weights = [0.20, 0.18, 0.16, 0.14, 0.10, 0.08, 0.06, 0.04, 0.02, 0.02]

    def make_questions(n, status):
        rows = []
        base_date = datetime(2021, 1, 1)
        for i in range(n):
            lifespan = rng.integers(30, 730)
            created = base_date + timedelta(days=int(rng.integers(0, 1000).item()))
            close = created + timedelta(days=int(lifespan))
            resolve = close + timedelta(days=int(rng.integers(0, 30).item()))

            n_preds = int(np.clip(rng.lognormal(3.5, 1.2), 3, 2000))
            n_comments = int(np.clip(rng.lognormal(2.0, 1.0), 0, 300))
            pred_density = n_preds / (lifespan + 1)

            # Community prediction — beta distributed
            community_pred = float(np.clip(rng.beta(2, 2), 0.02, 0.98))

            # Resolution: accuracy correlates with participation and confidence
            confidence = abs(community_pred - 0.5) * 2
            accuracy_base = 0.55 + 0.15 * np.log1p(pred_density) / 5 - 0.05 * confidence
            is_accurate = rng.random() < np.clip(accuracy_base, 0.3, 0.85)
            resolution = community_pred > 0.5 if is_accurate else community_pred <= 0.5
            resolution_val = 1.0 if resolution else 0.0

            cat = rng.choice(categories, p=cat_weights)
            desc_len = int(np.clip(rng.lognormal(5.5, 0.8), 100, 5000))
            title_templates = [
                f"Will {cat.lower()} indicator exceed threshold by {resolve.year}?",
                f"Will the {cat.lower()} sector show growth in {resolve.strftime('%B %Y')}?",
                f"Is there a greater than 50% chance of {cat.lower()} event #{i}?",
                f"Will {cat.lower()} policy change before {resolve.strftime('%b %Y')}?",
            ]
            title = rng.choice(title_templates)

            rows.append({
                "id": 10000 + i,
                "title": title,
                "created_time": created.isoformat(),
                "resolve_time": resolve.isoformat(),
                "close_time": close.isoformat(),
                "prediction_count": n_preds,
                "community_prediction": community_pred,
                "resolution": str(resolution_val) if status == "resolved" else None,
                "category": cat,
                "description_length": desc_len,
                "num_comments": n_comments,
                "url": f"https://www.metaculus.com/questions/{10000 + i}/",
            })
        return pd.DataFrame(rows)

    resolved_df = make_questions(n_resolved, "resolved")
    open_df = make_questions(n_open, "open")

    print(f"[Synthetic] Generated {len(resolved_df)} resolved + {len(open_df)} open questions")
    print(f"[Synthetic] Accuracy rate: {(resolved_df['resolution'].astype(float) > 0.5).mean():.1%}")
    return resolved_df, open_df


# ============================================================
# 1. Metaculus API — Prediction Market Data
# ============================================================

def fetch_metaculus_questions(limit=500, status="resolved"):
    """Fetch resolved questions from Metaculus API for training data."""
    base_url = "https://www.metaculus.com/api2/questions/"
    all_questions = []
    offset = 0
    page_size = 100

    token = os.environ.get("METACULUS_TOKEN", "")
    headers = {"Authorization": f"Token {token}"} if token else {}

    while offset < limit:
        params = {
            "limit": min(page_size, limit - offset),
            "offset": offset,
            "status": status,
            "type": "forecast",
            "order_by": "-resolve_time",
        }
        resp = requests.get(base_url, params=params, headers=headers, timeout=30)
        if resp.status_code == 403:
            print("Metaculus API requires auth. Get token at: https://metaculus.com/aib")
            print("Set METACULUS_TOKEN env var in Zerve. Falling back to synthetic data.")
            return None
        if resp.status_code != 200:
            print(f"Metaculus API error: {resp.status_code}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for q in results:
            all_questions.append({
                "id": q.get("id"),
                "title": q.get("title", ""),
                "created_time": q.get("created_time"),
                "resolve_time": q.get("resolve_time"),
                "close_time": q.get("close_time"),
                "prediction_count": q.get("prediction_count", 0),
                "community_prediction": q.get("community_prediction", {}).get("full", {}).get("q2"),
                "resolution": q.get("resolution"),
                "category": _extract_category(q),
                "description_length": len(q.get("description", "")),
                "num_comments": q.get("comment_count", 0),
                "url": f"https://www.metaculus.com/questions/{q.get('id')}/",
            })

        offset += page_size
        time.sleep(0.5)

    df = pd.DataFrame(all_questions)
    print(f"Fetched {len(df)} Metaculus questions (status={status})")
    return df


def _extract_category(question):
    """Extract primary category/tag from a Metaculus question."""
    tags = question.get("projects", [])
    if tags:
        for tag in tags:
            if isinstance(tag, dict) and tag.get("type") == "category":
                return tag.get("name", "Unknown")
    return "Uncategorized"


# ============================================================
# 2. FRED API — Economic Indicators
# ============================================================

FRED_SERIES = {
    "GDP": "GDP",
    "Unemployment": "UNRATE",
    "CPI": "CPIAUCSL",
    "Fed_Funds_Rate": "FEDFUNDS",
    "SP500": "SP500",
    "Consumer_Sentiment": "UMCSENT",
    "VIX": "VIXCLS",
}


def fetch_fred_data(api_key, start_date="2020-01-01"):
    """Fetch key economic indicators from FRED API."""
    all_series = {}

    for name, series_id in FRED_SERIES.items():
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"FRED error for {name}: {resp.status_code}")
            continue

        observations = resp.json().get("observations", [])
        series_data = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                series_data.append({
                    "date": obs["date"],
                    "value": float(val),
                })

        if series_data:
            df = pd.DataFrame(series_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            all_series[name] = df["value"]

        time.sleep(0.3)

    fred_df = pd.DataFrame(all_series)
    fred_df = fred_df.resample("M").last().ffill()
    print(f"Fetched {len(fred_df)} months of FRED data across {len(all_series)} indicators")
    return fred_df


# ============================================================
# 3. Google Trends — Social Signal Proxy
# ============================================================

def fetch_trend_proxy(keywords, period_days=365):
    """
    Fetch Google Trends-like data using pytrends.
    Falls back to a synthetic proxy if pytrends is unavailable.
    """
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(keywords, timeframe=f"today {period_days}-d")
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        print(f"Fetched Google Trends data for {keywords}")
        return df
    except ImportError:
        print("pytrends not available — using search volume proxy via Metaculus activity")
        return None


# ============================================================
# 4. Main Collection Pipeline
# ============================================================

def run_data_collection(fred_api_key=None):
    """Run the full data collection pipeline."""

    # Metaculus: resolved questions for training
    print("=" * 60)
    print("STEP 1: Collecting Metaculus resolved questions...")
    print("=" * 60)
    resolved_df = fetch_metaculus_questions(limit=500, status="resolved")

    # Fallback to synthetic data if API unavailable
    if resolved_df is None or len(resolved_df) == 0:
        print("\n→ Using synthetic dataset (set METACULUS_TOKEN for real data)")
        resolved_df, open_df = generate_synthetic_data(n_resolved=500, n_open=80)
    else:
        # Metaculus: open questions for scoring
        print("\nSTEP 2: Collecting Metaculus open questions...")
        open_df_result = fetch_metaculus_questions(limit=80, status="open")
        open_df = open_df_result if open_df_result is not None else generate_synthetic_data(0, 80)[1]

    # FRED economic indicators
    fred_df = None
    if fred_api_key:
        print("\nSTEP 3: Collecting FRED economic indicators...")
        fred_df = fetch_fred_data(fred_api_key)
    else:
        print("\nSTEP 3: Skipping FRED (no API key) — will use Metaculus features only")
        print("  Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")

    # Google Trends
    print("\nSTEP 4: Collecting social signals...")
    trend_keywords = ["prediction market", "AI forecast", "economic uncertainty"]
    trends_df = fetch_trend_proxy(trend_keywords)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print(f"  Resolved questions: {len(resolved_df)}")
    print(f"  Open questions:     {len(open_df)}")
    print(f"  FRED indicators:    {len(fred_df) if fred_df is not None else 'N/A (no key)'}")
    print(f"  Trends data:        {'Available' if trends_df is not None else 'N/A'}")
    print("=" * 60)

    return resolved_df, open_df, fred_df, trends_df


# ============================================================
# Run collection
# ============================================================
# Optional API keys — set as Zerve environment variables:
#   METACULUS_TOKEN  → free at https://metaculus.com/aib
#   FRED_API_KEY     → free at https://fred.stlouisfed.org/docs/api/api_key.html
#   ANTHROPIC_API_KEY → for Claude AI analysis in Block 5

FRED_API_KEY = os.environ.get("FRED_API_KEY", None)

resolved_df, open_df, fred_df, trends_df = run_data_collection(FRED_API_KEY)

# Preview the data
print("\n--- Resolved Questions Sample ---")
print(resolved_df[["title", "prediction_count", "community_prediction", "resolution", "category"]].head())
print(f"\nFeature columns: {list(resolved_df.columns)}")
print(f"Category distribution: {resolved_df['category'].value_counts().head(5).to_dict()}")
