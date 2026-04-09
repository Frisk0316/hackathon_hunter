"""
PredictPulse — Core Pipeline (AlgoFest version)
================================================
Self-contained ML pipeline. No platform lock-in.
Runs locally or on any Python environment.
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def generate_synthetic_data(n_resolved=500, n_open=80, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["Science", "Economics", "Politics", "Technology",
                  "Health", "Environment", "Sports", "Finance"]
    cat_w = [0.20, 0.18, 0.16, 0.14, 0.10, 0.08, 0.08, 0.06]
    from datetime import datetime, timedelta
    base = datetime(2021, 1, 1)

    def make(n, status):
        rows = []
        for i in range(n):
            lifespan = int(rng.integers(30, 730))
            created  = base + timedelta(days=int(rng.integers(0, 1000)))
            close    = created + timedelta(days=lifespan)
            resolve  = close + timedelta(days=int(rng.integers(0, 30)))
            n_preds  = int(np.clip(rng.lognormal(3.5, 1.2), 3, 2000))
            n_coms   = int(np.clip(rng.lognormal(2.0, 1.0), 0, 300))
            cp       = float(np.clip(rng.beta(2, 2), 0.02, 0.98))
            density  = n_preds / (lifespan + 1)
            conf     = abs(cp - 0.5) * 2
            acc_base = 0.55 + 0.15 * np.log1p(density) / 5 - 0.05 * conf
            is_acc   = rng.random() < np.clip(acc_base, 0.3, 0.85)
            res      = (cp > 0.5) if is_acc else (cp <= 0.5)
            cat      = str(rng.choice(categories, p=cat_w))
            desc_len = int(np.clip(rng.lognormal(5.5, 0.8), 100, 5000))
            templates = [
                f"Will {cat.lower()} indicator exceed threshold by {resolve.year}?",
                f"Will the {cat.lower()} sector show growth in {resolve.strftime('%B %Y')}?",
                f"Is the probability of {cat.lower()} event #{i} greater than 50%?",
                f"Will {cat.lower()} policy change before {resolve.strftime('%b %Y')}?",
            ]
            title = str(rng.choice(templates))
            rows.append({
                "id": 10000 + i, "title": title,
                "created_time": created.isoformat(),
                "resolve_time":  resolve.isoformat(),
                "close_time":    close.isoformat(),
                "prediction_count": n_preds,
                "community_prediction": cp,
                "resolution": str(float(res)) if status == "resolved" else None,
                "category": cat, "description_length": desc_len,
                "num_comments": n_coms,
                "url": f"https://www.metaculus.com/questions/{10000 + i}/",
            })
        return pd.DataFrame(rows)

    return make(n_resolved, "resolved"), make(n_open, "open")


def fetch_metaculus(limit=500, status="resolved"):
    """Try Metaculus API with token; fall back to synthetic data."""
    import requests, time
    token = os.environ.get("METACULUS_TOKEN", "")
    if not token:
        return None
    headers = {"Authorization": f"Token {token}"}
    base_url = "https://www.metaculus.com/api2/questions/"
    rows, offset = [], 0
    while offset < limit:
        params = {"limit": min(100, limit - offset), "offset": offset,
                  "status": status, "type": "forecast", "order_by": "-resolve_time"}
        r = requests.get(base_url, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            break
        for q in results:
            cp = (q.get("community_prediction") or {}).get("full", {}).get("q2")
            cat = "Unknown"
            for p in (q.get("projects") or []):
                if isinstance(p, dict) and p.get("type") == "category":
                    cat = p.get("name", "Unknown"); break
            rows.append({
                "id": q.get("id"), "title": q.get("title", ""),
                "created_time": q.get("created_time"), "resolve_time": q.get("resolve_time"),
                "close_time": q.get("close_time"), "prediction_count": q.get("prediction_count", 0),
                "community_prediction": cp, "resolution": q.get("resolution"),
                "category": cat, "description_length": len(q.get("description", "")),
                "num_comments": q.get("comment_count", 0),
                "url": f"https://www.metaculus.com/questions/{q.get('id')}/",
            })
        offset += 100; time.sleep(0.5)
    return pd.DataFrame(rows) if rows else None


def load_data():
    resolved = fetch_metaculus(500, "resolved")
    if resolved is None or len(resolved) == 0:
        resolved, open_df = generate_synthetic_data()
        source = "synthetic"
    else:
        open_df = fetch_metaculus(80, "open") or generate_synthetic_data(0, 80)[1]
        source = "metaculus"
    return resolved, open_df, source


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(resolved_df, open_df=None):
    from datetime import datetime

    def featurize(df, is_resolved=True):
        d = df.copy()
        d["created_time"] = pd.to_datetime(d["created_time"], errors="coerce").dt.tz_localize(None)
        d["resolve_time"]  = pd.to_datetime(d["resolve_time"],  errors="coerce").dt.tz_localize(None)
        d["close_time"]    = pd.to_datetime(d["close_time"],    errors="coerce").dt.tz_localize(None)
        now = pd.Timestamp.now()

        d["question_lifespan_days"] = ((d["resolve_time"] - d["created_time"])
                                       .dt.total_seconds() / 86400).fillna(90)
        d["time_to_close_days"]     = ((d["close_time"]   - d["created_time"])
                                       .dt.total_seconds() / 86400).fillna(90)
        d["resolve_month"]     = d["resolve_time"].dt.month.fillna(now.month)
        d["resolve_dayofweek"] = d["resolve_time"].dt.dayofweek.fillna(now.dayofweek)

        d["log_prediction_count"] = np.log1p(d["prediction_count"].fillna(0))
        d["log_comments"]         = np.log1p(d["num_comments"].fillna(0))
        d["engagement_ratio"]     = d["num_comments"] / (d["prediction_count"] + 1)
        d["prediction_density"]   = d["prediction_count"] / (d["question_lifespan_days"] + 1)

        d["log_description_length"] = np.log1p(d["description_length"].fillna(0))
        d["title_length"]     = d["title"].str.len().fillna(0)
        d["title_word_count"] = d["title"].str.split().str.len().fillna(0)
        d["has_number_in_title"] = d["title"].str.contains(r"\d", regex=True).astype(int)
        d["is_yes_no_question"]  = d["title"].str.lower().str.contains(
            r"\bwill\b|\bwould\b|\bshould\b", regex=True).astype(int)

        d["community_pred"]  = pd.to_numeric(d["community_prediction"], errors="coerce").fillna(0.5)
        d["pred_confidence"] = (d["community_pred"] - 0.5).abs() * 2
        d["pred_extreme"]    = ((d["community_pred"] < 0.1) | (d["community_pred"] > 0.9)).astype(int)

        # Category dummies (top 8)
        top_cats = d["category"].value_counts().head(8).index.tolist()
        d["category_clean"] = d["category"].apply(lambda x: x if x in top_cats else "Other")
        dummies = pd.get_dummies(d["category_clean"], prefix="cat")
        d = pd.concat([d, dummies], axis=1)

        if is_resolved:
            d["resolution_float"] = pd.to_numeric(d["resolution"], errors="coerce")
            mask = d["resolution_float"].isin([0.0, 1.0])
            d = d[mask].copy()
            d["prediction_error"]    = (d["community_pred"] - d["resolution_float"]).abs()
            d["accuracy_score"]      = 1 - d["prediction_error"]
            d["prediction_accurate"] = (d["prediction_error"] < 0.3).astype(int)
        return d

    resolved_feat = featurize(resolved_df, is_resolved=True)

    base_feat_cols = [
        "question_lifespan_days", "time_to_close_days", "resolve_month", "resolve_dayofweek",
        "log_prediction_count", "log_comments", "engagement_ratio", "prediction_density",
        "log_description_length", "title_length", "title_word_count",
        "has_number_in_title", "is_yes_no_question",
        "community_pred", "pred_confidence", "pred_extreme",
    ]
    cat_cols = [c for c in resolved_feat.columns if c.startswith("cat_")]
    feature_cols = base_feat_cols + cat_cols

    open_feat = None
    if open_df is not None:
        open_feat = featurize(open_df, is_resolved=False)
        for col in feature_cols:
            if col not in open_feat.columns:
                open_feat[col] = 0

    return resolved_feat, open_feat, feature_cols


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────

def train_model(df, feature_cols):
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, classification_report)
    from sklearn.calibration import calibration_curve

    X = df[feature_cols].fillna(0).values
    y = df["prediction_accurate"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, random_state=42),
    }

    cv_scores = {}
    for name, m in candidates.items():
        auc = cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()
        cv_scores[name] = auc

    best_name = max(cv_scores, key=cv_scores.get)
    model = candidates[best_name]
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]

    # Calibration curve
    cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    frac_pos, mean_pred = calibration_curve(y, cv_proba, n_bins=8, strategy="quantile")
    cal_df = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})

    # Brier score
    brier = brier_score_loss(y, y_proba)
    baseline_brier = y.mean() * (1 - y.mean())
    brier_skill = 1 - brier / baseline_brier

    # Feature importances
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        imps = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else np.zeros(len(feature_cols))
    else:
        imps = np.zeros(len(feature_cols))

    importance_df = pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values(
        "importance", ascending=False)

    # Category breakdown
    df_eval = df.copy()
    df_eval["proba"] = y_proba
    cat_stats = {}
    for cat, grp in df_eval.groupby("category_clean", observed=True):
        if len(grp) < 5:
            continue
        cat_stats[str(cat)] = {
            "n": len(grp),
            "mean_accuracy_score": float(grp["accuracy_score"].mean()),
            "brier": float(brier_score_loss(grp["prediction_accurate"], grp["proba"])),
        }

    metrics = {
        "best_model": best_name,
        "cv_auc": cv_scores,
        "auc_roc": float(roc_auc_score(y, y_proba)),
        "avg_precision": float(average_precision_score(y, y_proba)),
        "brier_score": float(brier),
        "brier_skill": float(brier_skill),
        "calibration_df": cal_df,
        "category_stats": cat_stats,
        "n_train": len(X),
        "n_features": len(feature_cols),
    }
    return model, metrics, importance_df


def score_questions(model, open_df, feature_cols):
    X = open_df[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    out = open_df.copy()
    out["reliability_score"] = proba
    out["reliability_tier"] = pd.cut(
        proba, bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"])
    return out.sort_values("reliability_score", ascending=False)


# ─────────────────────────────────────────────
# AI Analysis (Claude)
# ─────────────────────────────────────────────

def get_ai_analysis(title, score, community_pred, n_preds, category):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    tier = ("Very High" if score > 0.85 else "High" if score > 0.7 else
            "Medium" if score > 0.5 else "Low" if score > 0.3 else "Very Low")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": (
                    f"You are PredictPulse AI. Briefly analyze this prediction's reliability "
                    f"(2-3 sentences, be direct and specific):\n\n"
                    f"Question: {title}\n"
                    f"Community prediction: {community_pred:.0%}\n"
                    f"Reliability score: {score:.1%} ({tier})\n"
                    f"Forecasters: {n_preds} | Category: {category}"
                )}],
            )
            return msg.content[0].text
        except Exception:
            pass
    # Fallback
    crowd = "large" if n_preds > 100 else "moderate" if n_preds > 30 else "small"
    return (f"**{tier} reliability** — {score:.0%} confidence score based on {n_preds} forecasters "
            f"({crowd} crowd). Community consensus at {community_pred:.0%}. "
            f"{'High participation suggests reliable crowd wisdom.' if n_preds > 100 else 'Limited forecaster count reduces reliability.'}")
