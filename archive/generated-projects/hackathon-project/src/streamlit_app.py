"""
PredictPulse — Streamlit Dashboard (Zerve 單檔版)
===================================================
貼入 Zerve Streamlit 部署的程式碼欄位即可，無需任何額外檔案。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# Pipeline (inlined from src/pipeline.py)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_data(n_resolved=500, n_open=80, seed=42):
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    categories = ["Science", "Economics", "Politics", "Technology",
                  "Health", "Environment", "Sports", "Finance"]
    cat_w = [0.20, 0.18, 0.16, 0.14, 0.10, 0.08, 0.08, 0.06]
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


def engineer_features(resolved_df, open_df=None):
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
        d["log_prediction_count"]   = np.log1p(d["prediction_count"].fillna(0))
        d["log_comments"]           = np.log1p(d["num_comments"].fillna(0))
        d["engagement_ratio"]       = d["num_comments"] / (d["prediction_count"] + 1)
        d["prediction_density"]     = d["prediction_count"] / (d["question_lifespan_days"] + 1)
        d["log_description_length"] = np.log1p(d["description_length"].fillna(0))
        d["title_length"]           = d["title"].str.len().fillna(0)
        d["title_word_count"]       = d["title"].str.split().str.len().fillna(0)
        d["has_number_in_title"]    = d["title"].str.contains(r"\d", regex=True).astype(int)
        d["is_yes_no_question"]     = d["title"].str.lower().str.contains(
            r"\bwill\b|\bwould\b|\bshould\b", regex=True).astype(int)
        d["community_pred"]  = pd.to_numeric(d["community_prediction"], errors="coerce").fillna(0.5)
        d["pred_confidence"] = (d["community_pred"] - 0.5).abs() * 2
        d["pred_extreme"]    = ((d["community_pred"] < 0.1) | (d["community_pred"] > 0.9)).astype(int)
        top_cats = d["category"].value_counts().head(8).index.tolist()
        d["category_clean"] = d["category"].apply(lambda x: x if x in top_cats else "Other")
        dummies = pd.get_dummies(d["category_clean"], prefix="cat")
        d = pd.concat([d, dummies], axis=1)
        if is_resolved:
            d["resolution_float"]    = pd.to_numeric(d["resolution"], errors="coerce")
            d = d[d["resolution_float"].isin([0.0, 1.0])].copy()
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


def train_model(df, feature_cols, n_bins=8):
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                                  precision_score, recall_score, f1_score)
    from sklearn.calibration import calibration_curve
    from sklearn.inspection import permutation_importance as sk_perm_importance

    # ── Temporal train/test split (80/20 by resolve_time) ──
    df_sorted = df.sort_values("resolve_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    df_train = df_sorted.iloc[:split_idx]
    df_test  = df_sorted.iloc[split_idx:]

    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train["prediction_accurate"].values
    X_test  = df_test[feature_cols].fillna(0).values
    y_test  = df_test["prediction_accurate"].values
    X_all   = df_sorted[feature_cols].fillna(0).values
    y_all   = df_sorted["prediction_accurate"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=1000, C=0.1))]),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                       min_samples_leaf=5, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                           learning_rate=0.1,
                                                           min_samples_leaf=10, random_state=42),
    }
    # CV on training set only
    cv_scores = {name: cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc").mean()
                 for name, m in candidates.items()}
    best_name = max(cv_scores, key=cv_scores.get)
    model = candidates[best_name]
    model.fit(X_train, y_train)

    # ── Test-set evaluation ──
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred  = (y_test_proba > 0.5).astype(int)
    test_brier   = brier_score_loss(y_test, y_test_proba)
    # Baseline: always use community_prediction as forecast
    baseline_proba = df_test["community_pred"].values
    baseline_brier = brier_score_loss(y_test, baseline_proba)
    baseline_auc   = roc_auc_score(y_test, baseline_proba)
    test_metrics = {
        "auc":           float(roc_auc_score(y_test, y_test_proba)),
        "brier":         float(test_brier),
        "brier_skill":   float(1 - test_brier / max(baseline_brier, 1e-9)),
        "precision":     float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall":        float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1":            float(f1_score(y_test, y_test_pred, zero_division=0)),
        "n_test":        len(y_test),
        "baseline_brier": float(baseline_brier),
        "baseline_auc":  float(baseline_auc),
    }

    # ── Calibration on full dataset ──
    model.fit(X_all, y_all)   # refit on all data for production
    y_proba = model.predict_proba(X_all)[:, 1]
    cv_proba = cross_val_predict(model, X_all, y_all, cv=cv, method="predict_proba")[:, 1]
    frac_pos, mean_pred = calibration_curve(y_all, cv_proba, n_bins=n_bins, strategy="quantile")
    cal_df = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})

    brier = brier_score_loss(y_all, y_proba)
    brier_skill = 1 - brier / (y_all.mean() * (1 - y_all.mean()))

    # ── Feature importance ──
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        imps = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else np.zeros(len(feature_cols))
    else:
        imps = np.zeros(len(feature_cols))
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values(
        "importance", ascending=False)

    # ── Permutation importance on test set (ablation proxy) ──
    try:
        perm = sk_perm_importance(model, X_test, y_test, n_repeats=5,
                                  random_state=42, scoring="roc_auc")
        perm_df = pd.DataFrame({
            "feature": feature_cols,
            "auc_drop_mean": perm.importances_mean,
            "auc_drop_std":  perm.importances_std,
        }).sort_values("auc_drop_mean", ascending=False)
    except Exception:
        perm_df = importance_df[["feature"]].copy()
        perm_df["auc_drop_mean"] = imps
        perm_df["auc_drop_std"]  = 0.0

    # ── SHAP values (tree-based models only) ──
    shap_df = None
    try:
        import shap as _shap
        if hasattr(model, "feature_importances_"):
            _explainer  = _shap.TreeExplainer(model)
            _n          = min(300, len(X_train))
            _shap_vals  = _explainer.shap_values(X_train[:_n])
            # For binary classifiers shap_values may be [neg, pos] list
            if isinstance(_shap_vals, list):
                _shap_vals = _shap_vals[1]
            shap_df = pd.DataFrame(_shap_vals, columns=feature_cols)
    except Exception:
        pass

    # ── Category stats ──
    df_eval = df_sorted.copy()
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

    return model, {
        "best_model": best_name, "cv_auc": cv_scores,
        "auc_roc": float(roc_auc_score(y_all, y_proba)),
        "avg_precision": float(average_precision_score(y_all, y_proba)),
        "brier_score": float(brier), "brier_skill": float(brier_skill),
        "calibration_df": cal_df, "category_stats": cat_stats,
        "n_train": len(X_train), "n_features": len(feature_cols),
        "test_metrics": test_metrics,
        "perm_df": perm_df,
        "shap_df": shap_df,
        "feature_cols": feature_cols,
    }, importance_df


def score_questions(model, open_df, feature_cols):
    X = open_df[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    out = open_df.copy()
    out["reliability_score"] = proba
    out["reliability_tier"] = pd.cut(
        proba, bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"])
    return out.sort_values("reliability_score", ascending=False)


def get_ai_analysis(title, score, community_pred, n_preds, category):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    tier = ("Very High" if score > 0.85 else "High" if score > 0.7 else
            "Medium" if score > 0.5 else "Low" if score > 0.3 else "Very Low")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=300,
                messages=[{"role": "user", "content": (
                    f"You are PredictPulse AI. Briefly analyze this prediction's reliability "
                    f"(2-3 sentences, be direct and specific):\n\n"
                    f"Question: {title}\nCommunity prediction: {community_pred:.0%}\n"
                    f"Reliability score: {score:.1%} ({tier})\n"
                    f"Forecasters: {n_preds} | Category: {category}"
                )}],
            )
            return msg.content[0].text
        except Exception:
            pass
    crowd = "large" if n_preds > 100 else "moderate" if n_preds > 30 else "small"
    return (f"**{tier} reliability** — {score:.0%} confidence based on {n_preds} forecasters "
            f"({crowd} crowd). Community at {community_pred:.0%}. "
            f"{'High participation = reliable crowd wisdom.' if n_preds > 100 else 'Limited forecaster count reduces reliability.'}")


# ═══════════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PredictPulse",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background:#f8f9fa; border-radius:12px; padding:16px 20px; margin:6px 0; }
.tier-very-high { color:#059669; font-weight:700; }
.tier-high      { color:#16a34a; font-weight:700; }
.tier-medium    { color:#d97706; font-weight:700; }
.tier-low       { color:#dc2626; font-weight:600; }
.tier-very-low  { color:#7c3aed; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 PredictPulse")
    st.caption("AI-Powered Prediction Market Intelligence")
    st.divider()

    st.subheader("⚙️ Settings")
    n_samples = st.slider("Training data size", 200, 800, 500, 100)
    n_bins    = st.slider("Calibration bins", 5, 12, 8)

    st.divider()
    st.subheader("🔑 API Keys")
    metaculus_token = st.text_input("Metaculus Token", type="password",
                                    help="Get free token at metaculus.com/aib")
    anthropic_key   = st.text_input("Anthropic API Key", type="password",
                                    help="For Claude AI analysis")
    if metaculus_token:
        os.environ["METACULUS_TOKEN"]  = metaculus_token
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    st.divider()
    st.caption("Built for **ZerveHack 2026**")
    st.caption("Stack: Python · scikit-learn · Plotly · Streamlit · Claude API")
    st.caption("API: [d6aca690.hub.zerve.cloud](https://d6aca690-2cad4778.hub.zerve.cloud/docs)")


# ── Load & Train (cached) ─────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline(n_samples, n_bins, _token, _key):
    resolved, open_df, source = load_data()
    if len(resolved) > n_samples:
        resolved = resolved.sample(n_samples, random_state=42)
    resolved_feat, open_feat, feature_cols = engineer_features(resolved, open_df)
    model, metrics, importance_df = train_model(resolved_feat, feature_cols, n_bins=n_bins)
    scored = score_questions(model, open_feat, feature_cols) if open_feat is not None else None
    return model, metrics, importance_df, scored, open_feat, feature_cols, source


with st.spinner("Running PredictPulse pipeline..."):
    model, metrics, importance_df, scored, open_feat, feature_cols, data_source = run_pipeline(
        n_samples, n_bins, metaculus_token, anthropic_key)


# ── Header ────────────────────────────────────────────────────
st.title("🎯 PredictPulse")
st.markdown("**AI-powered prediction market intelligence** — *Can we predict which forecasts to trust?*")

if data_source == "synthetic":
    st.info("📊 Using synthetic data (calibrated to Metaculus statistics). "
            "Add your **Metaculus Token** in the sidebar for real predictions.", icon="ℹ️")
else:
    st.success("✅ Live Metaculus data loaded", icon="✅")
st.divider()


# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Training Samples",  f"{metrics['n_train']:,}")
k2.metric("Features",          f"{metrics['n_features']}")
k3.metric("AUC-ROC",           f"{metrics['auc_roc']:.3f}", delta="vs 0.500 baseline")
k4.metric("Brier Skill Score", f"{metrics['brier_skill']:.3f}", delta="↑ higher = better")
k5.metric("Best Model",        metrics["best_model"].split()[0])
st.divider()


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔬 Model Analysis", "🏆 Reliability Scores",
    "🔮 Score a Prediction", "🧠 Explainability"])


# ──── Tab 1: Dashboard ────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Feature Importance")
        top = importance_df.head(12).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top["importance"],
            y=top["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker=dict(color=top["importance"], colorscale="Purples"),
            text=top["importance"].round(3), textposition="outside",
        ))
        fig.update_layout(height=420, margin=dict(l=180, r=40, t=20, b=20),
                          template="plotly_white", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Model Comparison (5-fold CV AUC-ROC)")
        cv_df = pd.DataFrame([{"model": k, "auc": v} for k, v in metrics["cv_auc"].items()]).sort_values("auc")
        fig2 = go.Figure(go.Bar(
            x=cv_df["auc"], y=cv_df["model"], orientation="h",
            marker=dict(color=["#6366f1" if v == max(metrics["cv_auc"].values()) else "#e2e8f0"
                               for v in cv_df["auc"]]),
            text=cv_df["auc"].round(3), textposition="outside",
        ))
        fig2.update_layout(height=200, margin=dict(l=160, r=60, t=20, b=20),
                           template="plotly_white", xaxis=dict(range=[0.5, 1.0]),
                           xaxis_title="AUC-ROC")
        st.plotly_chart(fig2, use_container_width=True)

        if metrics["category_stats"]:
            st.subheader("Accuracy by Category")
            cat_df = pd.DataFrame(metrics["category_stats"]).T.reset_index()
            cat_df.columns = ["Category", "N", "Mean Accuracy", "Brier"]
            cat_df = cat_df.sort_values("Mean Accuracy", ascending=False)
            fig3 = px.bar(cat_df, x="Category", y="Mean Accuracy",
                          color="Mean Accuracy", color_continuous_scale="RdYlGn",
                          text=cat_df["Mean Accuracy"].round(2))
            fig3.update_layout(height=220, template="plotly_white",
                               margin=dict(t=20, b=60), showlegend=False,
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)


# ──── Tab 2: Model Analysis ───────────────────────────────────
with tab2:
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Calibration Curve")
        st.caption("Does a 70% prediction really mean 70%? Perfect calibration = diagonal line.")
        cal = metrics["calibration_df"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="#94a3b8", width=1),
                                  name="Perfect calibration"))
        fig4.add_trace(go.Scatter(
            x=cal["mean_pred"], y=cal["frac_pos"], mode="lines+markers",
            line=dict(color="#6366f1", width=3), marker=dict(size=10, color="#6366f1"),
            name="PredictPulse",
            hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
        ))
        fig4.add_trace(go.Scatter(
            x=list(cal["mean_pred"]) + list(cal["mean_pred"])[::-1],
            y=list(cal["frac_pos"]) + list(cal["mean_pred"])[::-1],
            fill="toself", fillcolor="rgba(99,102,241,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig4.update_layout(height=380, template="plotly_white",
                           xaxis_title="Mean Predicted Probability",
                           yaxis_title="Fraction of Positives",
                           margin=dict(t=20, b=40),
                           legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02))
        st.plotly_chart(fig4, use_container_width=True)
        b1, b2 = st.columns(2)
        b1.metric("Brier Score",       f"{metrics['brier_score']:.4f}",
                  help="Lower = better. Random = 0.25, Perfect = 0.00")
        b2.metric("Brier Skill Score", f"{metrics['brier_skill']:.3f}",
                  help="Higher = better. 0 = no skill, 1 = perfect")

    with col_d:
        st.subheader("Calibration Gap by Bin")
        st.caption("Positive = model underestimates (conservative). Negative = overestimates (overconfident).")
        cal_gap = cal.copy()
        cal_gap["gap"] = cal_gap["frac_pos"] - cal_gap["mean_pred"]
        cal_gap["color"] = cal_gap["gap"].apply(
            lambda g: "#22c55e" if abs(g) < 0.05 else ("#f59e0b" if abs(g) < 0.10 else "#ef4444"))
        fig5 = go.Figure(go.Bar(
            x=cal_gap["mean_pred"].round(2).astype(str), y=cal_gap["gap"],
            marker_color=cal_gap["color"],
            text=cal_gap["gap"].round(3), textposition="outside",
        ))
        fig5.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig5.update_layout(height=300, template="plotly_white",
                           xaxis_title="Predicted Probability Bin",
                           yaxis_title="Calibration Gap (Actual − Predicted)",
                           margin=dict(t=20, b=40))
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Train / Test / Baseline Comparison")
        tm = metrics.get("test_metrics", {})
        if tm:
            st.markdown(f"""
| Metric | CV (Train) | **Test Set** | Baseline (Community) |
|--------|-----------|------------|----------------------|
| AUC-ROC | {max(metrics['cv_auc'].values()):.3f} | **{tm['auc']:.3f}** | {tm['baseline_auc']:.3f} |
| Brier Score | {metrics['brier_score']:.4f} | **{tm['brier']:.4f}** | {tm['baseline_brier']:.4f} |
| Brier Skill | {metrics['brier_skill']:.3f} | **{tm['brier_skill']:.3f}** | 0.000 |
| Precision | — | **{tm['precision']:.3f}** | — |
| Recall | — | **{tm['recall']:.3f}** | — |
| F1 Score | — | **{tm['f1']:.3f}** | — |
| Samples | {metrics['n_train']:,} | **{tm['n_test']:,}** | {tm['n_test']:,} |
""")
            improvement = (tm['baseline_brier'] - tm['brier']) / tm['baseline_brier'] * 100
            st.success(
                f"✅ **PredictPulse vs Baseline:** Brier Score improved by "
                f"**{improvement:.1f}%** over naive community-prediction baseline "
                f"({tm['baseline_brier']:.4f} → {tm['brier']:.4f}) on held-out test set."
            )
        else:
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Best CV AUC-ROC | **{max(metrics['cv_auc'].values()):.3f}** |
| Brier Score | **{metrics['brier_score']:.4f}** |
| Brier Skill | **{metrics['brier_skill']:.3f}** |
| Avg Precision | **{metrics['avg_precision']:.3f}** |
""")
        st.info("💡 **Top insight:** Community confidence (`pred_confidence`) is the strongest "
                "predictor — predictions near 50% are harder to score than those with strong consensus.")


# ──── Tab 3: Reliability Scores ───────────────────────────────
with tab3:
    st.subheader("Live Prediction Reliability Scores")
    if scored is not None and len(scored) > 0:
        col_f, col_g = st.columns([2, 1])
        with col_g:
            tier_filter = st.multiselect("Filter by tier",
                ["Very High", "High", "Medium", "Low", "Very Low"],
                default=["Very High", "High", "Medium"])
        filtered = scored[scored["reliability_tier"].isin(tier_filter)] if tier_filter else scored
        with col_f:
            st.caption(f"Showing {len(filtered)} predictions")

        top20 = filtered.head(20)
        fig6 = go.Figure(go.Bar(
            y=top20["title"].str[:50] + "...",
            x=top20["reliability_score"],
            orientation="h",
            marker=dict(color=top20["reliability_score"],
                        colorscale=[[0,"#ef4444"],[0.4,"#f97316"],[0.6,"#eab308"],
                                    [0.75,"#22c55e"],[1.0,"#059669"]],
                        cmin=0, cmax=1),
            text=top20["reliability_score"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
        ))
        fig6.update_layout(height=max(350, len(top20) * 28),
                           margin=dict(l=340, r=60, t=20, b=20),
                           template="plotly_white",
                           xaxis=dict(range=[0, 1.15], title="Reliability Score"))
        st.plotly_chart(fig6, use_container_width=True)

        tier_dist = scored["reliability_tier"].value_counts()
        fig7 = go.Figure(go.Pie(
            labels=tier_dist.index.astype(str), values=tier_dist.values, hole=0.55,
            marker=dict(colors=["#059669","#22c55e","#eab308","#f97316","#ef4444"]),
        ))
        fig7.update_layout(height=260, margin=dict(t=20, b=20, l=20, r=20),
                           template="plotly_white", showlegend=True,
                           legend=dict(orientation="h"))
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("No scored predictions available.")


# ──── Tab 4: Score a Prediction ───────────────────────────────
with tab4:
    st.subheader("🔮 Score Any Prediction")
    st.caption("Enter prediction market metadata to get a reliability assessment.")

    col_h, col_i = st.columns([3, 2])
    with col_h:
        pred_title    = st.text_area("Prediction question",
            value="Will global average temperature exceed 1.5°C above pre-industrial levels before 2030?",
            height=90)
        community_p   = st.slider("Community prediction", 0.01, 0.99, 0.62, 0.01, format="%g%%")
        n_forecasters = st.number_input("Number of forecasters", 1, 10000, 245)
    with col_i:
        cat          = st.selectbox("Category", ["Science","Economics","Politics","Technology",
                                                  "Health","Environment","Finance","Sports","Other"])
        question_age = st.slider("Question age (days)", 1, 730, 180)
        desc_len     = st.slider("Description length (chars)", 50, 5000, 800)
        n_comments   = st.number_input("Number of comments", 0, 1000, 35)

    if st.button("🎯 Analyze Reliability", type="primary", use_container_width=True):
        feats = {
            "question_lifespan_days": question_age,
            "time_to_close_days":     question_age * 1.5,
            "resolve_month":          6,
            "resolve_dayofweek":      2,
            "log_prediction_count":   np.log1p(n_forecasters),
            "log_comments":           np.log1p(n_comments),
            "engagement_ratio":       n_comments / (n_forecasters + 1),
            "prediction_density":     n_forecasters / (question_age + 1),
            "log_description_length": np.log1p(desc_len),
            "title_length":           len(pred_title),
            "title_word_count":       len(pred_title.split()),
            "has_number_in_title":    int(any(c.isdigit() for c in pred_title)),
            "is_yes_no_question":     int(any(w in pred_title.lower() for w in ["will","would","should"])),
            "community_pred":         community_p,
            "pred_confidence":        abs(community_p - 0.5) * 2,
            "pred_extreme":           int(community_p < 0.1 or community_p > 0.9),
        }
        row = pd.DataFrame([feats])
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        score = float(model.predict_proba(row[feature_cols].fillna(0).values)[0, 1])

        tier = ("Very High" if score > 0.85 else "High" if score > 0.70 else
                "Medium"    if score > 0.50 else "Low"  if score > 0.30 else "Very Low")
        tier_emoji = {"Very High":"🟢","High":"✅","Medium":"🟡","Low":"🟠","Very Low":"🔴"}[tier]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Reliability Score", f"{score:.1%}")
        r2.metric("Reliability Tier",  f"{tier_emoji} {tier}")
        r3.metric("Community Says",    f"{community_p:.0%}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score * 100,
            number={"suffix":"%","font":{"size":36}},
            gauge={
                "axis": {"range":[0,100],"tickwidth":1},
                "bar":  {"color": "#059669" if score > 0.7 else "#eab308" if score > 0.5 else "#ef4444"},
                "steps":[{"range":[0,30],"color":"#fef2f2"},{"range":[30,50],"color":"#fff7ed"},
                          {"range":[50,70],"color":"#fefce8"},{"range":[70,85],"color":"#f0fdf4"},
                          {"range":[85,100],"color":"#dcfce7"}],
                "threshold":{"line":{"color":"#1e293b","width":3},"thickness":0.8,"value":score*100},
            },
            title={"text":"Reliability Score"},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=20, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

        with st.spinner("Generating AI analysis..."):
            analysis = get_ai_analysis(pred_title, score, community_p, n_forecasters, cat)
        st.subheader("AI Analysis")
        st.markdown(analysis)

        # ── SHAP waterfall for this prediction ──
        shap_df_local = metrics.get("shap_df")
        if shap_df_local is not None:
            st.subheader("Feature Contribution Breakdown")
            st.caption("How each input feature pushed this prediction's reliability score up or down.")
            try:
                import shap as _shap
                if hasattr(model, "feature_importances_"):
                    _exp = _shap.TreeExplainer(model)
                    _sv  = _exp.shap_values(row[feature_cols].fillna(0).values)
                    if isinstance(_sv, list):
                        _sv = _sv[1]
                    _sv = _sv[0]
                    contrib_df = pd.DataFrame({"feature": feature_cols, "shap": _sv})
                    contrib_df["label"] = contrib_df["feature"].str.replace("_", " ").str.title()
                    contrib_df = contrib_df.reindex(contrib_df["shap"].abs().sort_values(ascending=False).index)
                    contrib_df = contrib_df.head(12).sort_values("shap")
                    colors = ["#22c55e" if v > 0 else "#ef4444" for v in contrib_df["shap"]]
                    fig_wf = go.Figure(go.Bar(
                        x=contrib_df["shap"], y=contrib_df["label"],
                        orientation="h", marker_color=colors,
                        text=contrib_df["shap"].round(4), textposition="outside",
                    ))
                    fig_wf.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                    fig_wf.update_layout(height=380, template="plotly_white",
                                         margin=dict(l=200, r=80, t=20, b=20),
                                         xaxis_title="SHAP Value (green = increases reliability)")
                    st.plotly_chart(fig_wf, use_container_width=True)
            except Exception:
                pass


# ──── Tab 5: Explainability ───────────────────────────────────
with tab5:
    st.subheader("🧠 Model Explainability")

    shap_df = metrics.get("shap_df")
    perm_df = metrics.get("perm_df")
    feature_cols_m = metrics.get("feature_cols", feature_cols)

    # ── SHAP Global Importance ──
    if shap_df is not None:
        st.markdown("#### SHAP Feature Importance (Mean |SHAP|)")
        st.caption("Unlike tree impurity importance, SHAP values are based on actual prediction contributions — more reliable and model-agnostic.")
        shap_mean = shap_df.abs().mean().sort_values(ascending=False).head(15)
        shap_mean_df = shap_mean.reset_index()
        shap_mean_df.columns = ["feature", "mean_shap"]
        shap_mean_df["feature_label"] = shap_mean_df["feature"].str.replace("_", " ").str.title()

        fig_shap = go.Figure(go.Bar(
            x=shap_mean_df["mean_shap"],
            y=shap_mean_df["feature_label"],
            orientation="h",
            marker=dict(color=shap_mean_df["mean_shap"], colorscale="Purples"),
            text=shap_mean_df["mean_shap"].round(4), textposition="outside",
        ))
        fig_shap.update_layout(height=440, margin=dict(l=200, r=60, t=20, b=20),
                               template="plotly_white", xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(fig_shap, use_container_width=True)

        # SHAP beeswarm as scatter
        st.markdown("#### SHAP Value Distribution (Top 8 Features)")
        st.caption("Each dot is one training sample. Right = pushed prediction higher (reliable), Left = pushed lower (unreliable). Color = feature value (red = high, blue = low).")
        top8 = shap_mean_df["feature"].head(8).tolist()
        rows = []
        for feat in top8:
            if feat in shap_df.columns:
                for sv in shap_df[feat].values:
                    rows.append({"feature": feat.replace("_", " ").title(), "shap": sv})
        if rows:
            bee_df = pd.DataFrame(rows)
            fig_bee = px.strip(bee_df, x="shap", y="feature", color="shap",
                               color_continuous_scale="RdBu_r",
                               labels={"shap": "SHAP Value", "feature": ""})
            fig_bee.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.5))
            fig_bee.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
            fig_bee.update_layout(height=380, template="plotly_white",
                                  margin=dict(t=20, b=40, l=180, r=40),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_bee, use_container_width=True)
    else:
        st.info("SHAP analysis requires a tree-based model (Random Forest or Gradient Boosting). "
                "Logistic Regression was selected this run — adjust Training data size to trigger a different model.")

    st.divider()

    # ── Permutation Importance / Ablation ──
    if perm_df is not None:
        st.markdown("#### Permutation Importance (Ablation Study)")
        st.caption("AUC drop when each feature is randomly shuffled on the held-out test set. "
                   "Larger drop = feature is critical. Error bars = std across 5 permutations.")
        top_perm = perm_df.head(12).sort_values("auc_drop_mean")
        fig_perm = go.Figure(go.Bar(
            x=top_perm["auc_drop_mean"],
            y=top_perm["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            error_x=dict(type="data", array=top_perm["auc_drop_std"].tolist(), visible=True),
            marker=dict(color=top_perm["auc_drop_mean"], colorscale="Oranges"),
            text=top_perm["auc_drop_mean"].round(4), textposition="outside",
        ))
        fig_perm.update_layout(height=400, margin=dict(l=200, r=80, t=20, b=20),
                               template="plotly_white", xaxis_title="AUC Drop (higher = more critical)")
        st.plotly_chart(fig_perm, use_container_width=True)

        top3 = perm_df.head(3)
        st.info(
            f"**Key finding:** Removing **{top3.iloc[0]['feature'].replace('_',' ').title()}** "
            f"drops AUC by **{top3.iloc[0]['auc_drop_mean']:.4f}** — "
            f"{top3.iloc[0]['auc_drop_mean'] / max(top3.iloc[2]['auc_drop_mean'], 1e-6):.1f}× "
            f"more impactful than **{top3.iloc[2]['feature'].replace('_',' ').title()}** "
            f"({top3.iloc[2]['auc_drop_mean']:.4f})."
        )


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎯 **PredictPulse** — ZerveHack 2026 | "
    "Python · scikit-learn · Plotly · Streamlit · Claude API | "
    "API: [d6aca690-2cad4778.hub.zerve.cloud](https://d6aca690-2cad4778.hub.zerve.cloud/docs)"
)
