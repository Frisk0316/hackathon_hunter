"""
PredictPulse — Block 4: Visualization & Storytelling
=====================================================
Creates publication-quality visualizations that tell the story
of prediction market accuracy patterns.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Contained Bootstrap — makes this block runnable without prior blocks
# ═══════════════════════════════════════════════════════════════════════════════
def _run_bootstrap_pipeline():
    """Full pipeline from synthetic data. Returns (r_df,o_df,fred,df_features,
    feature_cols,model,evaluation,importance_df,open_prepared,scored_questions)."""
    import numpy as _np, pandas as _pd
    from datetime import datetime as _dt, timedelta as _td
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.metrics import roc_auc_score as _auc, brier_score_loss as _brier
    from sklearn.calibration import calibration_curve as _cal_curve
    from sklearn.model_selection import cross_val_predict as _cvp, StratifiedKFold as _SKF
    import warnings as _w; _w.filterwarnings("ignore")

    _np.random.seed(42); _rng = _np.random.default_rng(42)
    _CATS = ["Science","Economics","Politics","Technology","Health",
             "Environment","Sports","Finance","Social","Other"]
    _CW   = [0.20,0.18,0.16,0.14,0.10,0.08,0.06,0.04,0.02,0.02]
    _BASE = _dt(2021,1,1)

    def _mk(n, is_resolved):
        rows = []
        for i in range(n):
            ls  = int(_rng.integers(30,730))
            cre = _BASE + _td(days=int(_rng.integers(0,1000).item()))
            clo = cre + _td(days=ls)
            res = clo + _td(days=int(_rng.integers(0,30).item()))
            np_ = int(_np.clip(_rng.lognormal(3.5,1.2),3,2000))
            nc  = int(_np.clip(_rng.lognormal(2.0,1.0),0,300))
            cp  = float(_np.clip(_rng.beta(2,2),0.02,0.98))
            cat = str(_rng.choice(_CATS,p=_CW))
            dl  = int(_np.clip(_rng.lognormal(5.5,0.8),100,5000))
            if is_resolved:
                conf= abs(cp-0.5)*2
                acc = _np.clip(0.55+0.15*_np.log1p(np_/(ls+1))/5-0.05*conf,0.3,0.85)
                rv  = str(1.0 if (cp>0.5)==(_rng.random()<acc) else 0.0)
            else:
                rv = None
            rows.append({"id":10000+i if is_resolved else 20000+i,
                         "title":f"Will {cat.lower()} event #{i} occur by {res.year}?",
                         "created_time":cre.isoformat(),"resolve_time":res.isoformat(),
                         "close_time":clo.isoformat(),"prediction_count":np_,
                         "community_prediction":cp,"resolution":rv,"category":cat,
                         "description_length":dl,"num_comments":nc,
                         "url":f"https://www.metaculus.com/questions/{i}/"})
        return _pd.DataFrame(rows)

    r_df, o_df = _mk(500,True), _mk(80,False)

    def _eng(df, for_open=False):
        d = df.copy()
        d["created_time"] = _pd.to_datetime(d["created_time"],errors="coerce")
        if for_open:
            d["created_time"] = d["created_time"].dt.tz_localize(None)
            d["close_time"]   = _pd.to_datetime(d["close_time"],errors="coerce").dt.tz_localize(None)
            now = _pd.Timestamp.now()
            d["question_lifespan_days"] = (now-d["created_time"]).dt.total_seconds()/86400
            d["time_to_close_days"]     = (d["close_time"]-d["created_time"]).dt.total_seconds()/86400
            d["resolve_month"]    = now.month; d["resolve_dayofweek"]= now.dayofweek
        else:
            d["resolve_time"] = _pd.to_datetime(d["resolve_time"],errors="coerce")
            d["close_time"]   = _pd.to_datetime(d["close_time"],  errors="coerce")
            d["question_lifespan_days"] = (d["resolve_time"]-d["created_time"]).dt.total_seconds()/86400
            d["time_to_close_days"]     = (d["close_time"]-d["created_time"]).dt.total_seconds()/86400
            d["resolve_month"]    = d["resolve_time"].dt.month
            d["resolve_dayofweek"]= d["resolve_time"].dt.dayofweek
        d["log_prediction_count"]   = _np.log1p(d["prediction_count"])
        d["log_comments"]           = _np.log1p(d["num_comments"])
        d["engagement_ratio"]       = d["num_comments"]/(d["prediction_count"]+1)
        d["prediction_density"]     = d["prediction_count"]/(d["question_lifespan_days"]+1)
        d["log_description_length"] = _np.log1p(d["description_length"])
        d["title_length"]           = d["title"].str.len()
        d["title_word_count"]       = d["title"].str.split().str.len()
        d["has_number_in_title"]    = d["title"].str.contains(r"\d",regex=True).astype(int)
        d["is_yes_no_question"]     = d["title"].str.lower().str.contains(r"\bwill\b|\bwould\b",regex=True).astype(int)
        d["community_pred"]         = d["community_prediction"].astype(float)
        d["pred_confidence"]        = abs(d["community_pred"]-0.5)*2
        d["pred_extreme"]           = ((d["community_pred"]<0.1)|(d["community_pred"]>0.9)).astype(int)
        dums = _pd.get_dummies(d["category"],prefix="cat")
        return _pd.concat([d,dums],axis=1)

    d_r = _eng(r_df, False)
    d_r["resolution_float"]    = _pd.to_numeric(d_r["resolution"],errors="coerce")
    d_r = d_r[d_r["resolution_float"].isin([0.0,1.0])].copy()
    d_r["prediction_error"]    = abs(d_r["community_pred"]-d_r["resolution_float"])
    d_r["prediction_accurate"] = (d_r["prediction_error"]<0.3).astype(int)
    d_r["accuracy_score"]      = 1-d_r["prediction_error"]

    fcols = ["question_lifespan_days","time_to_close_days","resolve_month","resolve_dayofweek",
             "log_prediction_count","log_comments","engagement_ratio","prediction_density",
             "log_description_length","title_length","title_word_count","has_number_in_title",
             "is_yes_no_question","community_pred","pred_confidence","pred_extreme"]
    fcols += [c for c in d_r.columns if c.startswith("cat_")]
    d_r = d_r.dropna(subset=fcols+["prediction_accurate"])

    X = d_r[fcols].fillna(0).values; y = d_r["prediction_accurate"].values
    mdl = _RFC(n_estimators=200,max_depth=8,min_samples_leaf=5,random_state=42)
    mdl.fit(X,y); y_p = mdl.predict_proba(X)[:,1]
    imp_df = _pd.DataFrame({"feature":fcols,"importance":mdl.feature_importances_}).sort_values("importance",ascending=False)

    cv  = _SKF(n_splits=5,shuffle=True,random_state=42)
    cvp = _cvp(mdl,X,y,cv=cv,method="predict_proba")[:,1]
    fp,mp  = _cal_curve(y,cvp,n_bins=8,strategy="quantile")
    bs     = float(_brier(y,y_p)); bss = 1-(bs/(y.mean()*(1-y.mean())))
    mce    = float(abs(fp-mp).mean())
    cal_df = _pd.DataFrame({"mean_predicted_prob":mp,"fraction_of_positives":fp,"calibration_gap":fp-mp})
    evl = {"best_model":"Random Forest","auc_roc":float(_auc(y,y_p)),
           "brier_score":bs,"brier_skill_score":float(bss),"mean_calibration_error":mce,
           "calibration_data":cal_df,"category_stats":{},"cv_results":{},"feature_importances":imp_df}

    d_o = _eng(o_df, True)
    for c in fcols:
        if c not in d_o.columns: d_o[c] = 0
    o_p = mdl.predict_proba(d_o[fcols].fillna(0).values)[:,1]
    d_o["accuracy_score"]   = o_p
    d_o["community_pred"]   = d_o["community_prediction"].astype(float)
    d_o["reliability_tier"] = pd.cut(o_p,[0,.3,.5,.7,.85,1.],
                                     labels=["Very Low","Low","Medium","High","Very High"])
    scored = d_o.sort_values("accuracy_score",ascending=False)

    print(f"[Bootstrap] ✓ {len(d_r)} samples | AUC: {float(_auc(y,y_p)):.3f} | "
          f"Brier Skill: {float(bss):.3f} | Cal Error: {mce:.3f}")
    return r_df, o_df, None, d_r, fcols, mdl, evl, imp_df, d_o, scored
# ═══════════════════════════════════════════════════════════════════════════════


def create_accuracy_overview(df_features):
    """Main dashboard: prediction accuracy landscape."""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Prediction Accuracy Distribution",
            "Accuracy by Community Confidence",
            "Accuracy by Participation Level",
            "Accuracy Over Time"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Accuracy Distribution
    fig.add_trace(
        go.Histogram(
            x=df_features["prediction_error"],
            nbinsx=30,
            marker_color="#6366f1",
            opacity=0.8,
            name="Prediction Error",
        ),
        row=1, col=1
    )

    # 2. Accuracy vs Confidence
    fig.add_trace(
        go.Scatter(
            x=df_features["pred_confidence"],
            y=df_features["accuracy_score"],
            mode="markers",
            marker=dict(
                size=6,
                color=df_features["log_prediction_count"],
                colorscale="Viridis",
                opacity=0.6,
                colorbar=dict(title="Log Predictions", x=0.48),
            ),
            name="Questions",
        ),
        row=1, col=2
    )

    # 3. Accuracy by Participation (binned)
    df_features["participation_bin"] = pd.qcut(
        df_features["prediction_count"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    accuracy_by_participation = df_features.groupby("participation_bin", observed=True).agg(
        mean_accuracy=("accuracy_score", "mean"),
        count=("accuracy_score", "count"),
    ).reset_index()

    fig.add_trace(
        go.Bar(
            x=accuracy_by_participation["participation_bin"].astype(str),
            y=accuracy_by_participation["mean_accuracy"],
            marker_color=["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4"],
            text=accuracy_by_participation["mean_accuracy"].round(2),
            textposition="auto",
            name="Avg Accuracy",
        ),
        row=2, col=1
    )

    # 4. Accuracy over time
    df_features["resolve_quarter"] = df_features["resolve_time"].dt.to_period("Q").astype(str)
    accuracy_over_time = df_features.groupby("resolve_quarter").agg(
        mean_accuracy=("accuracy_score", "mean"),
        count=("accuracy_score", "count"),
    ).reset_index()

    fig.add_trace(
        go.Scatter(
            x=accuracy_over_time["resolve_quarter"],
            y=accuracy_over_time["mean_accuracy"],
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=3),
            marker=dict(size=8),
            name="Quarterly Accuracy",
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text="PredictPulse: Prediction Market Accuracy Landscape",
            font=dict(size=20),
        ),
        height=700,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
    )

    fig.show()
    return fig


def create_feature_importance_chart(importance_df):
    """Horizontal bar chart of feature importances."""

    top_n = importance_df.head(12).sort_values("importance")

    fig = go.Figure(go.Bar(
        x=top_n["importance"],
        y=top_n["feature"].str.replace("_", " ").str.title(),
        orientation="h",
        marker=dict(
            color=top_n["importance"],
            colorscale="Purples",
        ),
        text=top_n["importance"].round(4),
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text="What Drives Prediction Accuracy?",
            subtitle=dict(text="Top 12 features by model importance"),
            font=dict(size=18),
        ),
        xaxis_title="Feature Importance",
        height=500,
        template="plotly_white",
        margin=dict(l=200),
        font=dict(family="Inter, sans-serif"),
    )

    fig.show()
    return fig


def create_reliability_scorecard(scored_questions):
    """Visual scorecard of open prediction reliability."""

    top = scored_questions.head(20).copy()
    top["title_short"] = top["title"].str[:50] + "..."
    top["color"] = top["accuracy_score"].apply(
        lambda x: "#22c55e" if x > 0.7 else "#eab308" if x > 0.5 else "#ef4444"
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top["title_short"].iloc[::-1],
        x=top["accuracy_score"].iloc[::-1],
        orientation="h",
        marker=dict(color=top["color"].iloc[::-1]),
        text=top["accuracy_score"].iloc[::-1].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text="Current Prediction Reliability Scores",
            subtitle=dict(text="Green = High confidence | Yellow = Moderate | Red = Low"),
            font=dict(size=18),
        ),
        xaxis=dict(title="Reliability Score", range=[0, 1.1]),
        height=600,
        template="plotly_white",
        margin=dict(l=350),
        font=dict(family="Inter, sans-serif"),
    )

    fig.show()
    return fig


def create_category_analysis(df_features):
    """Accuracy patterns by prediction category."""

    cat_stats = df_features.groupby("category_clean").agg(
        mean_accuracy=("accuracy_score", "mean"),
        median_error=("prediction_error", "median"),
        count=("accuracy_score", "count"),
        avg_predictions=("prediction_count", "mean"),
    ).reset_index()

    cat_stats = cat_stats[cat_stats["count"] >= 5].sort_values("mean_accuracy", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cat_stats["avg_predictions"],
        y=cat_stats["mean_accuracy"],
        mode="markers+text",
        marker=dict(
            size=cat_stats["count"] / cat_stats["count"].max() * 40 + 10,
            color=cat_stats["mean_accuracy"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Accuracy"),
        ),
        text=cat_stats["category_clean"],
        textposition="top center",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title=dict(
            text="Prediction Accuracy by Category",
            subtitle=dict(text="Bubble size = number of questions | Color = accuracy"),
            font=dict(size=18),
        ),
        xaxis_title="Average Predictions per Question",
        yaxis_title="Mean Accuracy Score",
        height=500,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
    )

    fig.show()
    return fig


# ─── Bootstrap Check ─────────────────────────────────────────────────────────
if 'df_features' not in globals() or 'scored_questions' not in globals():
    print("[Block 4 Bootstrap] Prior blocks not detected — running full pipeline...")
    (_bs_r, _bs_o, _bs_f, df_features, feature_cols, model, evaluation,
     importance_df, open_prepared, scored_questions) = _run_bootstrap_pipeline()
    resolved_df, open_df, fred_df = _bs_r, _bs_o, _bs_f
# ─────────────────────────────────────────────────────────────────────────────

# Generate all visualizations
print("Generating visualizations...")
fig1 = create_accuracy_overview(df_features)
fig2 = create_feature_importance_chart(importance_df)
fig3 = create_reliability_scorecard(scored_questions)
fig4 = create_category_analysis(df_features)
print("All visualizations complete!")
