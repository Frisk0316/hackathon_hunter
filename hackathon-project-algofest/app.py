"""
PredictPulse — Streamlit Web App
=================================
Interactive dashboard for prediction market reliability analysis.
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, "src")
from pipeline import load_data, engineer_features, train_model, score_questions, get_ai_analysis


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PredictPulse",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 16px 20px; margin: 6px 0;
    }
    .tier-very-high { color: #059669; font-weight: 700; }
    .tier-high      { color: #16a34a; font-weight: 700; }
    .tier-medium    { color: #d97706; font-weight: 700; }
    .tier-low       { color: #dc2626; font-weight: 600; }
    .tier-very-low  { color: #7c3aed; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("public/logo.png") if os.path.exists("public/logo.png") else st.title("🎯 PredictPulse")
    st.caption("AI-Powered Prediction Market Intelligence")
    st.divider()

    st.subheader("⚙️ Settings")
    n_samples = st.slider("Training data size", 200, 800, 500, 100)
    n_bins = st.slider("Calibration bins", 5, 12, 8)

    st.divider()
    st.subheader("🔑 API Keys (optional)")
    metaculus_token = st.text_input("Metaculus Token", type="password",
                                    help="Get free token at metaculus.com/aib")
    anthropic_key   = st.text_input("Anthropic API Key", type="password",
                                    help="For AI-powered analysis")

    if metaculus_token:
        os.environ["METACULUS_TOKEN"]  = metaculus_token
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    st.divider()
    st.caption("Built for **AlgoFest Hackathon 2026**")
    st.caption("Stack: Python · scikit-learn · Plotly · Claude API")


# ─────────────────────────────────────────────
# Load & train (cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline(n_samples, _token, _key):
    resolved, open_df, source = load_data()
    if len(resolved) > n_samples:
        resolved = resolved.sample(n_samples, random_state=42)
    resolved_feat, open_feat, feature_cols = engineer_features(resolved, open_df)
    model, metrics, importance_df = train_model(resolved_feat, feature_cols)
    scored = score_questions(model, open_feat, feature_cols) if open_feat is not None else None
    return model, metrics, importance_df, scored, open_feat, feature_cols, source


with st.spinner("Running PredictPulse pipeline..."):
    model, metrics, importance_df, scored, open_feat, feature_cols, data_source = run_pipeline(
        n_samples, metaculus_token, anthropic_key
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("🎯 PredictPulse")
st.markdown("**AI-powered prediction market intelligence** — *Can we predict which forecasts to trust?*")

if data_source == "synthetic":
    st.info("📊 Using synthetic data (calibrated to Metaculus statistics). "
            "Add your **Metaculus Token** in the sidebar for real predictions.", icon="ℹ️")
else:
    st.success(f"✅ Live Metaculus data loaded", icon="✅")

st.divider()


# ─────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Training Samples",   f"{metrics['n_train']:,}")
k2.metric("Features",           f"{metrics['n_features']}")
k3.metric("AUC-ROC",            f"{metrics['auc_roc']:.3f}", delta="vs 0.500 baseline")
k4.metric("Brier Skill Score",  f"{metrics['brier_skill']:.3f}", delta="↑ higher = better")
k5.metric("Best Model",         metrics["best_model"].split()[0])

st.divider()


# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔬 Model Analysis",
    "🏆 Reliability Scores",
    "🔮 Score a Prediction",
])


# ──── Tab 1: Dashboard ────────────────────────

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
            text=top["importance"].round(3),
            textposition="outside",
        ))
        fig.update_layout(height=420, margin=dict(l=180, r=40, t=20, b=20),
                          template="plotly_white", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Model Comparison (5-fold CV AUC-ROC)")
        cv_df = pd.DataFrame([
            {"model": k.split()[0] + " " + k.split()[-1] if len(k.split()) > 1 else k, "auc": v}
            for k, v in metrics["cv_auc"].items()
        ]).sort_values("auc")
        fig2 = go.Figure(go.Bar(
            x=cv_df["auc"],
            y=cv_df["model"],
            orientation="h",
            marker=dict(color=["#6366f1" if v == max(metrics["cv_auc"].values()) else "#e2e8f0"
                               for v in cv_df["auc"]]),
            text=cv_df["auc"].round(3),
            textposition="outside",
        ))
        fig2.update_layout(height=200, margin=dict(l=160, r=60, t=20, b=20),
                           template="plotly_white", xaxis=dict(range=[0.5, 1.0]),
                           xaxis_title="AUC-ROC")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Accuracy by Category")
        if metrics["category_stats"]:
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


# ──── Tab 2: Model Analysis ───────────────────

with tab2:
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Calibration Curve")
        st.caption("Does a 70% prediction really mean 70%? Perfect calibration = diagonal line.")
        cal = metrics["calibration_df"]
        fig4 = go.Figure()
        # Perfect calibration reference
        fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="#94a3b8", width=1),
                                  name="Perfect calibration"))
        # Model calibration
        fig4.add_trace(go.Scatter(
            x=cal["mean_pred"], y=cal["frac_pos"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=10, color="#6366f1"),
            name="PredictPulse",
            hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
        ))
        # Shade the gap
        fig4.add_trace(go.Scatter(
            x=list(cal["mean_pred"]) + list(cal["mean_pred"])[::-1],
            y=list(cal["frac_pos"]) + list(cal["mean_pred"])[::-1],
            fill="toself", fillcolor="rgba(99,102,241,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig4.update_layout(
            height=380, template="plotly_white",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            margin=dict(t=20, b=40),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )
        st.plotly_chart(fig4, use_container_width=True)

        brier_col, skill_col = st.columns(2)
        brier_col.metric("Brier Score",       f"{metrics['brier_score']:.4f}",
                          help="Lower = better. Random = 0.25, Perfect = 0.00")
        skill_col.metric("Brier Skill Score", f"{metrics['brier_skill']:.3f}",
                          help="Higher = better. 0 = no skill, 1 = perfect")

    with col_d:
        st.subheader("Calibration Gap by Bin")
        st.caption("Positive gap = model underestimates true probability (conservative). "
                   "Negative gap = overestimates (overconfident).")
        cal_gap = cal.copy()
        cal_gap["gap"] = cal_gap["frac_pos"] - cal_gap["mean_pred"]
        cal_gap["color"] = cal_gap["gap"].apply(
            lambda g: "#22c55e" if abs(g) < 0.05 else ("#f59e0b" if abs(g) < 0.10 else "#ef4444"))
        fig5 = go.Figure(go.Bar(
            x=cal_gap["mean_pred"].round(2).astype(str),
            y=cal_gap["gap"],
            marker_color=cal_gap["color"],
            text=cal_gap["gap"].round(3),
            textposition="outside",
        ))
        fig5.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig5.update_layout(
            height=300, template="plotly_white",
            xaxis_title="Predicted Probability Bin",
            yaxis_title="Calibration Gap (Actual − Predicted)",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Key Findings")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Best CV AUC-ROC | **{max(metrics['cv_auc'].values()):.3f}** |
| Brier Score | **{metrics['brier_score']:.4f}** |
| Brier Skill | **{metrics['brier_skill']:.3f}** |
| Avg Precision | **{metrics['avg_precision']:.3f}** |
| Training Samples | **{metrics['n_train']:,}** |
| Features Used | **{metrics['n_features']}** |
""")
        st.info("💡 **Top insight:** Community confidence (`pred_confidence`) is the strongest "
                "predictor of accuracy — predictions near 50% are harder to score reliably "
                "than those with strong consensus.")


# ──── Tab 3: Reliability Scores ───────────────

with tab3:
    st.subheader("Live Prediction Reliability Scores")
    if scored is not None and len(scored) > 0:
        tier_colors = {
            "Very High": "🟢", "High": "🟩", "Medium": "🟡", "Low": "🟠", "Very Low": "🔴"
        }

        col_f, col_g = st.columns([2, 1])
        with col_g:
            tier_filter = st.multiselect(
                "Filter by tier",
                ["Very High", "High", "Medium", "Low", "Very Low"],
                default=["Very High", "High", "Medium"],
            )
        filtered = scored[scored["reliability_tier"].isin(tier_filter)] if tier_filter else scored

        with col_f:
            st.caption(f"Showing {len(filtered)} predictions")

        # Reliability scorecard chart
        top20 = filtered.head(20)
        fig6 = go.Figure(go.Bar(
            y=top20["title"].str[:50] + "...",
            x=top20["reliability_score"],
            orientation="h",
            marker=dict(
                color=top20["reliability_score"],
                colorscale=[[0, "#ef4444"], [0.4, "#f97316"], [0.6, "#eab308"],
                            [0.75, "#22c55e"], [1.0, "#059669"]],
                cmin=0, cmax=1,
            ),
            text=top20["reliability_score"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
        ))
        fig6.update_layout(
            height=max(350, len(top20) * 28),
            margin=dict(l=340, r=60, t=20, b=20),
            template="plotly_white",
            xaxis=dict(range=[0, 1.15], title="Reliability Score"),
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Distribution donut
        tier_dist = scored["reliability_tier"].value_counts()
        fig7 = go.Figure(go.Pie(
            labels=tier_dist.index.astype(str),
            values=tier_dist.values,
            hole=0.55,
            marker=dict(colors=["#059669", "#22c55e", "#eab308", "#f97316", "#ef4444"]),
        ))
        fig7.update_layout(height=260, margin=dict(t=20, b=20, l=20, r=20),
                           template="plotly_white", showlegend=True,
                           legend=dict(orientation="h"))
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("No scored predictions available.")


# ──── Tab 4: Score a Prediction ───────────────

with tab4:
    st.subheader("🔮 Score Any Prediction")
    st.caption("Enter prediction market metadata to get an AI-powered reliability assessment.")

    col_h, col_i = st.columns([3, 2])
    with col_h:
        pred_title   = st.text_area("Prediction question",
            value="Will global average temperature exceed 1.5°C above pre-industrial levels before 2030?",
            height=90)
        community_p  = st.slider("Community prediction", 0.01, 0.99, 0.62, 0.01,
                                  format="%g%%", help="Current market probability")
        n_forecasters = st.number_input("Number of forecasters", 1, 10000, 245)

    with col_i:
        cat          = st.selectbox("Category", ["Science", "Economics", "Politics",
                                                  "Technology", "Health", "Environment",
                                                  "Finance", "Sports", "Other"])
        question_age = st.slider("Question age (days)", 1, 730, 180)
        desc_len     = st.slider("Description length (chars)", 50, 5000, 800)
        n_comments   = st.number_input("Number of comments", 0, 1000, 35)

    if st.button("🎯 Analyze Reliability", type="primary", use_container_width=True):
        # Build feature vector
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
        X_pred = row[feature_cols].fillna(0).values
        score = float(model.predict_proba(X_pred)[0, 1])

        tier = ("Very High" if score > 0.85 else "High" if score > 0.70 else
                "Medium" if score > 0.50 else "Low" if score > 0.30 else "Very Low")
        tier_emoji = {"Very High": "🟢", "High": "✅", "Medium": "🟡",
                      "Low": "🟠", "Very Low": "🔴"}[tier]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Reliability Score", f"{score:.1%}")
        r2.metric("Reliability Tier", f"{tier_emoji} {tier}")
        r3.metric("Community Says", f"{community_p:.0%}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": ("#059669" if score > 0.7 else "#eab308" if score > 0.5 else "#ef4444")},
                "steps": [
                    {"range": [0, 30],  "color": "#fef2f2"},
                    {"range": [30, 50], "color": "#fff7ed"},
                    {"range": [50, 70], "color": "#fefce8"},
                    {"range": [70, 85], "color": "#f0fdf4"},
                    {"range": [85, 100],"color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#1e293b", "width": 3},
                              "thickness": 0.8, "value": score * 100},
            },
            title={"text": "Reliability Score"},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=20, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

        with st.spinner("Generating AI analysis..."):
            analysis = get_ai_analysis(pred_title, score, community_p, n_forecasters, cat)

        st.subheader("AI Analysis")
        st.markdown(analysis)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()
st.caption("🎯 **PredictPulse** — Built for AlgoFest Hackathon 2026 | "
           "Python · scikit-learn · Plotly · Streamlit · Claude API")
