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


# Generate all visualizations
print("Generating visualizations...")
fig1 = create_accuracy_overview(df_features)
fig2 = create_feature_importance_chart(importance_df)
fig3 = create_reliability_scorecard(scored_questions)
fig4 = create_category_analysis(df_features)
print("All visualizations complete!")
