"""
PredictPulse — Block 2: Feature Engineering
=============================================
Transforms raw prediction market data into ML-ready features
that capture the signals behind forecast accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def engineer_prediction_features(resolved_df, fred_df=None):
    """
    Create features from resolved Metaculus questions that predict accuracy.

    Features capture:
    - Market dynamics (participation, timing)
    - Question characteristics (complexity, domain)
    - Economic context (macro environment at question time)
    """

    df = resolved_df.copy()

    # ========================================
    # Time-based features
    # ========================================
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    df["resolve_time"] = pd.to_datetime(df["resolve_time"], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")

    df["question_lifespan_days"] = (
        (df["resolve_time"] - df["created_time"]).dt.total_seconds() / 86400
    )
    df["time_to_close_days"] = (
        (df["close_time"] - df["created_time"]).dt.total_seconds() / 86400
    )
    df["resolve_month"] = df["resolve_time"].dt.month
    df["resolve_year"] = df["resolve_time"].dt.year
    df["resolve_dayofweek"] = df["resolve_time"].dt.dayofweek

    # ========================================
    # Participation features
    # ========================================
    df["log_prediction_count"] = np.log1p(df["prediction_count"])
    df["log_comments"] = np.log1p(df["num_comments"])
    df["engagement_ratio"] = df["num_comments"] / (df["prediction_count"] + 1)

    # Prediction density (predictions per day of question life)
    df["prediction_density"] = df["prediction_count"] / (df["question_lifespan_days"] + 1)

    # ========================================
    # Question complexity features
    # ========================================
    df["log_description_length"] = np.log1p(df["description_length"])

    # Title-based features
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["has_number_in_title"] = df["title"].str.contains(r"\d", regex=True).astype(int)
    df["is_yes_no_question"] = df["title"].str.lower().str.contains(
        r"\bwill\b|\bwould\b|\bshould\b|\bis\b.*\?", regex=True
    ).astype(int)

    # ========================================
    # Community prediction features
    # ========================================
    df["community_pred"] = df["community_prediction"].astype(float)
    df["pred_confidence"] = abs(df["community_pred"] - 0.5) * 2  # 0=uncertain, 1=confident
    df["pred_extreme"] = ((df["community_pred"] < 0.1) | (df["community_pred"] > 0.9)).astype(int)

    # ========================================
    # Category encoding
    # ========================================
    category_counts = df["category"].value_counts()
    top_categories = category_counts.head(10).index.tolist()
    df["category_clean"] = df["category"].apply(
        lambda x: x if x in top_categories else "Other"
    )
    category_dummies = pd.get_dummies(df["category_clean"], prefix="cat")
    df = pd.concat([df, category_dummies], axis=1)

    # ========================================
    # Target variable: prediction accuracy
    # ========================================
    # For binary questions (resolution is 0 or 1):
    # accuracy = 1 - |community_prediction - resolution|
    df["resolution_float"] = pd.to_numeric(df["resolution"], errors="coerce")

    # Filter to binary resolutions (0 or 1)
    binary_mask = df["resolution_float"].isin([0.0, 1.0])
    df_binary = df[binary_mask].copy()

    df_binary["prediction_error"] = abs(
        df_binary["community_pred"] - df_binary["resolution_float"]
    )
    df_binary["prediction_accurate"] = (df_binary["prediction_error"] < 0.3).astype(int)
    df_binary["accuracy_score"] = 1 - df_binary["prediction_error"]

    # ========================================
    # Economic context features (if FRED data available)
    # ========================================
    if fred_df is not None and not fred_df.empty:
        df_binary["resolve_month_start"] = df_binary["resolve_time"].dt.to_period("M").dt.to_timestamp()

        fred_monthly = fred_df.copy()
        fred_monthly.index = fred_monthly.index.to_period("M").to_timestamp()

        df_binary = df_binary.merge(
            fred_monthly, left_on="resolve_month_start", right_index=True, how="left"
        )

        # Economic volatility features
        for col in fred_monthly.columns:
            if col in df_binary.columns:
                df_binary[f"{col}_zscore"] = (
                    (df_binary[col] - df_binary[col].mean()) / (df_binary[col].std() + 1e-8)
                )

    # ========================================
    # Clean up
    # ========================================
    # Define feature columns
    feature_cols = [
        "question_lifespan_days", "time_to_close_days",
        "resolve_month", "resolve_dayofweek",
        "log_prediction_count", "log_comments",
        "engagement_ratio", "prediction_density",
        "log_description_length", "title_length",
        "title_word_count", "has_number_in_title",
        "is_yes_no_question",
        "community_pred", "pred_confidence", "pred_extreme",
    ]

    # Add category dummies
    feature_cols += [c for c in df_binary.columns if c.startswith("cat_")]

    # Add FRED features if available
    if fred_df is not None:
        feature_cols += [c for c in df_binary.columns if c.endswith("_zscore")]

    # Remove rows with NaN in features
    df_clean = df_binary.dropna(subset=feature_cols + ["prediction_accurate"])

    print(f"\nFeature Engineering Complete:")
    print(f"  Total resolved questions: {len(resolved_df)}")
    print(f"  Binary questions: {len(df_binary)}")
    print(f"  Clean samples: {len(df_clean)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Accuracy rate: {df_clean['prediction_accurate'].mean():.1%}")
    print(f"  Mean prediction error: {df_clean['prediction_error'].mean():.3f}")

    return df_clean, feature_cols


def prepare_open_questions(open_df, feature_cols):
    """Prepare open/active questions for scoring (same features, no target)."""
    df = open_df.copy()

    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce").dt.tz_localize(None)
    now = pd.Timestamp.now()
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce").dt.tz_localize(None)

    df["question_lifespan_days"] = (now - df["created_time"]).dt.total_seconds() / 86400
    df["time_to_close_days"] = (
        (df["close_time"] - df["created_time"]).dt.total_seconds() / 86400
    )
    df["resolve_month"] = datetime.now().month
    df["resolve_dayofweek"] = datetime.now().weekday()
    df["log_prediction_count"] = np.log1p(df["prediction_count"])
    df["log_comments"] = np.log1p(df["num_comments"])
    df["engagement_ratio"] = df["num_comments"] / (df["prediction_count"] + 1)
    df["prediction_density"] = df["prediction_count"] / (df["question_lifespan_days"] + 1)
    df["log_description_length"] = np.log1p(df["description_length"])
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["has_number_in_title"] = df["title"].str.contains(r"\d", regex=True).astype(int)
    df["is_yes_no_question"] = df["title"].str.lower().str.contains(
        r"\bwill\b|\bwould\b|\bshould\b|\bis\b.*\?", regex=True
    ).astype(int)
    df["community_pred"] = df["community_prediction"].astype(float)
    df["pred_confidence"] = abs(df["community_pred"] - 0.5) * 2
    df["pred_extreme"] = ((df["community_pred"] < 0.1) | (df["community_pred"] > 0.9)).astype(int)

    # Add missing category/FRED columns as zeros
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df


# Run feature engineering
df_features, feature_cols = engineer_prediction_features(resolved_df, fred_df)

# Preview
print("\n--- Feature Distributions ---")
print(df_features[feature_cols].describe().round(2))
