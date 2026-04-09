"""
PredictPulse — Block 6: API Deployment
========================================
Deploys PredictPulse as a live API on Zerve for real-time
prediction market reliability scoring.

Deploy this block as a Zerve API endpoint.
"""

import json
import pandas as pd
import numpy as np


def predict_reliability(request_data):
    """
    API endpoint: Score a prediction market question's reliability.

    Input (JSON):
    {
        "title": "Will X happen by Y date?",
        "community_prediction": 0.75,
        "prediction_count": 150,
        "description_length": 500,
        "num_comments": 25,
        "question_age_days": 30,
        "category": "Science"
    }

    Output (JSON):
    {
        "reliability_score": 0.82,
        "reliability_tier": "High",
        "analysis": "...",
        "top_factors": [...],
        "metadata": {...}
    }
    """

    # Parse input
    title = request_data.get("title", "Unknown question")
    community_pred = float(request_data.get("community_prediction", 0.5))
    prediction_count = int(request_data.get("prediction_count", 0))
    description_length = int(request_data.get("description_length", 0))
    num_comments = int(request_data.get("num_comments", 0))
    question_age_days = float(request_data.get("question_age_days", 0))
    category = request_data.get("category", "Unknown")

    # Engineer features (same as training pipeline)
    features = {
        "question_lifespan_days": question_age_days,
        "time_to_close_days": question_age_days * 1.5,  # estimate
        "resolve_month": pd.Timestamp.now().month,
        "resolve_dayofweek": pd.Timestamp.now().dayofweek,
        "log_prediction_count": np.log1p(prediction_count),
        "log_comments": np.log1p(num_comments),
        "engagement_ratio": num_comments / (prediction_count + 1),
        "prediction_density": prediction_count / (question_age_days + 1),
        "log_description_length": np.log1p(description_length),
        "title_length": len(title),
        "title_word_count": len(title.split()),
        "has_number_in_title": int(any(c.isdigit() for c in title)),
        "is_yes_no_question": int(any(
            w in title.lower() for w in ["will", "would", "should"]
        )),
        "community_pred": community_pred,
        "pred_confidence": abs(community_pred - 0.5) * 2,
        "pred_extreme": int(community_pred < 0.1 or community_pred > 0.9),
    }

    # Add zero-valued category/FRED columns
    for col in feature_cols:
        if col not in features:
            features[col] = 0

    # Score with trained model
    X = pd.DataFrame([features])[feature_cols].fillna(0).values
    reliability_score = float(model.predict_proba(X)[0, 1])

    # Determine tier
    if reliability_score > 0.85:
        tier = "Very High"
    elif reliability_score > 0.7:
        tier = "High"
    elif reliability_score > 0.5:
        tier = "Medium"
    elif reliability_score > 0.3:
        tier = "Low"
    else:
        tier = "Very Low"

    # Get top contributing factors
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_cols))

    feature_values = pd.DataFrame([features])[feature_cols].fillna(0).iloc[0]
    factor_scores = importances * np.abs(feature_values.values)
    top_indices = np.argsort(factor_scores)[::-1][:5]

    top_factors = []
    for idx in top_indices:
        fname = feature_cols[idx]
        fval = float(feature_values.iloc[idx])
        fimp = float(importances[idx])
        top_factors.append({
            "feature": fname,
            "value": round(fval, 4),
            "importance": round(fimp, 4),
            "direction": "positive" if fval > 0 else "neutral",
        })

    # Generate AI analysis
    feature_insights = {f["feature"]: f["value"] for f in top_factors}
    analysis = analyze_prediction_with_claude(
        question_title=title,
        accuracy_score=reliability_score,
        community_prediction=community_pred,
        prediction_count=prediction_count,
        feature_insights=feature_insights,
        category=category,
    )

    return {
        "reliability_score": round(reliability_score, 4),
        "reliability_tier": tier,
        "community_prediction": community_pred,
        "analysis": analysis,
        "top_factors": top_factors,
        "metadata": {
            "model": "GradientBoosting",
            "training_samples": len(df_features),
            "features_used": len(feature_cols),
            "version": "1.0.0",
        },
    }


# ============================================================
# Test the API endpoint
# ============================================================

test_request = {
    "title": "Will global average temperature exceed 1.5°C above pre-industrial levels before 2030?",
    "community_prediction": 0.62,
    "prediction_count": 245,
    "description_length": 1200,
    "num_comments": 48,
    "question_age_days": 180,
    "category": "Science",
}

print("=" * 60)
print("API TEST — Prediction Reliability Scoring")
print("=" * 60)
print(f"\nInput: {test_request['title']}")
print(f"Community prediction: {test_request['community_prediction']:.0%}")

result = predict_reliability(test_request)

print(f"\n{'─' * 40}")
print(f"Reliability Score: {result['reliability_score']:.1%}")
print(f"Reliability Tier:  {result['reliability_tier']}")
print(f"\nAnalysis:")
print(result["analysis"])
print(f"\nTop Contributing Factors:")
for f in result["top_factors"]:
    print(f"  • {f['feature']}: {f['value']} (importance: {f['importance']})")

# Pretty print full response
print(f"\n{'─' * 40}")
print("Full API Response (JSON):")
print(json.dumps(result, indent=2, default=str))
