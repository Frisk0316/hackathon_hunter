"""
PredictPulse — Block 3: Model Training & Evaluation
=====================================================
Trains an ensemble model to predict which prediction market
forecasts are likely to be accurate, with full evaluation.

Metrics reported:
  - AUC-ROC, F1 (standard classification)
  - Brier Score (probabilistic accuracy — lower is better)
  - Calibration Curve (reliability diagram — does 70% really mean 70%?)
  - Cross-category accuracy breakdown
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def train_accuracy_model(df, feature_cols):
    """
    Train an ensemble to predict prediction market accuracy.
    Returns trained model, evaluation metrics, and feature importances.
    """

    X = df[feature_cols].fillna(0).values
    y = df["prediction_accurate"].values

    print("=" * 60)
    print("MODEL TRAINING — Prediction Accuracy Classifier")
    print("=" * 60)
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Positive rate: {y.mean():.1%}")

    # ========================================
    # Model comparison with cross-validation
    # ========================================
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n--- Cross-Validation Results (5-fold) ---")
    print(f"{'Model':<25} {'AUC-ROC':>10} {'Accuracy':>10} {'F1':>10}")
    print("-" * 55)

    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        results[name] = {
            "auc": auc_scores.mean(),
            "accuracy": acc_scores.mean(),
            "f1": f1_scores.mean(),
        }
        print(f"{name:<25} {auc_scores.mean():>10.3f} {acc_scores.mean():>10.3f} {f1_scores.mean():>10.3f}")

    # ========================================
    # Select best model and train on full data
    # ========================================
    best_model_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nBest model: {best_model_name} (AUC={results[best_model_name]['auc']:.3f})")

    best_model = models[best_model_name]
    best_model.fit(X, y)

    # ========================================
    # Feature importance analysis
    # ========================================
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        clf = best_model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            importances = np.zeros(len(feature_cols))
    else:
        importances = np.zeros(len(feature_cols))

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n--- Top 10 Most Important Features ---")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

    # ========================================
    # Detailed evaluation on full dataset
    # ========================================
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y, y_pred, target_names=["Inaccurate", "Accurate"]))

    cm = confusion_matrix(y, y_pred)
    print("--- Confusion Matrix ---")
    print(f"  True Negatives:  {cm[0, 0]:>5}  |  False Positives: {cm[0, 1]:>5}")
    print(f"  False Negatives: {cm[1, 0]:>5}  |  True Positives:  {cm[1, 1]:>5}")

    # ========================================
    # Brier Score — probabilistic calibration quality
    # ========================================
    # Brier Score = mean((predicted_prob - actual_outcome)^2)
    # Perfect model = 0.0 | No-skill baseline = class_imbalance * (1 - class_imbalance)
    brier = brier_score_loss(y, y_proba)
    baseline_brier = y.mean() * (1 - y.mean())
    brier_skill = 1 - (brier / baseline_brier)

    print(f"\n--- Brier Score (Probabilistic Accuracy) ---")
    print(f"  Brier Score:          {brier:.4f}  (lower = better, 0.0 = perfect)")
    print(f"  No-skill baseline:    {baseline_brier:.4f}")
    print(f"  Brier Skill Score:    {brier_skill:.3f}  (higher = better, 1.0 = perfect)")

    # ========================================
    # Calibration Curve — "does 70% really mean 70%?"
    # ========================================
    # Use cross-validated predictions to avoid overfitting bias
    cv_proba = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, cv_proba, n_bins=8, strategy="quantile")

    print(f"\n--- Calibration Curve (Cross-Validated) ---")
    print(f"  {'Predicted':>12} {'Actual':>12} {'Gap':>10} {'Status':>10}")
    print(f"  {'-'*46}")
    total_gap = 0
    for pred_p, actual_p in zip(mean_pred, fraction_pos):
        gap = actual_p - pred_p
        status = "✓ Calibrated" if abs(gap) < 0.05 else ("↑ Underconfident" if gap > 0 else "↓ Overconfident")
        print(f"  {pred_p:>12.3f} {actual_p:>12.3f} {gap:>+10.3f} {status:>16}")
        total_gap += abs(gap)
    mean_cal_error = total_gap / len(mean_pred)
    print(f"  Mean Calibration Error: {mean_cal_error:.4f} (0.0 = perfect)")

    # Store calibration data for visualization
    calibration_data = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": fraction_pos,
        "calibration_gap": fraction_pos - mean_pred,
    })

    # ========================================
    # Cross-category accuracy breakdown
    # ========================================
    print(f"\n--- Accuracy by Category ---")
    print(f"  {'Category':<18} {'N':>5} {'Accuracy':>10} {'Mean Score':>12} {'Brier':>8}")
    print(f"  {'-'*55}")

    cat_col = None
    for col in ["category", "category_clean"]:
        if col in df.columns:
            cat_col = col
            break

    category_stats = {}
    if cat_col:
        df_eval = df.copy()
        df_eval["predicted_proba"] = y_proba
        df_eval["correct"] = (best_model.predict(X) == y)

        for cat, grp in df_eval.groupby(cat_col):
            if len(grp) < 5:
                continue
            cat_y = grp["prediction_accurate"].values
            cat_p = grp["predicted_proba"].values
            cat_brier = brier_score_loss(cat_y, cat_p)
            category_stats[cat] = {
                "n": len(grp),
                "accuracy": grp["correct"].mean(),
                "mean_score": cat_p.mean(),
                "brier": cat_brier,
            }
            print(f"  {str(cat):<18} {len(grp):>5} {grp['correct'].mean():>10.1%} "
                  f"{cat_p.mean():>12.3f} {cat_brier:>8.4f}")
    else:
        print("  (category column not found — skipping)")

    evaluation = {
        "best_model": best_model_name,
        "cv_results": results,
        "feature_importances": importance_df,
        "auc_roc": roc_auc_score(y, y_proba),
        "avg_precision": average_precision_score(y, y_proba),
        "brier_score": brier,
        "brier_skill_score": brier_skill,
        "mean_calibration_error": mean_cal_error,
        "calibration_data": calibration_data,
        "category_stats": category_stats,
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  AUC-ROC:             {evaluation['auc_roc']:.4f}")
    print(f"  Average Precision:   {evaluation['avg_precision']:.4f}")
    print(f"  Brier Score:         {evaluation['brier_score']:.4f}  (baseline: {baseline_brier:.4f})")
    print(f"  Brier Skill Score:   {evaluation['brier_skill_score']:.4f}  (0=no skill, 1=perfect)")
    print(f"  Mean Calib. Error:   {evaluation['mean_calibration_error']:.4f}  (0=perfectly calibrated)")

    return best_model, evaluation, importance_df


def score_open_questions(model, open_df, feature_cols):
    """Score open/active prediction market questions."""
    X_open = open_df[feature_cols].fillna(0).values

    proba = model.predict_proba(X_open)[:, 1]
    open_df = open_df.copy()
    open_df["accuracy_score"] = proba
    open_df["reliability_tier"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    scored = open_df.sort_values("accuracy_score", ascending=False)

    print("\n" + "=" * 60)
    print("OPEN QUESTION RELIABILITY SCORES")
    print("=" * 60)
    print(f"\n{'Score':>6} {'Tier':<12} {'Prediction':>5} {'Title'}")
    print("-" * 80)
    for _, row in scored.head(15).iterrows():
        title = row["title"][:55] + "..." if len(str(row["title"])) > 55 else row["title"]
        pred = f"{row.get('community_pred', 0):.0%}" if pd.notna(row.get("community_pred")) else "N/A"
        print(f"  {row['accuracy_score']:.3f} {str(row['reliability_tier']):<12} {pred:>5}  {title}")

    tier_dist = scored["reliability_tier"].value_counts()
    print(f"\nReliability Distribution:")
    for tier, count in tier_dist.items():
        print(f"  {tier}: {count} questions")

    return scored


# Run model training
model, evaluation, importance_df = train_accuracy_model(df_features, feature_cols)

# Score open questions
open_prepared = prepare_open_questions(open_df, feature_cols)
scored_questions = score_open_questions(model, open_prepared, feature_cols)
