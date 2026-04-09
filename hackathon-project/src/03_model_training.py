"""
PredictPulse — Block 3: Model Training & Evaluation
=====================================================
Trains an ensemble model to predict which prediction market
forecasts are likely to be accurate, with full evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
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

    evaluation = {
        "best_model": best_model_name,
        "cv_results": results,
        "feature_importances": importance_df,
        "auc_roc": roc_auc_score(y, y_proba),
        "avg_precision": average_precision_score(y, y_proba),
    }

    print(f"\nFinal AUC-ROC: {evaluation['auc_roc']:.3f}")
    print(f"Average Precision: {evaluation['avg_precision']:.3f}")

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
