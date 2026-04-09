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


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Contained Bootstrap — makes this block runnable without prior blocks
# ═══════════════════════════════════════════════════════════════════════════════
def _run_bootstrap_pipeline():
    """Full pipeline from synthetic data. Returns all needed variables as a tuple."""
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
            d["resolve_month"]    = now.month
            d["resolve_dayofweek"]= now.dayofweek
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
    bs     = float(_brier(y,y_p))
    bss    = 1-(bs/(y.mean()*(1-y.mean())))
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
    d_o["reliability_tier"] = _pd.cut(o_p,[0,.3,.5,.7,.85,1.],
                                       labels=["Very Low","Low","Medium","High","Very High"])
    scored = d_o.sort_values("accuracy_score",ascending=False)

    print(f"[Bootstrap] ✓ {len(d_r)} samples | AUC: {float(_auc(y,y_p)):.3f} | "
          f"Brier Skill: {float(bss):.3f} | Cal Error: {mce:.3f}")
    return r_df, o_df, None, d_r, fcols, mdl, evl, imp_df, d_o, scored
# ═══════════════════════════════════════════════════════════════════════════════


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


# ─── Bootstrap Check ─────────────────────────────────────────────────────────
# Runs automatically when this block is executed without prior blocks
if 'df_features' not in globals():
    print("[Block 3 Bootstrap] Prior blocks not detected — running full pipeline...")
    (_bs_r, _bs_o, _bs_f, df_features, feature_cols, model, evaluation,
     importance_df, open_prepared, scored_questions) = _run_bootstrap_pipeline()
    resolved_df, open_df, fred_df = _bs_r, _bs_o, _bs_f
    print("[Bootstrap] ✓ All variables ready — skipping normal execution below.")
else:
    # Normal flow: prior blocks provided df_features
    model, evaluation, importance_df = train_accuracy_model(df_features, feature_cols)

    # prepare_open_questions is defined in Block 2 (available in shared namespace)
    # Falls back to bootstrap if Block 2 functions are not in scope
    try:
        open_prepared = prepare_open_questions(open_df, feature_cols)
    except NameError:
        print("[Block 3] prepare_open_questions not found — running bootstrap for open scoring...")
        _, _, _, _, _, _, _, _, open_prepared, scored_questions = _run_bootstrap_pipeline()
    else:
        scored_questions = score_open_questions(model, open_prepared, feature_cols)
# ─────────────────────────────────────────────────────────────────────────────
