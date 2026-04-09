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
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Contained Bootstrap — makes this block runnable without prior blocks
# ═══════════════════════════════════════════════════════════════════════════════
def _run_bootstrap_pipeline():
    """Full pipeline from synthetic data. Returns all needed variables."""
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
    d_o["reliability_tier"] = _pd.cut(o_p,[0,.3,.5,.7,.85,1.],
                                       labels=["Very Low","Low","Medium","High","Very High"])
    scored = d_o.sort_values("accuracy_score",ascending=False)

    print(f"[Bootstrap] ✓ {len(d_r)} samples | AUC: {float(_auc(y,y_p)):.3f} | "
          f"Brier Skill: {float(bss):.3f} | Cal Error: {mce:.3f}")
    return r_df, o_df, None, d_r, fcols, mdl, evl, imp_df, d_o, scored
# ═══════════════════════════════════════════════════════════════════════════════


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


# ─── Bootstrap Check ─────────────────────────────────────────────────────────
if 'model' not in globals() or 'feature_cols' not in globals():
    print("[Block 6 Bootstrap] Prior blocks not detected — running full pipeline...")
    (_bs_r, _bs_o, _bs_f, df_features, feature_cols, model, evaluation,
     importance_df, open_prepared, scored_questions) = _run_bootstrap_pipeline()
    resolved_df, open_df, fred_df = _bs_r, _bs_o, _bs_f

# analyze_prediction_with_claude is defined in Block 5 (shared namespace).
# If Block 5 was not run, define a local fallback here.
if 'analyze_prediction_with_claude' not in globals():
    def analyze_prediction_with_claude(question_title, accuracy_score,
                                       community_prediction, prediction_count,
                                       feature_insights, category):
        tier = "High" if accuracy_score > 0.7 else "Medium" if accuracy_score > 0.5 else "Low"
        crowd = "strong" if prediction_count > 100 else "moderate" if prediction_count > 30 else "limited"
        return (f"**Reliability: {tier}** — Score {accuracy_score:.1%} | "
                f"Community: {community_prediction:.1%} ({prediction_count} forecasters, {crowd} crowd). "
                f"Category: {category}. "
                f"{'Strong consensus may indicate overconfidence.' if community_prediction > 0.8 or community_prediction < 0.2 else 'Moderate consensus; genuine uncertainty remains.'}")
# ─────────────────────────────────────────────────────────────────────────────

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
