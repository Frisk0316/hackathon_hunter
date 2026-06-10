"""
PredictPulse — Block 5: Claude AI Analysis Engine
===================================================
Uses Anthropic's Claude API to generate natural language
explanations of prediction reliability and market insights.
"""

import json
import os
import pandas as pd

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("anthropic package not installed — will use template-based analysis")


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


def get_claude_client():
    """Initialize Claude API client."""
    if not ANTHROPIC_AVAILABLE:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set. Using template analysis.")
        return None
    return anthropic.Anthropic(api_key=api_key)


def analyze_prediction_with_claude(
    question_title,
    accuracy_score,
    community_prediction,
    prediction_count,
    feature_insights,
    category,
):
    """
    Generate a natural language analysis of a prediction's reliability
    using Claude API.
    """
    client = get_claude_client()

    prompt = f"""You are PredictPulse, an AI prediction market analyst. Analyze this forecast's reliability.

**Prediction Market Question:** {question_title}

**Data Points:**
- Reliability Score: {accuracy_score:.1%} (from our ML model)
- Community Consensus: {community_prediction:.1%} probability
- Number of Forecasters: {prediction_count}
- Category: {category}

**Model Feature Insights:**
{json.dumps(feature_insights, indent=2)}

Provide a concise analysis (150-200 words) covering:
1. **Reliability Assessment**: Is this prediction trustworthy? Why?
2. **Key Factors**: What drives this reliability score?
3. **Caveats**: What could make this prediction wrong?
4. **Confidence Level**: Your overall confidence (Low/Medium/High)

Use precise language. Cite specific numbers. Be honest about uncertainty."""

    if client:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    else:
        return _template_analysis(
            question_title, accuracy_score, community_prediction,
            prediction_count, category
        )


def _template_analysis(title, score, pred, count, category):
    """Fallback template when Claude API is unavailable."""
    tier = (
        "High" if score > 0.7 else
        "Medium" if score > 0.5 else
        "Low"
    )
    crowd = "strong" if count > 100 else "moderate" if count > 30 else "limited"

    return f"""**Reliability Assessment: {tier}**

This {category} prediction has a reliability score of {score:.1%}, indicating {tier.lower()} confidence in its accuracy.

**Key Factors:**
- Community consensus at {pred:.1%} based on {count} forecasters ({crowd} crowd wisdom)
- Historical accuracy for {category} predictions in this confidence range averages {'above' if score > 0.6 else 'below'} baseline

**Caveats:**
- {'Strong consensus can create overconfidence bias' if pred > 0.8 or pred < 0.2 else 'Moderate consensus suggests genuine uncertainty remains'}
- External shocks and black swan events are not captured by historical patterns

**Confidence Level: {tier}** — {'Reliable enough for informed decision-making' if tier == 'High' else 'Exercise caution and seek additional sources' if tier == 'Medium' else 'Treat as speculative; low historical precedent for accuracy'}"""


def generate_market_report(scored_questions, top_n=5):
    """Generate a comprehensive market intelligence report."""

    top = scored_questions.head(top_n)

    report_sections = []
    report_sections.append("# PredictPulse Market Intelligence Report\n")
    report_sections.append(f"*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}*\n")
    report_sections.append(f"**Questions Analyzed:** {len(scored_questions)}")
    report_sections.append(f"**High Reliability (>70%):** {(scored_questions['accuracy_score'] > 0.7).sum()}")
    report_sections.append(f"**Low Reliability (<30%):** {(scored_questions['accuracy_score'] < 0.3).sum()}\n")
    report_sections.append("---\n")

    for i, (_, row) in enumerate(top.iterrows(), 1):
        feature_insights = {
            "prediction_count": int(row.get("prediction_count", 0)),
            "question_age_days": round(row.get("question_lifespan_days", 0), 1),
            "confidence_level": round(row.get("pred_confidence", 0), 3),
            "engagement_ratio": round(row.get("engagement_ratio", 0), 3),
        }

        analysis = analyze_prediction_with_claude(
            question_title=row.get("title", "Unknown"),
            accuracy_score=row.get("accuracy_score", 0),
            community_prediction=row.get("community_pred", 0.5),
            prediction_count=int(row.get("prediction_count", 0)),
            feature_insights=feature_insights,
            category=row.get("category", "Unknown"),
        )

        report_sections.append(f"## {i}. {row.get('title', 'Unknown')}\n")
        report_sections.append(f"**Reliability Score:** {row.get('accuracy_score', 0):.1%} "
                              f"| **Community Prediction:** {row.get('community_pred', 0):.1%} "
                              f"| **Forecasters:** {int(row.get('prediction_count', 0))}\n")
        report_sections.append(analysis)
        report_sections.append("\n---\n")

    full_report = "\n".join(report_sections)
    print(full_report)
    return full_report


# ─── Bootstrap Check ─────────────────────────────────────────────────────────
if 'scored_questions' not in globals():
    print("[Block 5 Bootstrap] Prior blocks not detected — running full pipeline...")
    (_bs_r, _bs_o, _bs_f, df_features, feature_cols, model, evaluation,
     importance_df, open_prepared, scored_questions) = _run_bootstrap_pipeline()
    resolved_df, open_df, fred_df = _bs_r, _bs_o, _bs_f
# ─────────────────────────────────────────────────────────────────────────────

# Generate report for top scored predictions
report = generate_market_report(scored_questions, top_n=5)
