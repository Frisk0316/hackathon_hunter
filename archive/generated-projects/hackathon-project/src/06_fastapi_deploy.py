"""
PredictPulse — FastAPI Deployment (Zerve)
==========================================
貼入 Zerve FastAPI 部署的程式碼欄位。
啟動後會自動訓練模型（約 10 秒），然後開始接受請求。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PredictPulse API",
    description="AI-powered prediction market reliability scorer. "
                "Input any prediction question metadata, get a reliability score + AI analysis.",
    version="1.0.0",
)

# ─── Request / Response Schemas ───────────────────────────────────────────────
class PredictionRequest(BaseModel):
    title: str                          = Field(..., example="Will global temperature exceed 1.5°C before 2030?")
    community_prediction: float         = Field(0.5,  ge=0.0, le=1.0, example=0.62)
    prediction_count: int               = Field(0,    ge=0,            example=245)
    description_length: int             = Field(0,    ge=0,            example=1200)
    num_comments: int                   = Field(0,    ge=0,            example=48)
    question_age_days: float            = Field(0.0,  ge=0.0,          example=180.0)
    category: Optional[str]             = Field("Unknown",             example="Science")

class FactorItem(BaseModel):
    feature: str
    value: float
    importance: float
    direction: str

class PredictionResponse(BaseModel):
    reliability_score: float
    reliability_tier: str
    community_prediction: float
    analysis: str
    top_factors: list
    metadata: dict

# ─── Bootstrap: train model on startup ────────────────────────────────────────
_model = None
_feature_cols = None
_df_features = None

def _bootstrap():
    """Train model from synthetic data at startup."""
    import numpy as _np, pandas as _pd
    from datetime import datetime as _dt, timedelta as _td
    from sklearn.ensemble import RandomForestClassifier as _RFC
    import warnings as _w; _w.filterwarnings("ignore")

    _np.random.seed(42); _rng = _np.random.default_rng(42)
    _CATS = ["Science","Economics","Politics","Technology","Health",
             "Environment","Sports","Finance","Social","Other"]
    _CW   = [0.20,0.18,0.16,0.14,0.10,0.08,0.06,0.04,0.02,0.02]
    _BASE = _dt(2021,1,1)

    rows = []
    for i in range(500):
        ls  = int(_rng.integers(30,730))
        cre = _BASE + _td(days=int(_rng.integers(0,1000).item()))
        clo = cre + _td(days=ls)
        res = clo + _td(days=int(_rng.integers(0,30).item()))
        np_ = int(_np.clip(_rng.lognormal(3.5,1.2),3,2000))
        nc  = int(_np.clip(_rng.lognormal(2.0,1.0),0,300))
        cp  = float(_np.clip(_rng.beta(2,2),0.02,0.98))
        cat = str(_rng.choice(_CATS,p=_CW))
        dl  = int(_np.clip(_rng.lognormal(5.5,0.8),100,5000))
        conf= abs(cp-0.5)*2
        acc = _np.clip(0.55+0.15*_np.log1p(np_/(ls+1))/5-0.05*conf,0.3,0.85)
        rv  = str(1.0 if (cp>0.5)==(_rng.random()<acc) else 0.0)
        rows.append({"created_time":cre.isoformat(),"resolve_time":res.isoformat(),
                     "close_time":clo.isoformat(),"prediction_count":np_,
                     "community_prediction":cp,"resolution":rv,"category":cat,
                     "description_length":dl,"num_comments":nc,
                     "title":f"Will {cat.lower()} event #{i} occur by {res.year}?"})
    d = _pd.DataFrame(rows)
    for c in ["created_time","resolve_time","close_time"]:
        d[c] = _pd.to_datetime(d[c], errors="coerce")
    d["question_lifespan_days"] = (d["resolve_time"]-d["created_time"]).dt.total_seconds()/86400
    d["time_to_close_days"]     = (d["close_time"]-d["created_time"]).dt.total_seconds()/86400
    d["resolve_month"]          = d["resolve_time"].dt.month
    d["resolve_dayofweek"]      = d["resolve_time"].dt.dayofweek
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
    d = _pd.concat([d,dums],axis=1)
    d["resolution_float"]    = _pd.to_numeric(d["resolution"],errors="coerce")
    d = d[d["resolution_float"].isin([0.0,1.0])].copy()
    d["prediction_error"]    = abs(d["community_pred"]-d["resolution_float"])
    d["prediction_accurate"] = (d["prediction_error"]<0.3).astype(int)
    d["accuracy_score"]      = 1-d["prediction_error"]

    fcols = ["question_lifespan_days","time_to_close_days","resolve_month","resolve_dayofweek",
             "log_prediction_count","log_comments","engagement_ratio","prediction_density",
             "log_description_length","title_length","title_word_count","has_number_in_title",
             "is_yes_no_question","community_pred","pred_confidence","pred_extreme"]
    fcols += [c for c in d.columns if c.startswith("cat_")]
    d = d.dropna(subset=fcols+["prediction_accurate"])

    X = d[fcols].fillna(0).values; y = d["prediction_accurate"].values
    mdl = _RFC(n_estimators=200,max_depth=8,min_samples_leaf=5,random_state=42)
    mdl.fit(X,y)
    print(f"[PredictPulse] Model ready — {len(d)} training samples, {len(fcols)} features")
    return mdl, fcols, d

@app.on_event("startup")
def startup_event():
    global _model, _feature_cols, _df_features
    _model, _feature_cols, _df_features = _bootstrap()

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "PredictPulse API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /predict": "Score a prediction market question's reliability",
            "GET /health":   "Health check",
            "GET /docs":     "Interactive API documentation",
        },
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "training_samples": len(_df_features) if _df_features is not None else 0,
        "features": len(_feature_cols) if _feature_cols is not None else 0,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Try again in a few seconds.")

    # Engineer features
    features = {
        "question_lifespan_days": req.question_age_days,
        "time_to_close_days":     req.question_age_days * 1.5,
        "resolve_month":          pd.Timestamp.now().month,
        "resolve_dayofweek":      pd.Timestamp.now().dayofweek,
        "log_prediction_count":   np.log1p(req.prediction_count),
        "log_comments":           np.log1p(req.num_comments),
        "engagement_ratio":       req.num_comments / (req.prediction_count + 1),
        "prediction_density":     req.prediction_count / (req.question_age_days + 1),
        "log_description_length": np.log1p(req.description_length),
        "title_length":           len(req.title),
        "title_word_count":       len(req.title.split()),
        "has_number_in_title":    int(any(c.isdigit() for c in req.title)),
        "is_yes_no_question":     int(any(w in req.title.lower() for w in ["will","would","should"])),
        "community_pred":         req.community_prediction,
        "pred_confidence":        abs(req.community_prediction - 0.5) * 2,
        "pred_extreme":           int(req.community_prediction < 0.1 or req.community_prediction > 0.9),
    }
    for col in _feature_cols:
        if col not in features:
            features[col] = 0

    X = pd.DataFrame([features])[_feature_cols].fillna(0).values
    score = float(_model.predict_proba(X)[0, 1])

    tier = ("Very High" if score > 0.85 else "High" if score > 0.7 else
            "Medium"    if score > 0.5  else "Low"  if score > 0.3 else "Very Low")

    importances = _model.feature_importances_
    fv = pd.DataFrame([features])[_feature_cols].fillna(0).iloc[0]
    top_idx = np.argsort(importances * np.abs(fv.values))[::-1][:5]
    top_factors = [{"feature": _feature_cols[i], "value": round(float(fv.iloc[i]),4),
                    "importance": round(float(importances[i]),4),
                    "direction": "positive" if fv.iloc[i] > 0 else "neutral"}
                   for i in top_idx]

    # Analysis text
    crowd = "strong" if req.prediction_count > 100 else "moderate" if req.prediction_count > 30 else "limited"
    analysis = (
        f"**Reliability: {tier}** — Score {score:.1%} | "
        f"Community consensus: {req.community_prediction:.1%} "
        f"({req.prediction_count} forecasters, {crowd} crowd wisdom). "
        f"Category: {req.category}. "
        f"{'Strong consensus may indicate overconfidence bias.' if req.community_prediction > 0.8 or req.community_prediction < 0.2 else 'Moderate consensus; genuine uncertainty remains.'}"
    )

    return PredictionResponse(
        reliability_score=round(score, 4),
        reliability_tier=tier,
        community_prediction=req.community_prediction,
        analysis=analysis,
        top_factors=top_factors,
        metadata={
            "model": "RandomForest",
            "training_samples": len(_df_features),
            "features_used": len(_feature_cols),
            "version": "1.0.0",
        },
    )
