"""
PredictPulse — Block 5: Claude AI Analysis Engine
===================================================
Uses Anthropic's Claude API to generate natural language
explanations of prediction reliability and market insights.
"""

import json
import os

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("anthropic package not installed — will use template-based analysis")


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


import pandas as pd

# Generate report for top scored predictions
report = generate_market_report(scored_questions, top_n=5)
