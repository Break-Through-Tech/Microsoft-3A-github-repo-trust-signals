
from dataclasses import dataclass


# Trust-tier thresholds
HIGH_THRESHOLD = 0.80
MODERATE_THRESHOLD = 0.60
LOW_THRESHOLD = 0.40

UNCERTAIN_LABEL = "Uncertain"


@dataclass
class TrustResult:
    lr_score: float
    rf_score: float
    composite_score: float
    lr_prediction: str
    rf_prediction: str
    trust_tier: str
    explanation: str


def classify_prediction(score: float) -> str:
    """Convert a good-class score into a binary model prediction."""
    return "good" if score >= 0.50 else "uncertain"


def calculate_trust_score(
    lr_score: float,
    rf_score: float
) -> TrustResult:
    """
    Combines the Logistic Regression and Random Forest scores.

    Model disagreement will always result in an Uncertain tier.
    Scores are assumed to represent the models' scores for 'good' classification.
    """

    # Validate input scores
    if not 0 <= lr_score <= 1:
        raise ValueError("LR score must be between 0 and 1.")

    if not 0 <= rf_score <= 1:
        raise ValueError("RF score must be between 0 and 1.")

    # Calculate each model's prediction
    lr_prediction = classify_prediction(lr_score)
    rf_prediction = classify_prediction(rf_score)

    # Equal-weight composite score
    composite_score = (lr_score + rf_score) / 2

    # Model disagreement takes priority
    if lr_prediction != rf_prediction:
        tier = "Uncertain"
        explanation = (
            "Models disagree. Manual review is recommended."
        )

    # Explicit uncertainty range
    elif LOW_THRESHOLD <= composite_score < MODERATE_THRESHOLD:
        tier = "Uncertain"
        explanation = (
            "Composite score falls within the uncertainty range."
        )

    # High trust
    elif composite_score >= HIGH_THRESHOLD:
        tier = "High"
        explanation = (
            "Both models agree the repository is good "
            "with a high composite score."
        )

    # Moderate trust
    elif composite_score >= MODERATE_THRESHOLD:
        tier = "Moderate"
        explanation = (
            "Both models agree the repository is good "
            "with a moderate composite score."
        )

    # Low trust / caution
    else:
        tier = "Low / Caution"
        explanation = (
            "Both models agree the repository is uncertain "
            "and the composite score is low."
        )

    return TrustResult(
        lr_score=lr_score,
        rf_score=rf_score,
        composite_score=round(composite_score, 4),
        lr_prediction=lr_prediction,
        rf_prediction=rf_prediction,
        trust_tier=tier,
        explanation=explanation
    )
