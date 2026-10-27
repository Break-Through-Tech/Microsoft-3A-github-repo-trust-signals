"""Scores repositories outside the labeled dataset with the team's combined models.

Fetches the same signals as the shared dataset for every repo in ``data/external_repos.csv``, trains the combined
logistic regression and random forest from ``notebooks/006_combined_model.ipynb`` on all labeled repos, and writes the
signal values and scores. The 0-100 score and its bands are provisional until the trust score task defines the official
ones. Run with ``uv run python -m gh_signals.external_eval``.
"""

import asyncio

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gh_signals.build_dataset import (
    DATA_DIR,
    FEATURES_PATH,
    RAW_DIR,
    SIGNAL_COLUMNS,
    build_row,
    collect,
    read_snapshot_time,
)

EXTERNAL_REPOS_PATH = DATA_DIR / "external_repos.csv"
EXTERNAL_RAW_DIR = RAW_DIR / "external"
EXTERNAL_FEATURES_PATH = DATA_DIR / "external_features.csv"
EXTERNAL_SCORES_PATH = DATA_DIR / "external_scores.csv"

# The signals the combined models in notebooks/006_combined_model.ipynb use.
MODEL_COLUMNS = [
    "stargazers_count",
    "forks_count",
    "subscribers_count",
    "open_issues_count",
    "size",
    "age_days",
    "days_since_push",
    "stars_per_day",
    "has_license",
    "is_org_owned",
    "fork_star_ratio",
    "subscriber_star_ratio",
    "issues_per_star",
    "contributor_activity_ratio",
    "ghost_contributor_ratio",
]
# Diagnostic only: drops the two signals that are mostly a repo's age, to see how much a score depends on age.
NO_AGE_COLUMNS = [column for column in MODEL_COLUMNS if column not in ("age_days", "stars_per_day")]

# Chosen from out-of-fold scores on the labeled repos alone, before scoring any external repo. Out-of-fold, 71 of 82 good
# and 1 of 24 uncertain repos score above 70, and 18 of 24 uncertain and 4 of 82 good repos score below 40. The band
# between is where the evidence is genuinely mixed. The main block prints these counts.
TRUSTED_FROM = 70
UNCERTAIN_FROM = 40
RANDOM_SEED = 42


def make_models() -> dict[str, Pipeline]:
    return {
        "lr": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "rf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=5, class_weight="balanced", random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
    }


def band(score: float) -> str:
    """Names a 0-100 score. Low scores mean little evidence of trust, never that a repo is bad."""
    if score >= TRUSTED_FROM:
        return "trusted"
    if score >= UNCERTAIN_FROM:
        return "uncertain"
    return "low_confidence"


def score_repos(labeled: pd.DataFrame, repos: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scores each repo with both models, fit on every labeled repo. ``trust_score`` is their mean, from 0 to 100."""
    scores = pd.DataFrame(index=repos.index)
    for name, model in make_models().items():
        model.fit(labeled[columns].astype(float), labeled["target_label"])
        scores[f"{name}_score"] = model.predict_proba(repos[columns].astype(float))[:, 1] * 100
    scores["trust_score"] = scores[["lr_score", "rf_score"]].mean(axis=1)
    return scores


def labeled_out_of_fold_scores(labeled: pd.DataFrame) -> pd.Series:
    """Scores every labeled repo with models that never saw it, so the scores can be compared with the external ones."""
    folds = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    features, target = labeled[MODEL_COLUMNS].astype(float), labeled["target_label"]
    probabilities = [
        cross_val_predict(model, features, target, cv=folds, method="predict_proba")[:, 1]
        for model in make_models().values()
    ]
    return pd.Series(np.mean(probabilities, axis=0) * 100, index=labeled["full_name"])


def main() -> None:
    labeled = pd.read_csv(FEATURES_PATH)
    external = pd.read_csv(EXTERNAL_REPOS_PATH)
    known = set(labeled["full_name"].str.lower()) | set(labeled["listed_as"].str.lower())
    overlap = known & set(external["full_name"].str.lower())
    if overlap:
        raise ValueError(f"External repos must not be in the labeled dataset: {sorted(overlap)}")

    snapshot_path = EXTERNAL_RAW_DIR / "snapshot.json"
    as_of = read_snapshot_time(snapshot_path)
    records = asyncio.run(collect([(name, "unlabeled") for name in external["full_name"]], EXTERNAL_RAW_DIR))
    features = pd.DataFrame([build_row(record, as_of) for record in records]).drop(columns=["label", "target_label"])
    features.insert(1, "group", external["group"].to_numpy())

    scores = score_repos(labeled, features, MODEL_COLUMNS)
    scores["band"] = scores["trust_score"].map(band)
    no_age = score_repos(labeled, features, NO_AGE_COLUMNS)["trust_score"]
    scores["trust_score_no_age"] = no_age
    scores["band_no_age"] = no_age.map(band)
    scores["models_disagree"] = (scores["lr_score"] - scores["rf_score"]).abs() >= 30
    summary = features[["full_name", "group", "stargazers_count", "age_days"]].join(scores).round(1)

    features.to_csv(EXTERNAL_FEATURES_PATH, index=False)
    summary.to_csv(EXTERNAL_SCORES_PATH, index=False)
    print(f"Scored {len(summary)} external repos from the snapshot taken {as_of.isoformat()}.")
    held_out = labeled_out_of_fold_scores(labeled)
    print("Labeled repos per band, out-of-fold (for reference):")
    print(pd.crosstab(labeled["label"].to_numpy(), held_out.map(band).to_numpy()).to_string())
    labels = labeled.set_index("full_name")["label"]
    wrong_side = ((labels == "good") & (held_out < UNCERTAIN_FROM)) | (
        (labels == "uncertain") & (held_out >= TRUSTED_FROM)
    )
    print("Labeled repos scored on the wrong side, out-of-fold:")
    print(held_out[wrong_side].round(1).to_frame("score").join(labels).to_string())
    print(
        f"Signals written: {len(SIGNAL_COLUMNS)} columns to {EXTERNAL_FEATURES_PATH.name}; scores to {EXTERNAL_SCORES_PATH.name}."
    )


if __name__ == "__main__":
    main()
