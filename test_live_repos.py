
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from gh_signals.github_client import GitHubClient
from gh_signals.build_dataset import DATA_DIR, build_row, fetch_record


# Get project directory and load original datasets
FILE = DATA_DIR.parent

train_df = pd.read_csv(DATA_DIR / "train_split.csv")
test_df = pd.read_csv(DATA_DIR / "test_split.csv")


# New live GitHub repositories
LIVE_REPOS = [
    "huggingface/transformers",
    "pytest-dev/pytest",
    "godotengine/godot",
    "fastapi/fastapi",
    "encode/httpx",
    "astral-sh/ruff",
    "pallets/click",
]


# Fetch live repository data using the existing project pipeline
async def collect_live_repos():
    as_of = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(4)

    async with GitHubClient(
        cache_dir=Path(".cache/github"),
        cache_ttl_seconds=24 * 60 * 60,
    ) as client:

        tasks = [
            fetch_record(client, semaphore, repo_name, "external")
            for repo_name in LIVE_REPOS
        ]

        fetched = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

    rows = []
    errors = []

    for repo_name, result in zip(LIVE_REPOS, fetched):

        if isinstance(result, Exception):
            errors.append({
                "full_name": repo_name,
                "error": str(result)
            })
            continue

        try:
            row = build_row(result, as_of)
            rows.append(row)

        except Exception as exc:
            errors.append({
                "full_name": repo_name,
                "error": str(exc)
            })

    return pd.DataFrame(rows), pd.DataFrame(errors)


# Run collection
live_df, fetch_errors = asyncio.run(collect_live_repos())

print(f"Successfully fetched: {len(live_df)}")
print(f"Fetch errors: {len(fetch_errors)}")

if not fetch_errors.empty:
    print(fetch_errors.to_string(index=False))

if live_df.empty:
    raise ValueError("No live repositories were collected.")


# Check that repositories were not in the original dataset
original_names = set(train_df["full_name"].dropna().str.lower())
original_names.update(test_df["full_name"].dropna().str.lower())

overlap = sorted(
    set(live_df["full_name"].dropna().str.lower())
    .intersection(original_names)
)

if overlap:
    raise ValueError(
        f"These repositories already exist in the original dataset: {overlap}"
    )

print("No overlap found with original training/test data.")


# Features
signals = [
    "stargazers_count",
    "forks_count",
    "subscribers_count",
    "open_issues_count",
    "size",
    "age_days",
    "days_since_push",
    "fork_star_ratio",
    "subscriber_star_ratio",
    "issues_per_star",
    "stars_per_day",
    "has_license",
    "is_org_owned",
    "contributor_activity_ratio",
    "ghost_contributor_ratio",
]


# Prepare training and live repository data
X_train = train_df[signals].astype(float)
y_train = train_df["target_label"]

X_test = test_df[signals].astype(float)
y_test = test_df["target_label"]

X_live = live_df[signals].astype(float)


# Logistic Regression
lr = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr.fit(X_train, y_train)

lr_test_probs = lr.predict_proba(X_test)[:, 1]
lr_live_probs = lr.predict_proba(X_live)[:, 1]


# Random Forest
rf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ))
])

rf.fit(X_train, y_train)

rf_test_probs = rf.predict_proba(X_test)[:, 1]
rf_live_probs = rf.predict_proba(X_live)[:, 1]


# Evaluate models using the original test split
print("\n--- Original Test Split Metrics ---")

print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, lr_test_probs):.3f}")
print(f"Random Forest ROC-AUC:       {roc_auc_score(y_test, rf_test_probs):.3f}")


# Additional evaluation metrics
lr_predictions = (lr_test_probs >= 0.5).astype(int)
rf_predictions = (rf_test_probs >= 0.5).astype(int)

print(f"\nLogistic Regression Accuracy:  {accuracy_score(y_test, lr_predictions):.3f}")
print(f"Logistic Regression Precision: {precision_score(y_test, lr_predictions, zero_division=0):.3f}")
print(f"Logistic Regression Recall:    {recall_score(y_test, lr_predictions, zero_division=0):.3f}")

print(f"\nRandom Forest Accuracy:  {accuracy_score(y_test, rf_predictions):.3f}")
print(f"Random Forest Precision: {precision_score(y_test, rf_predictions, zero_division=0):.3f}")
print(f"Random Forest Recall:    {recall_score(y_test, rf_predictions, zero_division=0):.3f}")


# Live repository results and trust scores
eval_results = live_df[
    [
        "full_name",
        "age_days",
        "stargazers_count",
        "forks_count",
        "target_label",
    ]
].copy()

# External repositories have no verified labels.
eval_results = eval_results.drop(columns=["target_label"])

eval_results["lr_trust_score"] = lr_live_probs.round(3)
eval_results["rf_trust_score"] = rf_live_probs.round(3)

eval_results["lr_prediction"] = (lr_live_probs >= 0.5).astype(int)
eval_results["rf_prediction"] = (rf_live_probs >= 0.5).astype(int)

eval_results["lr_predicted_class"] = eval_results["lr_prediction"].map({
    1: "good",
    0: "uncertain"
})

eval_results["rf_predicted_class"] = eval_results["rf_prediction"].map({
    1: "good",
    0: "uncertain"
})


# Display results sorted by Random Forest trust score
eval_results = eval_results.sort_values(
    by="rf_trust_score",
    ascending=False
)

print("\n--- Live Repository Predictions ---")
print(eval_results.to_string(index=False))


# Include individual signal values for analysis
signal_review = live_df[["full_name"] + signals]

signal_review = eval_results[
    [
        "full_name",
        "lr_trust_score",
        "rf_trust_score",
        "lr_predicted_class",
        "rf_predicted_class",
    ]
].merge(
    signal_review,
    on="full_name",
    how="left"
)

print("\n--- Repository Signal Values ---")
print(signal_review.to_string(index=False))


# Export results to CSV
output_path = DATA_DIR / "live_eval_results.csv"

eval_results.to_csv(output_path, index=False)

print(f"\nResults exported to: {output_path}")
