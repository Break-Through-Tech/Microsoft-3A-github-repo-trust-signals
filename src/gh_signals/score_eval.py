import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from pathlib import Path

# Get directory
DIR = Path(__file__).resolve().parent

# Find files in directory
FILE = DIR.parent.parent
train_df = pd.read_csv(FILE / "data" / "train_split.csv")
test_df = pd.read_csv(FILE / "data" / "test_split.csv")

# generated to test for low scores on young repos, currently missing 10 new live repos
# note: these test cases are not real and are only meant to simulate possible real world repos
test_cases = [
    # Young Legit: Young (45 days), high contributor activity, license, good ratios
    {
        'full_name': 'test_case_young_legit_01',
        'label': 'good',
        'stargazers_count': 4800,
        'forks_count': 380,
        'subscribers_count': 85,
        'open_issues_count': 42,
        'size': 18500,
        'age_days': 45,
        'days_since_push': 1,
        'fork_star_ratio': 380 / 4800,
        'subscriber_star_ratio': 85 / 4800,
        'issues_per_star': 42 / 4800,
        'stars_per_day': 4800 / 45,
        'has_license': 1,
        'is_org_owned': 1,
        'contributor_activity_ratio': 0.35,
        'ghost_contributor_ratio': 0.00,
        'target_label': 1
    },
    # Young Legit: 90 days old, high fork ratio, verified organization owner
    {
        'full_name': 'test_case_young_legit_02',
        'label': 'good',
        'stargazers_count': 3200,
        'forks_count': 410,
        'subscribers_count': 92,
        'open_issues_count': 35,
        'size': 24000,
        'age_days': 90,
        'days_since_push': 0,
        'fork_star_ratio': 410 / 3200,
        'subscriber_star_ratio': 92 / 3200,
        'issues_per_star': 35 / 3200,
        'stars_per_day': 3200 / 90,
        'has_license': 1,
        'is_org_owned': 1,
        'contributor_activity_ratio': 0.42,
        'ghost_contributor_ratio': 0.00,
        'target_label': 1
    },
    # Young Suspicious: 22 days old, 18.5k stars, low forks, zero issues, ghost activity
    {
        'full_name': 'test_case_young_suspicious_01',
        'label': 'uncertain',
        'stargazers_count': 18500,
        'forks_count': 45,
        'subscribers_count': 5,
        'open_issues_count': 0,
        'size': 1200,
        'age_days': 22,
        'days_since_push': 18,
        'fork_star_ratio': 45 / 18500,
        'subscriber_star_ratio': 5 / 18500,
        'issues_per_star': 0 / 18500,
        'stars_per_day': 18500 / 22,
        'has_license': 0,
        'is_org_owned': 0,
        'contributor_activity_ratio': 0.02,
        'ghost_contributor_ratio': 0.08,
        'target_label': 0
    },
    # Young Suspicious: 30 days old, 25k stars, inflated, no license, ghost contributors
    {
        'full_name': 'test_case_young_suspicious_02',
        'label': 'uncertain',
        'stargazers_count': 25000,
        'forks_count': 120,
        'subscribers_count': 8,
        'open_issues_count': 2,
        'size': 850,
        'age_days': 30,
        'days_since_push': 25,
        'fork_star_ratio': 120 / 25000,
        'subscriber_star_ratio': 8 / 25000,
        'issues_per_star': 2 / 25000,
        'stars_per_day': 25000 / 30,
        'has_license': 0,
        'is_org_owned': 0,
        'contributor_activity_ratio': 0.01,
        'ghost_contributor_ratio': 0.09,
        'target_label': 0
    }
]

# Append test cases to df
test_df = pd.concat([test_df, pd.DataFrame(test_cases)], ignore_index=True)

# features
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

X_train = train_df[signals].astype(float)
y_train = train_df["target_label"]
X_test = test_df[signals].astype(float)
y_test = test_df["target_label"]

# logistic regression
lr = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]

# random forest
rf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Summary dataframe and trust scores
eval_results = test_df[['full_name', 'label', 'age_days', 'stargazers_count', 'target_label']].copy()
eval_results['lr_trust_score'] = lr_probs.round(3)
eval_results['rf_trust_score'] = rf_probs.round(3)
eval_results['lr_prediction'] = (lr_probs >= 0.5).astype(int)
eval_results['rf_prediction'] = (rf_probs >= 0.5).astype(int)

# Print Model Metrics
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, lr_probs):.3f}")
print(f"Random Forest ROC-AUC:       {roc_auc_score(y_test, rf_probs):.3f}")

print(eval_results.sort_values(by='rf_trust_score', ascending=False).to_string(index=False))

# Export results to CSV
output_path = FILE / "data" / "eval_results.csv"
eval_results.sort_values(by='rf_trust_score', ascending=False).to_csv(output_path, index=False)