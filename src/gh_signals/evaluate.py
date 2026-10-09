"""Reports how well each star timing feature separates uncertain repositories from good ones."""

import csv

from gh_signals.star_burstiness import FEATURES_PATH

FEATURE_DIRECTIONS = {"max_day_share": 1, "active_day_ratio": -1, "burstiness": 1}


def auc(positives: list[float], negatives: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def main() -> None:
    with FEATURES_PATH.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["status"] == "ok"]

    print("AUC, uncertain as positive class:")
    for feature, direction in FEATURE_DIRECTIONS.items():
        uncertain = [direction * float(row[feature]) for row in rows if row["label"] == "uncertain"]
        good = [direction * float(row[feature]) for row in rows if row["label"] == "good"]
        print(f"  {feature}: {auc(uncertain, good):.3f}")

    print("Top 5 by max_day_share:")
    for row in sorted(rows, key=lambda row: float(row["max_day_share"]), reverse=True)[:5]:
        print(f"  https://github.com/{row['repo_name']} ({row['label']}): {float(row['max_day_share']):.1%}")


if __name__ == "__main__":
    main()
