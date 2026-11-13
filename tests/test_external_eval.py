import pandas as pd
import pytest

from gh_signals.external_eval import (
    EXTERNAL_REPOS_PATH,
    EXTERNAL_SCORES_PATH,
    MODEL_COLUMNS,
    TRUSTED_FROM,
    UNCERTAIN_FROM,
    band,
    score_repos,
)

FEATURES_PATH = EXTERNAL_REPOS_PATH.parent / "features.csv"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, "trusted"),
        (TRUSTED_FROM, "trusted"),
        (TRUSTED_FROM - 0.1, "uncertain"),
        (UNCERTAIN_FROM, "uncertain"),
        (UNCERTAIN_FROM - 0.1, "low_confidence"),
        (0, "low_confidence"),
    ],
)
def test_band_boundaries(score: float, expected: str) -> None:
    assert band(score) == expected


def test_score_is_the_mean_of_both_models_on_a_zero_to_one_hundred_scale() -> None:
    labeled = pd.read_csv(FEATURES_PATH)

    scores = score_repos(labeled, labeled.head(10), MODEL_COLUMNS)

    assert scores["trust_score"].between(0, 100).all()
    assert (scores["trust_score"] == scores[["lr_score", "rf_score"]].mean(axis=1)).all()


def test_external_repos_are_new_unique_and_all_scored() -> None:
    external = pd.read_csv(EXTERNAL_REPOS_PATH)
    labeled = pd.read_csv(FEATURES_PATH)
    known = set(labeled["full_name"].str.lower()) | set(labeled["listed_as"].str.lower())

    assert external["full_name"].str.lower().is_unique
    assert not known & set(external["full_name"].str.lower())
    assert 15 <= len(external) <= 20

    scores = pd.read_csv(EXTERNAL_SCORES_PATH)
    assert scores["full_name"].str.lower().tolist() == external["full_name"].str.lower().tolist()
    assert scores["band"].tolist() == [band(value) for value in scores["trust_score"]]
