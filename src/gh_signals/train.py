"""Fit the live trust model and write ``models/trust_model.json``.

The saved model uses the signals a single repository lookup can compute quickly: the repository
payload, the owner profile, and daily star history. Contributor stats stay out. They are slow to
fetch and, on this dataset, weak on their own.

Run ``uv run python -m gh_signals.train`` after the shared dataset is built.
"""

from collections.abc import Iterable
import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gh_signals.build_dataset import MANIFEST_PATH, TEST_PATH, TRAIN_PATH
from gh_signals.score import FEATURE_NAMES, MODEL_PATH, TrainedModel, predict_probability


def main() -> None:
    """Fit on the train split, check the manual score against scikit-learn, and write the artifact."""
    train_rows = pd.read_csv(TRAIN_PATH)
    test_rows = pd.read_csv(TEST_PATH)
    train_frame = _numeric(train_rows)
    test_frame = _numeric(test_rows)
    labels_train = train_rows["target_label"].astype(int)
    labels_test = test_rows["target_label"].astype(int)

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(train_frame, labels_train)
    probabilities = pipeline.predict_proba(test_frame)[:, 1]
    auc = float(roc_auc_score(labels_test, probabilities))

    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    classifier = pipeline.named_steps["model"]
    if not isinstance(imputer, SimpleImputer):
        raise TypeError("Expected the impute step to be a SimpleImputer.")
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Expected the scale step to be a StandardScaler.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Expected the model step to be a LogisticRegression.")
    classes = [int(label) for label in classifier.classes_]
    if classes != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], got {classes}.")

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        snapshot = str(json.load(file)["as_of"])

    model = TrainedModel(
        feature_names=list(FEATURE_NAMES),
        medians=_floats(imputer.statistics_),
        means=_floats(scaler.mean_),
        scales=_floats(scaler.scale_),
        coefficients=_floats(classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        snapshot=snapshot,
        n_train=len(train_frame),
        n_test=len(test_frame),
        test_roc_auc=auc,
    )
    _check_matches_sklearn(model, test_frame, float(probabilities[0]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as file:
        file.write(model.model_dump_json(indent=2))
        file.write("\n")

    print(f"Test ROC-AUC: {auc:.3f}")
    ranked = sorted(zip(FEATURE_NAMES, model.coefficients, strict=True), key=lambda item: abs(item[1]), reverse=True)
    print("Largest standardized coefficients:")
    for name, coefficient in ranked[:8]:
        print(f"  {name}: {coefficient:.3f}")
    print(f"Wrote {MODEL_PATH}")


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.loc[:, list(FEATURE_NAMES)].astype("float64")
    if list(numeric.columns) != list(FEATURE_NAMES):
        raise RuntimeError("Feature columns were reordered while converting to numbers.")
    return numeric


def _floats(values: Iterable[float] | None) -> list[float]:
    if values is None:
        raise RuntimeError("A fitted model step did not produce a numeric array.")
    return [float(value) for value in values]


def _check_matches_sklearn(model: TrainedModel, test_frame: pd.DataFrame, probability: float) -> None:
    """Fail the training run if the saved arithmetic drifts from scikit-learn."""
    sample = {name: float(test_frame.iloc[0][name]) for name in FEATURE_NAMES}
    manual, _contributions = predict_probability(model, sample)
    if abs(manual - probability) > 1e-6:
        raise RuntimeError(f"Saved model probability {manual:.6f} does not match scikit-learn {probability:.6f}.")


if __name__ == "__main__":
    main()
