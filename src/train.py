import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.features import build_features, build_preprocessor

TRAIN_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/random_forest_v9.joblib")


def build_model() -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])


def train(train_path: Path, model_path: Path) -> float:
    data = pd.read_csv(train_path)
    X = build_features(data)
    y = data["Survived"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(build_model(), X, y, cv=cv, scoring="accuracy").mean()

    model = build_model().fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return cv_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne la Random Forest V9 sur tout le jeu d'entraînement.")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    cv_accuracy = train(args.train, args.model)
    print(f"Accuracy CV (5 folds) : {cv_accuracy:.4f}")
    print(f"Modèle enregistré : {args.model}")


if __name__ == "__main__":
    main()
