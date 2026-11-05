import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_features
from src.train import MODEL_PATH

TEST_PATH = Path("data/test.csv")
SUBMISSION_PATH = Path("submissions/submission.csv")


def predict(model_path: Path, test_path: Path, output_path: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    test = pd.read_csv(test_path)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": model.predict(build_features(test))
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère une soumission Kaggle avec le modèle entraîné.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"Modèle introuvable : {args.model}. Lance d'abord python -m src.train")

    submission = predict(args.model, args.test, args.output)
    print(f"{len(submission)} prédictions enregistrées dans {args.output}")


if __name__ == "__main__":
    main()
