from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURES, build_features, build_preprocessor
from src.predict import predict
from src.train import train

DATA = Path("data")


@pytest.fixture(scope="module")
def train_data():
    return pd.read_csv(DATA / "train.csv")


@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv(DATA / "test.csv")


def test_kaggle_files_have_expected_shape(train_data, test_data):
    assert train_data.shape == (891, 12)
    assert test_data.shape == (418, 11)
    assert set(train_data.columns) - set(test_data.columns) == {"Survived"}


def test_build_features_returns_model_columns(train_data):
    features = build_features(train_data)

    assert list(features.columns) == FEATURES
    assert len(features) == len(train_data)


def test_family_features():
    passengers = pd.DataFrame({
        "Pclass": [3, 1],
        "Age": [22.0, 38.0],
        "Sex": ["male", "female"],
        "SibSp": [0, 1],
        "Parch": [0, 2],
        "Fare": [7.25, 71.28],
    })

    features = build_features(passengers)

    assert features["FamilySize"].tolist() == [1, 4]
    assert features["IsAlone"].tolist() == [1, 0]


def test_build_features_leaves_input_untouched(train_data):
    columns = list(train_data.columns)
    build_features(train_data)

    assert list(train_data.columns) == columns


def test_preprocessor_fills_missing_values(train_data, test_data):
    preprocessor = build_preprocessor().fit(build_features(train_data))

    for data in (train_data, test_data):
        encoded = preprocessor.transform(build_features(data))
        assert encoded.shape == (len(data), 7)
        assert not pd.isna(encoded).any()


def test_submission_format(tmp_path, test_data):
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "submission.csv"

    train(DATA / "train.csv", model_path)
    predict(model_path, DATA / "test.csv", output_path)
    submission = pd.read_csv(output_path)

    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert submission["PassengerId"].tolist() == test_data["PassengerId"].tolist()
    assert set(submission["Survived"]) <= {0, 1}
