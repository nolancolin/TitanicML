import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["Pclass", "Age", "Sex", "FamilySize", "IsAlone", "Fare"]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    return data[FEATURES]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("impute", SimpleImputer(strategy="median"), ["Age", "Fare"]),
        ("numeric", "passthrough", ["Pclass", "FamilySize", "IsAlone"]),
        ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"])
    ])
