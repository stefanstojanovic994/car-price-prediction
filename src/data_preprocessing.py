# ============================================================
# PRETPROCESIRANJE PODATAKA
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import clean_data
from feature_engineering import add_features


# Ciljna promenljiva
TARGET_COLUMN = "priceUSD"

# Numeričke karakteristike koje koristimo za model
NUMERIC_FEATURES = [
    "mileage_km",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car"
]

# Kategorijske karakteristike koje koristimo za model
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]


def split_features_target(df):
    """
    Razdvaja ulazne karakteristike od ciljne promenljive.
    """

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def create_preprocessor():
    """
    Pravi preprocessing pipeline za numeričke i kategorijske kolone.
    """

    # Obrada numeričkih kolona
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Obrada kategorijskih kolona
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Spajanje numeričke i kategorijske obrade
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


if __name__ == "__main__":
    # Putanja do skupa podataka
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cars.csv"

    # Učitavanje i priprema podataka
    original_df = pd.read_csv(data_path)
    cleaned_df = clean_data(original_df)
    featured_df = add_features(cleaned_df)

    X, y = split_features_target(featured_df)

    # Testiranje preprocessing pipeline-a
    preprocessor = create_preprocessor()
    X_prepared = preprocessor.fit_transform(X)

    print(f"Broj redova: {X.shape[0]}")
    print(f"Broj ulaznih kolona pre pretprocesiranja: {X.shape[1]}")
    print(
        "Broj kolona posle pretprocesiranja: "
        f"{X_prepared.shape[1]}"
    )
    print(f"Broj ciljnih vrednosti: {len(y)}")
    print("\nPretprocesiranje je uspešno završeno.")