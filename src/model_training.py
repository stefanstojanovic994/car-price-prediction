# ============================================================
# TRENIRANJE FINALNOG MODELA
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import (
    create_preprocessor,
    split_features_target
)


# Putanje projekta
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "cars.csv"
models_dir = project_root / "models"
model_path = models_dir / "car_price_model.joblib"

# Učitavanje i priprema podataka
original_df = pd.read_csv(data_path)
cleaned_df = clean_data(original_df)
featured_df = add_features(cleaned_df)

X, y = split_features_target(featured_df)

# Ista podela kao prilikom poređenja modela
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Finalni Random Forest model
final_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    (
        "model",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("Treniranje finalnog Random Forest modela...")

final_model.fit(X_train, y_train)

# Čuvanje treniranog pipeline-a i modela
models_dir.mkdir(exist_ok=True)

joblib.dump(
    final_model,
    model_path,
    compress=3
)

model_size_mb = model_path.stat().st_size / (1024 ** 2)

print("Finalni model je uspešno istreniran.")
print(f"Model je sačuvan u: {model_path}")
print(f"Veličina modela: {model_size_mb:.2f} MB")