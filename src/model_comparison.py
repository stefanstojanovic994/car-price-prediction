# ============================================================
# POREĐENJE REGRESIONIH MODELA
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import (
    create_preprocessor,
    split_features_target
)


# Učitavanje podataka
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "cars.csv"

original_df = pd.read_csv(data_path)
cleaned_df = clean_data(original_df)
featured_df = add_features(cleaned_df)

X, y = split_features_target(featured_df)

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Trening skup: {len(X_train)} redova")
print(f"Test skup: {len(X_test)} redova")

# Modeli koje upoređujemo
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Treniranje i evaluacija svakog modela
for model_name, model in models.items():
    print(f"\nTreniranje modela: {model_name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE:  {mae:.2f} USD")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"R²:   {r2:.4f}")

# Tabela rezultata
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nPOREĐENJE MODELA")
print(results_df.round(2).to_string(index=False))

# Čuvanje rezultata poređenja
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

results_path = models_dir / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print(f"\nRezultati su sačuvani u: {results_path}")