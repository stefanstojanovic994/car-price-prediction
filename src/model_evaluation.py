# ============================================================
# EVALUACIJA FINALNOG MODELA
# ============================================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import split_features_target


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

# Ista podela koja je korišćena tokom treniranja
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Učitavanje sačuvanog modela
model = joblib.load(model_path)

# Predviđanje cena
predictions = model.predict(X_test)

# Izračunavanje metrika
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("EVALUACIJA FINALNOG MODELA")
print(f"MAE:  {mae:.2f} USD")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} USD")
print(f"R²:   {r2:.4f}")

# Čuvanje metrika
metrics_df = pd.DataFrame({
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics_path = models_dir / "evaluation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# Primeri stvarnih i predviđenih cena
examples_df = pd.DataFrame({
    "stvarna_cena": y_test.iloc[:10].values,
    "predvidjena_cena": predictions[:10]
})

examples_df["apsolutna_greska"] = np.abs(
    examples_df["stvarna_cena"] -
    examples_df["predvidjena_cena"]
)

print("\nPRIMERI PREDVIĐANJA")
print(examples_df.round(2).to_string(index=False))

examples_path = models_dir / "prediction_examples.csv"
examples_df.to_csv(examples_path, index=False)

# Grafikon stvarnih i predviđenih cena
plot_limit = y_test.quantile(0.99)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.25,
    color="steelblue"
)

plt.plot(
    [0, plot_limit],
    [0, plot_limit],
    color="red",
    linestyle="--",
    label="Idealno predviđanje"
)

plt.xlim(0, plot_limit)
plt.ylim(0, plot_limit)

plt.title("Stvarne i predviđene cene")
plt.xlabel("Stvarna cena (USD)")
plt.ylabel("Predviđena cena (USD)")
plt.legend()
plt.tight_layout()

plot_path = models_dir / "model_evaluation.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nMetrike su sačuvane u: {metrics_path}")
print(f"Primeri su sačuvani u: {examples_path}")
print(f"Grafikon je sačuvan u: {plot_path}")