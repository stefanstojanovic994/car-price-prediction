# ============================================================
# ČIŠĆENJE PODATAKA
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


def clean_data(df):
    """
    Čisti podatke o polovnim automobilima i vraća očišćen DataFrame.
    """

    cleaned_df = df.copy()

    # Preimenovanje kolona radi lakšeg korišćenja
    cleaned_df = cleaned_df.rename(columns={
        "mileage(kilometers)": "mileage_km",
        "volume(cm3)": "engine_volume_cm3"
    })

    # Standardizacija tekstualnih vrednosti
    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]

    for column in categorical_columns:
        cleaned_df[column] = (
            cleaned_df[column]
            .str.strip()
            .str.lower()
        )

    # Uklanjanje potpuno dupliranih redova
    cleaned_df = cleaned_df.drop_duplicates()

    # Uklanjanje redova sa nevalidnom cenom
    cleaned_df = cleaned_df[cleaned_df["priceUSD"] > 0]

    # Uklanjanje godišta pre 1930. godine
    cleaned_df = cleaned_df[
        cleaned_df["year"].between(1930, 2019)
    ]

    # Ekstremne kilometraže označavamo kao nedostajuće
    cleaned_df.loc[
        cleaned_df["mileage_km"] > 1_000_000,
        "mileage_km"
    ] = np.nan

    # Nevalidne zapremine motora označavamo kao nedostajuće
    cleaned_df.loc[
        (cleaned_df["engine_volume_cm3"] <= 0) |
        (cleaned_df["engine_volume_cm3"] > 8_000),
        "engine_volume_cm3"
    ] = np.nan

    return cleaned_df.reset_index(drop=True)


if __name__ == "__main__":
    # Pronalaženje glavnog foldera projekta
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cars.csv"

    # Učitavanje i čišćenje podataka
    original_df = pd.read_csv(data_path)
    cleaned_df = clean_data(original_df)

    # Prikaz osnovnih rezultata
    print(f"Broj redova pre čišćenja: {len(original_df)}")
    print(f"Broj redova posle čišćenja: {len(cleaned_df)}")

    print("\nNedostajuće vrednosti posle čišćenja:")
    print(cleaned_df.isna().sum())