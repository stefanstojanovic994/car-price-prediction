# ============================================================
# INŽENJERING KARAKTERISTIKA
# ============================================================

from pathlib import Path

import pandas as pd

from data_cleaning import clean_data


# Dataset sadrži automobile zaključno sa 2019. godinom
REFERENCE_YEAR = 2020


def add_features(df):
    """
    Dodaje nove karakteristike korisne za predviđanje cene.
    """

    featured_df = df.copy()

    # Starost automobila
    featured_df["car_age"] = (
        REFERENCE_YEAR - featured_df["year"]
    )

    # Prosečna kilometraža po godini starosti
    featured_df["mileage_per_year"] = (
        featured_df["mileage_km"] /
        featured_df["car_age"].clip(lower=1)
    )

    # Zapremina motora izražena u litrima
    featured_df["engine_volume_liters"] = (
        featured_df["engine_volume_cm3"] / 1000
    )

    # Indikator novijeg automobila
    featured_df["is_newer_car"] = (
        featured_df["year"] >= 2010
    ).astype(int)

    return featured_df


if __name__ == "__main__":
    # Putanja do originalnog skupa podataka
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cars.csv"

    # Učitavanje, čišćenje i dodavanje novih karakteristika
    original_df = pd.read_csv(data_path)
    cleaned_df = clean_data(original_df)
    featured_df = add_features(cleaned_df)

    new_features = [
        "car_age",
        "mileage_per_year",
        "engine_volume_liters",
        "is_newer_car"
    ]

    print("Uspešno su napravljene nove karakteristike:")
    print(new_features)

    print("\nPrimer novih karakteristika:")
    print(featured_df[new_features].head())