# Predikcija cene polovnih automobila

Projekat mašinskog učenja za predviđanje cene polovnog automobila na osnovu njegovih karakteristika.

**Autor:** Stefan Stojanović

## Opis projekta

Cilj projekta je razvoj regresionog modela koji predviđa vrednost kolone `priceUSD`.

Projekat obuhvata kompletan ML workflow:

1. istraživačku analizu podataka;
2. čišćenje podataka;
3. inženjering karakteristika;
4. pretprocesiranje;
5. treniranje i poređenje modela;
6. evaluaciju i izbor finalnog modela;
7. čuvanje istreniranog modela.

## Skup podataka

Projekat koristi skup `cars.csv`, koji sadrži 56.244 oglasa i 12 kolona sa podacima o polovnim automobilima.

Ciljna promenljiva je:

```text
priceUSD
```

Ulazni podaci uključuju marku, model, godište, kilometražu, stanje vozila, gorivo, zapreminu motora, boju, menjač, pogon i segment.

## Struktura projekta

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_car_price_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   ├── car_price_model.joblib
│   ├── model_comparison.csv
│   ├── evaluation_metrics.csv
│   ├── prediction_examples.csv
│   └── model_evaluation.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Čišćenje podataka

Tokom čišćenja:

- preimenovane su kolone sa složenim nazivima;
- standardizovane su tekstualne vrednosti;
- uklonjeni su potpuno duplirani redovi;
- uklonjeni su redovi sa nevalidnom cenom i godištem pre 1930;
- ekstremne kilometraže i zapremine motora označene su kao nedostajuće vrednosti.

## Nove karakteristike

Napravljene su sledeće karakteristike:

- `car_age` – starost automobila;
- `mileage_per_year` – prosečna kilometraža po godini;
- `engine_volume_liters` – zapremina motora u litrima;
- `is_newer_car` – indikator novijeg automobila.

## Pretprocesiranje

Za numeričke kolone korišćeni su:

- `SimpleImputer` sa medijanom;
- `StandardScaler`.

Za kategorijske kolone korišćeni su:

- `SimpleImputer` sa najčešćom vrednošću;
- `OneHotEncoder`.

Podaci su podeljeni na 80% trening i 20% test skupa uz `random_state=42`.

## Poređenje modela

Upoređena su tri regresiona modela:

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Random Forest | 1.113,76 | 2.657,19 | 0,8978 |
| Decision Tree | 1.297,10 | 2.959,45 | 0,8732 |
| Linear Regression | 1.975,92 | 4.271,56 | 0,7359 |

Random Forest je izabran kao finalni model jer ima najmanji MAE i RMSE, kao i najveći R².

Model u proseku odstupa od stvarne cene za približno 1.114 USD.

## Pokretanje projekta

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/stefanstojanovic994/car-price-prediction.git
cd car-price-prediction
```

### 2. Kreiranje virtuelnog okruženja

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux ili macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instaliranje biblioteka

```bash
pip install -r requirements.txt
```

### 4. Pokretanje Python skripti

Skripte se pokreću iz glavnog foldera projekta sledećim redosledom:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/data_preprocessing.py
python src/model_comparison.py
python src/model_training.py
python src/model_evaluation.py
```

### 5. Pokretanje Jupyter sveske

```bash
jupyter notebook notebooks/01_car_price_analysis.ipynb
```

## Finalni model

Finalni Random Forest pipeline sa pretprocesiranjem sačuvan je u:

```text
models/car_price_model.joblib
```

Rezultati evaluacije, primeri predviđanja i grafikon nalaze se u folderu `models`.

## Zaključak

Model uspešno koristi karakteristike polovnih automobila za okvirnu procenu njihove cene. Predviđena cena predstavlja procenu i može odstupati od stvarne tržišne vrednosti.