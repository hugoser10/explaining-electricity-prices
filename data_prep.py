import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed/")

# Import data
def load_raw_data():
    x_train = pd.read_csv(DATA_DIR + "X_train.csv")
    y_train = pd.read_csv(DATA_DIR + "y_train.csv")
    x_test  = pd.read_csv(DATA_DIR + "X_test.csv")

    # Join X and y train sets
    train = x_train.merge(y_train, on='ID')

    print(f"X_train : {x_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_test  : {x_test.shape}")

    return train, x_test

# Fill nans
def fill_na(df, x_test):
    cols_to_fill = [col for col in df.columns if df[col].isnull().sum() > 0]
    medians = df[cols_to_fill].median()

    df_filled     = df.copy()
    x_test_filled = x_test.copy()

    df_filled[cols_to_fill]     = df[cols_to_fill].fillna(medians)
    x_test_filled[cols_to_fill] = x_test[cols_to_fill].fillna(medians)

    print(f"Valeurs manquantes train : {df_filled.isnull().sum().sum()}")
    print(f"Valeurs manquantes test  : {x_test_filled.isnull().sum().sum()}")

    return df_filled, x_test_filled

# Run data import and preparation
def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load data
    train, x_test = load_raw_data()

    # Fill nans
    train_filled, x_test_filled = fill_na(train, x_test)

    print(f"Train : {train_filled.shape}")
    print(f"Test : {x_test_filled.shape}")

    # Export to CSV
    train_filled.to_csv(PROCESSED_DIR + "train_pr.csv", index=False)
    x_test_filled.to_csv(PROCESSED_DIR + "x_test_pr.csv", index=False)

    print(f"\nProcessed data has been exported to {PROCESSED_DIR}")

if __name__ == "__main__":
    run()
