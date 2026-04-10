import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

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
    cols_to_fill = df.select_dtypes(include='number').columns.tolist()
    cols_to_fill = [col for col in cols_to_fill if col != 'TARGET']

    imputer = SimpleImputer(strategy='median')

    # Fit sur le train
    df_train_filled = df.copy()
    df_train_filled[cols_to_fill] = imputer.fit_transform(df[cols_to_fill])

    # Transform sur train et test
    x_test_filled = x_test.copy()
    x_test_filled[cols_to_fill] = imputer.transform(x_test_filled[cols_to_fill])

    assert df_train_filled.isnull().sum().sum() == 0, "NaN restants dans le train"
    assert x_test_filled.isnull().sum().sum() == 0, "NaN restants dans le test"

    return df_train_filled, x_test_filled

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
