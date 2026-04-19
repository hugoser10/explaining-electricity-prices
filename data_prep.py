import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import rankdata
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT_DIR      = os.path.abspath(os.path.dirname(__file__))
DATA_DIR      = os.path.join(ROOT_DIR, "data/")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed/")

# Constantes économiques
GAS_EFFICIENCY       = 0.5
GAS_EMISSION_FACTOR  = 0.4
COAL_EFFICIENCY      = 0.5
COAL_EMISSION_FACTOR = 1
id_cols = ['ID', 'COUNTRY', 'DAY_ID']

def load_raw_data():
    '''
    Charge les fichiers bruts X_train, y_train, X_test depuis DATA_DIR.
    Merge X_train et y_train sur la colonne ID.

    Retourne
    --------
    train  : DataFrame  X_train + y_train mergés sur ID.
    x_test : DataFrame  Données de test brutes.
    '''
    x_train = pd.read_csv(DATA_DIR + "X_train.csv")
    y_train = pd.read_csv(DATA_DIR + "y_train.csv")
    x_test  = pd.read_csv(DATA_DIR + "X_test.csv")

    train = x_train.merge(y_train, on='ID')

    print(f"X_train : {x_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_test  : {x_test.shape}")

    return train, x_test

def drop_colinear_features(df, list_to_drop=None, threshold=1):
    '''
    Supprime les features colinéaires au-delà d'un seuil de corrélation.

    Paramètres
    ----------
    df        : DataFrame  DataFrame avec features numériques.
    list_to_drop : list   Liste des colonnes à supprimer. Si None, calcul automatique basé sur la matrice de corrélation.
    threshold : float      Seuil de corrélation pour supprimer une feature.

    Retourne
    --------
    df_reduced : DataFrame  DataFrame avec features colinéaires supprimées.
    '''
    if list_to_drop is None:
        corr_matrix = df[[c for c in df.columns if c not in id_cols]].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] == threshold)]
    else:
        to_drop = list_to_drop
    print(f"Features colinéaires à supprimer (corr = {threshold}): {to_drop}")
    
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced, to_drop


def build_eco_features(train, x_test):
    '''
    Construit les features de coût marginal AVANT standardisation.

    Les features économiques sont construites sur les valeurs brutes
    car les coefficients ont un sens physique (efficacité énergétique,
    facteur d'émission CO2). Les construire après standardisation
    rendrait les coefficients arbitraires.

    MARGINAL_GAS  = GAS_EFFICIENCY  * GAS_RET  + GAS_EMISSION_FACTOR  * CARBON_RET
    MARGINAL_COAL = COAL_EFFICIENCY * COAL_RET + COAL_EMISSION_FACTOR * CARBON_RET

    Supprime ensuite GAS_RET, COAL_RET, CARBON_RET pour éviter
    la redondance avec les features construites.

    Paramètres
    ----------
    train  : DataFrame  Données d'entraînement brutes.
    x_test : DataFrame  Données de test brutes.

    Retourne
    --------
    train_eco  : DataFrame  Train avec MARGINAL_GAS et MARGINAL_COAL.
    x_test_eco : DataFrame  Test  avec MARGINAL_GAS et MARGINAL_COAL.
    '''
    train_eco  = train.copy()
    x_test_eco = x_test.copy()

    for df in [train_eco, x_test_eco]:
        df['MARGINAL_GAS']  = (GAS_EFFICIENCY  * df['GAS_RET']
                               + GAS_EMISSION_FACTOR  * df['CARBON_RET'])
        df['MARGINAL_COAL'] = (COAL_EFFICIENCY * df['COAL_RET']
                               + COAL_EMISSION_FACTOR * df['CARBON_RET'])
        df.drop(columns=['GAS_RET', 'COAL_RET', 'CARBON_RET'], inplace=True)

    print(f"Features éco construites : MARGINAL_GAS, MARGINAL_COAL")
    print(f"GAS_RET, COAL_RET, CARBON_RET supprimées")

    return train_eco, x_test_eco


def fill_na(train, x_test):
    '''
    Remplace les valeurs manquantes par la médiane calculée sur le train.

    Les colonnes préfixées DE_ et FR_ représentent déjà les données
    de chaque pays — chaque ligne du dataset contient les features
    des deux pays simultanément. La médiane globale est donc adaptée.

    Le x_test est imputé avec les médianes du train (pas du test)
    pour éviter tout data leakage.

    Paramètres
    ----------
    train  : DataFrame  Données d'entraînement.
    x_test : DataFrame  Données de test.

    Retourne
    --------
    train_filled  : DataFrame  Train sans valeurs manquantes.
    x_test_filled : DataFrame  Test sans valeurs manquantes.
    '''
    train_filled  = train.copy()
    x_test_filled = x_test.copy()

    cols_num = [col for col in train.columns
                if col not in ('ID', 'COUNTRY', 'TARGET', 'DAY_ID')
                and train[col].dtype in (np.float64, np.int64)]

    # Médiane calculée sur le train uniquement
    medians = train_filled[cols_num].median()

    train_filled[cols_num]  = train_filled[cols_num].fillna(medians)
    x_test_filled[cols_num] = x_test_filled[cols_num].fillna(medians)

    print(f"Valeurs manquantes train : {train_filled.isnull().sum().sum()}")
    print(f"Valeurs manquantes test  : {x_test_filled.isnull().sum().sum()}")

    return train_filled, x_test_filled


def standardize(train, x_test):
    '''
    Standardise les features numériques (moyenne=0, std=1).

    Les colonnes préfixées DE_ et FR_ représentent déjà les données
    de chaque pays — chaque ligne du dataset contient les features
    des deux pays simultanément. La standardisation globale est adaptée.

    La moyenne et l'écart-type sont calculés sur le train uniquement
    et appliqués au train ET au test pour éviter tout data leakage.
    Les colonnes ID, COUNTRY, TARGET et DAY_ID sont exclues.

    Paramètres
    ----------
    train  : DataFrame  Données d'entraînement après imputation.
    x_test : DataFrame  Données de test après imputation.

    Retourne
    --------
    train_std  : DataFrame  Train standardisé.
    x_test_std : DataFrame  Test standardisé (params du train).
    '''
    train_std  = train.copy()
    x_test_std = x_test.copy()

    cols_num = [col for col in train.columns
                if col not in ('ID', 'COUNTRY', 'TARGET', 'DAY_ID')
                and train[col].dtype in (np.float64, np.int64)]

    # Moyenne et std calculés uniquement sur le train
    means = train_std[cols_num].mean()
    stds  = train_std[cols_num].std().replace(0, 1)

    train_std[cols_num]  = (train_std[cols_num]  - means) / stds
    x_test_std[cols_num] = (x_test_std[cols_num] - means) / stds

    print(f"  {len(cols_num)} features standardisées "
          f"(train : {len(train_std)} obs, test : {len(x_test_std)} obs)")

    return train_std, x_test_std


def build_y_train(train):
    '''
    Construit y_train_pr avec deux colonnes :
      - values : TARGET brute (variation de prix)
      - rank   : rang normalisé de TARGET dans [0, 1], calculé PAR PAYS
                 sur le train uniquement — aligné avec la métrique Spearman

    Le rang est calculé PAR PAYS car FR et DE ont des distributions
    de TARGET différentes. Un rang global mélangrait les deux régimes.
    Le rang est normalisé par le nombre d'observations du pays pour
    être comparable entre FR et DE.

    Paramètres
    ----------
    train : DataFrame  Train avec colonnes TARGET et COUNTRY, index = ID.

    Retourne
    --------
    y_train : DataFrame  Index = ID, colonnes = ['values', 'rank'].
    '''
    y_parts = []

    for country in train['COUNTRY'].unique():
        mask   = train['COUNTRY'] == country
        y_vals = train.loc[mask, 'TARGET']

        # Rang normalisé [0, 1] calculé sur ce pays uniquement
        y_rnk = pd.Series(
            rankdata(y_vals) / len(y_vals),
            index=y_vals.index
        )
        y_parts.append(pd.DataFrame({'values': y_vals, 'rank': y_rnk}))

    y_train = pd.concat(y_parts).sort_index()
    y_train.index.name = 'ID'

    return y_train


def run():
    '''
    Pipeline principal de préparation des données.

    Produit 3 fichiers CSV dans PROCESSED_DIR :
      - x_train_pr.csv : features d'entraînement
      - x_test_pr.csv  : features de test
      - y_train_pr.csv : target avec colonnes 'values' et 'rank'

    Étapes :
      1. Chargement des données brutes
      2. Construction des features économiques (avant standardisation)
      3. Imputation par médiane globale (params du train)
      4. Standardisation globale       (params du train)
      5. Indexation par ID
      6. Construction de y_train_pr    (values + rank par pays)
      7. Export CSV
    '''
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Chargement
    train, x_test = load_raw_data()

    # 2 Suppression des colonnes colinéaires
    train, to_drop = drop_colinear_features(train, list_to_drop=['FR_NET_IMPORT', 'DE_NET_IMPORT','FR_DE_EXCHANGE'], threshold=1)
    x_test = x_test.drop(columns=to_drop)

    # 3. Features économiques (avant standardisation)
    print("\n── Construction des features économiques ──")
    train_eco, x_test_eco = build_eco_features(train, x_test)

    # 4. Imputation par médiane globale
    print("\n── Imputation (médiane globale, params du train) ──")
    train_filled, x_test_filled = fill_na(train_eco, x_test_eco)

    # 5. Standardisation globale
    print("\n── Standardisation globale (params du train) ──")
    train_std, x_test_std = standardize(train_filled, x_test_filled)

    # 6. Indexation par ID
    train_std.set_index('ID', inplace=True)
    x_test_std.set_index('ID', inplace=True)

    # 7. Construction de y_train_pr
    print("\n── Construction de y_train_pr ──")
    y_train_pr = build_y_train(train_std)
    print(f"y_train_pr : {y_train_pr.shape} | colonnes : {y_train_pr.columns.tolist()}")

    # x_train_pr : features uniquement (sans TARGET, COUNTRY, DAY_ID)
    cols_drop_test  = [c for c in id_cols  if c in x_test_std.columns]
    cols_drop_train = [c for c in id_cols + ['TARGET'] if c in train_std.columns]

    x_train_pr = train_std.drop(columns=cols_drop_train)
    x_test_pr  = x_test_std.drop(columns=cols_drop_test)

    print(f"x_train_pr : {x_train_pr.shape}")
    print(f"x_test_pr  : {x_test_pr.shape}")

    # 7. Export CSV
    x_train_pr.to_csv(PROCESSED_DIR + "x_train_pr.csv", index=True)
    x_test_pr.to_csv(PROCESSED_DIR  + "x_test_pr.csv",  index=True)
    y_train_pr.to_csv(PROCESSED_DIR + "y_train_pr.csv", index=True)

    print(f"\n Fichiers exportés dans {PROCESSED_DIR}")
    print(f"   x_train_pr.csv | x_test_pr.csv | y_train_pr.csv")

    return x_train_pr, x_test_pr, y_train_pr


if __name__ == "__main__":
    run()
