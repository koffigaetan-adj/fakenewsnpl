"""
welfake_loader.py
-----------------
Chargement et preprocessing du WELFake Dataset pour l'évaluation
out-of-domain.

WELFake (Verma et al., 2021) est un dataset de 72 134 articles de presse,
fusionnant quatre sources (Kaggle, McIntire, Reuters, BuzzFeed).

Colonnes : Unnamed: 0 (index), title, text, label
Labels   : 0 = fake, 1 = real  (binaire — pas de classe 'nuanced')

Mapping vers nos 3 classes :
  0 (fake) → label_encoded = 0
  1 (real) → label_encoded = 2
  La classe 'nuanced' (1) sera absente — c'est attendu et documenté.

Référence :
  Verma P.K., Agrawal P., Amorim I., Prodan R. (2021).
  WELFake: Word Embedding over Linguistic Features for Fake News Detection.
  IEEE Transactions on Computational Social Science.
"""

import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text


# Mapping WELFake → nos 3 classes
WELFAKE_LABEL_MAP = {
    0: "fake",
    1: "real",
}

LABEL_ENCODE = {
    "fake"    : 0,
    "nuanced" : 1,
    "real"    : 2,
}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_welfake(
    path: str = "data/external/WELFake_Dataset.csv",
    sample_n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Charge le WELFake Dataset et le normalise pour l'évaluation.

    Args:
        path         : chemin vers WELFake_Dataset.csv
        sample_n     : nombre d'exemples PAR CLASSE à échantillonner.
                       None = tout le dataset (72k articles, évaluation longue).
                       Recommandé : 5000 par classe pour un bon équilibre vitesse/représentativité.
        random_state : graine aléatoire pour la reproductibilité

    Returns:
        DataFrame normalisé avec colonnes :
        'statement', 'label_3class', 'label_encoded'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"WELFake_Dataset.csv introuvable à : {path}\n"
            "Place le fichier dans data/external/"
        )

    df = pd.read_csv(path)

    # Normalisation des colonnes
    # On concatène title + text pour maximiser le signal textuel
    title = df["title"].fillna("")
    text  = df["text"].fillna("")
    df["statement"] = (title + " " + text).str.strip()

    # Suppression des lignes sans contenu
    df = df[df["statement"].str.strip() != ""].copy()
    df = df.dropna(subset=["label"])

    # Mapping des labels
    df["label_3class"]  = df["label"].map(WELFAKE_LABEL_MAP)
    df = df.dropna(subset=["label_3class"])
    df["label_encoded"] = df["label_3class"].map(LABEL_ENCODE).astype(int)

    # Échantillonnage équilibré par classe
    if sample_n is not None:
        groups = []
        for label in df["label_3class"].unique():
            g = df[df["label_3class"] == label]
            n = min(sample_n, len(g))
            groups.append(g.sample(n, random_state=random_state))
        df = (
            pd.concat(groups, ignore_index=True)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    print(f"WELFake chargé — {len(df)} articles")
    print("Distribution des labels :")
    counts = df["label_3class"].value_counts()
    for label, count in counts.items():
        print(f"  {label:<10} : {count:>6}  ({count/len(df)*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_welfake(df: pd.DataFrame, max_chars: int = 500) -> pd.DataFrame:
    """
    Nettoie le texte WELFake pour l'inférence avec les modèles LIAR.

    Pourquoi tronquer à max_chars ?
    Les articles WELFake font en moyenne 540 mots (~2700 caractères), alors
    que les déclarations LIAR font ~18 mots (~100 caractères). Sans troncature :
    - Le TF-IDF est noyé par du vocabulaire absent de LIAR
    - BERT dépasse son max_length de 128 tokens
    On tronque donc à ~500 caractères pour se rapprocher du domaine LIAR.

    Args:
        df       : DataFrame WELFake (issu de load_welfake)
        max_chars: longueur max du texte brut avant nettoyage

    Returns:
        df avec colonne 'statement_clean' ajoutée
    """
    df = df.copy()
    df["statement_clean"] = df["statement"].apply(
        lambda t: clean_text(str(t)[:max_chars])
    )
    empty = (df["statement_clean"].str.strip() == "").sum()
    if empty > 0:
        print(f"  {empty} textes vides après nettoyage")

    mean_len = df["statement_clean"].str.split().str.len().mean()
    print(f"Preprocessing terminé — longueur moy. après nettoyage : {mean_len:.0f} mots")
    return df


# ---------------------------------------------------------------------------
# Métadonnées manquantes
# ---------------------------------------------------------------------------

def add_dummy_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes de métadonnées LIAR manquantes avec valeur 0.

    ⚠ Biais documenté : credibility_score et lie_rate sont à 0 pour tous
    les articles WELFake. Les modèles classiques entraînés avec ces features
    vont donc se retrouver privés de leur signal le plus discriminant.
    C'est exactement ce que l'évaluation out-of-domain cherche à mesurer.

    Args:
        df : DataFrame WELFake préprocessé

    Returns:
        df avec colonnes de métadonnées à 0
    """
    for col in ["credibility_score", "lie_rate", "history_total", "is_politician"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ---------------------------------------------------------------------------
# Analyse du domain shift
# ---------------------------------------------------------------------------

def print_domain_shift_summary(liar_df: pd.DataFrame, welfake_df: pd.DataFrame) -> None:
    """
    Affiche un résumé des différences structurelles entre LIAR et WELFake
    pour documenter le domain shift attendu.

    Args:
        liar_df    : DataFrame LIAR test (préprocessé)
        welfake_df : DataFrame WELFake (préprocessé)
    """
    print("=" * 58)
    print("  ANALYSE DU DOMAIN SHIFT : LIAR vs WELFake")
    print("=" * 58)

    liar_len    = liar_df["statement_clean"].str.split().str.len()
    welfake_len = welfake_df["statement_clean"].str.split().str.len()

    print(f"\n{'Caractéristique':<28} {'LIAR':>10} {'WELFake':>10}")
    print("-" * 50)
    print(f"  {'Taille (n exemples)':<26} {len(liar_df):>10} {len(welfake_df):>10}")
    print(f"  {'Longueur moy. (mots)':<26} {liar_len.mean():>10.1f} {welfake_len.mean():>10.1f}")
    print(f"  {'Longueur max (mots)':<26} {liar_len.max():>10} {welfake_len.max():>10}")
    print(f"  {'Type de texte':<26} {'déclaration':>10} {'article':>10}")
    print(f"  {'Métadonnées speaker':<26} {'oui':>10} {'non':>10}")
    print(f"  {'Nb de classes':<26} {'3':>10} {'2':>10}")

    print(f"\n  Distribution des classes :")
    liar_dist    = liar_df["label_3class"].value_counts(normalize=True)
    welfake_dist = welfake_df["label_3class"].value_counts(normalize=True)
    for label in ["fake", "nuanced", "real"]:
        l = liar_dist.get(label, 0)
        w = welfake_dist.get(label, 0)
        print(f"    {label:<12} : LIAR {l*100:>5.1f}%   WELFake {w*100:>5.1f}%")
    print()
