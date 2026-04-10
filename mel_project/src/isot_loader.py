"""
isot_loader.py
--------------
Chargement et preprocessing du ISOT Fake News Dataset pour l'évaluation
out-of-domain.

Le ISOT Dataset (University of Victoria, 2019) contient :
  - True.csv  : ~21 000 articles réels (source : Reuters)
  - Fake.csv  : ~23 000 articles faux (source : sites non fiables)

Colonnes : title, text, subject, date
Labels : déduits du fichier source (True → 'real', Fake → 'fake')

Référence :
  Ahmed H., Traore I., Saad S. (2017). Detection of Online Fake News Using
  N-Gram Analysis and Machine Learning Techniques.

Mapping vers nos 3 classes :
  ISOT est binaire (real / fake) → pas de classe 'nuanced'.
  On mappe directement : real → 2, fake → 0.
  La classe nuanced (1) sera absente — c'est attendu et documenté.
"""

import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text, LABEL_ENCODE


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Sujets politiques dans ISOT (colonne 'subject')
# On filtre sur ces sujets pour rester dans le domaine politique
POLITICAL_SUBJECTS = {
    "politicsNews", "politics", "Government News",
    "left-news", "US_News", "Middle-east", "worldnews",
}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_isot(
    data_dir: str = "data/external",
    political_only: bool = True,
    sample_n: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Charge et fusionne True.csv et Fake.csv du ISOT dataset.

    Args:
        data_dir       : dossier contenant True.csv et Fake.csv
        political_only : si True, filtre sur les sujets politiques uniquement
                         (recommandé pour rester comparable à LIAR)
        sample_n       : si fourni, échantillonne N exemples par classe
                         (utile pour équilibrer ou accélérer l'évaluation)
        random_state   : graine aléatoire pour la reproductibilité

    Returns:
        DataFrame avec colonnes normalisées pour l'évaluation
    """
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    if not os.path.exists(true_path):
        raise FileNotFoundError(
            f"True.csv introuvable dans {data_dir}.\n"
            "Télécharge le ISOT dataset depuis :\n"
            "https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php\n"
            "et place True.csv et Fake.csv dans data/external/"
        )
    if not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Fake.csv introuvable dans {data_dir}.\n"
            "Télécharge le ISOT dataset depuis :\n"
            "https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php\n"
            "et place True.csv et Fake.csv dans data/external/"
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label_3class"] = "real"
    df_fake["label_3class"] = "fake"

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Normalisation de la colonne texte
    # ISOT a 'title' + 'text' — on concatène pour avoir plus de contexte
    if "text" in df.columns:
        if "title" in df.columns:
            df["statement"] = (
                df["title"].fillna("") + " " + df["text"].fillna("")
            ).str.strip()
        else:
            df["statement"] = df["text"].fillna("")
    else:
        raise ValueError("Colonne 'text' introuvable dans le dataset ISOT.")

    # Filtrage politique (optionnel)
    if political_only and "subject" in df.columns:
        mask = df["subject"].isin(POLITICAL_SUBJECTS)
        before = len(df)
        df = df[mask].copy()
        print(f"Filtrage politique : {before} → {len(df)} articles "
              f"({len(df)/before*100:.1f}% conservés)")
    elif not political_only:
        print(f"Pas de filtrage par sujet — {len(df)} articles au total")

    # Suppression des lignes sans texte
    df = df.dropna(subset=["statement"])
    df = df[df["statement"].str.strip() != ""]

    # Échantillonnage équilibré (optionnel)
    if sample_n is not None:
        groups = []
        for label in df["label_3class"].unique():
            g = df[df["label_3class"] == label]
            n = min(sample_n, len(g))
            groups.append(g.sample(n, random_state=random_state))
        df = pd.concat(groups, ignore_index=True).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        print(f"Après échantillonnage : {len(df)} articles")

    # Encodage des labels (pas de nuanced dans ISOT)
    df["label_encoded"] = df["label_3class"].map(LABEL_ENCODE)
    df = df.dropna(subset=["label_encoded"])
    df["label_encoded"] = df["label_encoded"].astype(int)

    print(f"\nISOT chargé — {len(df)} articles")
    print("Distribution des labels :")
    counts = df["label_3class"].value_counts()
    for label, count in counts.items():
        print(f"  {label:<10} : {count:>6}  ({count/len(df)*100:.1f}%)")
    print()

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_isot(df: pd.DataFrame, max_chars: int = 1000) -> pd.DataFrame:
    """
    Nettoie le texte du dataset ISOT pour l'inférence.

    Note : on tronque à max_chars caractères avant le nettoyage car les
    articles ISOT sont bien plus longs que les déclarations LIAR (~18 mots).
    Sans troncature, le TF-IDF est dominé par des mots fréquents dans les
    articles longs (boilerplate, répétitions) qui n'existent pas dans LIAR.

    Args:
        df       : DataFrame ISOT (issu de load_isot)
        max_chars: longueur max du texte avant nettoyage

    Returns:
        df avec colonne 'statement_clean' ajoutée
    """
    df = df.copy()
    df["statement_clean"] = df["statement"].apply(
        lambda t: clean_text(str(t)[:max_chars])
    )
    empty = (df["statement_clean"].str.strip() == "").sum()
    if empty > 0:
        print(f"Attention : {empty} textes vides après nettoyage")
    print(f"Preprocessing ISOT terminé — {len(df)} articles, "
          f"longueur moy. : {df['statement_clean'].str.split().str.len().mean():.0f} mots")
    return df


# ---------------------------------------------------------------------------
# Métadonnées manquantes
# ---------------------------------------------------------------------------

def add_dummy_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des colonnes de métadonnées vides (valeur 0) pour l'évaluation
    des modèles qui attendent ces colonnes (RF, XGBoost, LR avec CombinedFeatures).

    ⚠ Biais documenté : credibility_score et lie_rate seront à 0 pour tous
    les exemples ISOT car l'historique PolitiFact n'existe pas dans ce dataset.
    C'est précisément ce qu'on veut mesurer : à quel point le modèle dépend
    de ces features spécifiques à LIAR ?

    Args:
        df : DataFrame ISOT préprocessé

    Returns:
        df avec les colonnes de métadonnées ajoutées à 0
    """
    meta_cols = [
        "credibility_score", "lie_rate", "history_total", "is_politician",
    ]
    for col in meta_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ---------------------------------------------------------------------------
# Résumé du domain shift
# ---------------------------------------------------------------------------

def print_domain_shift_summary(liar_df: pd.DataFrame, isot_df: pd.DataFrame) -> None:
    """
    Affiche un résumé des différences entre LIAR et ISOT pour documenter
    le domain shift attendu.

    Args:
        liar_df : DataFrame LIAR test (préprocessé)
        isot_df : DataFrame ISOT (préprocessé)
    """
    print("=" * 55)
    print("  ANALYSE DU DOMAIN SHIFT : LIAR vs ISOT")
    print("=" * 55)

    liar_len = liar_df["statement_clean"].str.split().str.len()
    isot_len = isot_df["statement_clean"].str.split().str.len()

    print(f"\n{'Caractéristique':<28} {'LIAR':>8} {'ISOT':>8}")
    print("-" * 46)
    print(f"  {'Taille (n exemples)':<26} {len(liar_df):>8} {len(isot_df):>8}")
    print(f"  {'Longueur moy. (mots)':<26} {liar_len.mean():>8.1f} {isot_len.mean():>8.1f}")
    print(f"  {'Longueur max (mots)':<26} {liar_len.max():>8} {isot_len.max():>8}")
    print(f"  {'Source':<26} {'PolitiFact':>8} {'Reuters/':>8}")
    print(f"  {'':<26} {'':>8} {'sites':>8}")
    print(f"  {'Type de texte':<26} {'déclaration':>8} {'article':>8}")
    print(f"  {'Métadonnées speaker':<26} {'oui':>8} {'non':>8}")

    print(f"\n  Distribution des classes :")
    liar_dist = liar_df["label_3class"].value_counts(normalize=True)
    isot_dist = isot_df["label_3class"].value_counts(normalize=True)
    for label in ["fake", "nuanced", "real"]:
        l = liar_dist.get(label, 0)
        i = isot_dist.get(label, 0)
        print(f"    {label:<12} : LIAR {l*100:>5.1f}%   ISOT {i*100:>5.1f}%")
    print()
