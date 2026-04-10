"""
preprocessing.py
----------------
Nettoyage du texte, regroupement des labels (6 -> 3 classes),
gestion des valeurs manquantes et calcul des features de métadonnées.

Utilisé par : notebooks/02_models.ipynb, notebooks/03_generalization.ipynb
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

COLONNES = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Regroupement des 6 labels LIAR en 3 classes
# Choix méthodologique : on conserve une classe "nuancée" pour refléter
# le fait que la véracité est un spectre, pas un binaire.
LABEL_MAP = {
    "pants-fire"  : "fake",
    "false"       : "fake",
    "barely-true" : "nuanced",
    "half-true"   : "nuanced",
    "mostly-true" : "real",
    "true"        : "real",
}

LABEL_ENCODE = {
    "fake"    : 0,
    "nuanced" : 1,
    "real"    : 2,
}

HISTORY_COLS = [
    "pants_on_fire_counts",
    "false_counts",
    "barely_true_counts",
    "half_true_counts",
    "mostly_true_counts",
]

# Stopwords anglais minimaliste (on garde les négations car discriminantes)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "it", "its",
    "they", "them", "their", "he", "she", "we", "our", "you", "your", "my",
    "his", "her", "i", "s", "says", "said",
}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_liar(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les 3 splits du LIAR Dataset (train, valid, test).

    Args:
        data_dir : chemin vers le dossier contenant train.tsv, valid.tsv, test.tsv

    Returns:
        tuple (train, valid, test) de DataFrames pandas
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        path = f"{data_dir}/{split}.tsv"
        splits[split] = pd.read_csv(
            path, sep="\t", header=None, names=COLONNES
        )
    print(f"LIAR chargé — train: {len(splits['train'])}, "
          f"valid: {len(splits['valid'])}, test: {len(splits['test'])}")
    return splits["train"], splits["valid"], splits["test"]


def load_buzzfeed(path: str = "data/external/buzzfeed.csv") -> pd.DataFrame:
    """
    Charge le BuzzFeed Political Dataset.
    Attendu : colonnes 'text' (ou 'content') et 'label' (fake/real).

    Args:
        path : chemin vers le fichier CSV BuzzFeed

    Returns:
        DataFrame avec colonnes normalisées ['statement', 'label_3class', 'label_encoded']
    """
    df = pd.read_csv(path)

    # Normalisation du nom de colonne texte selon la version du dataset
    if "content" in df.columns and "statement" not in df.columns:
        df = df.rename(columns={"content": "statement"})
    if "text" in df.columns and "statement" not in df.columns:
        df = df.rename(columns={"text": "statement"})

    # Mapping des labels BuzzFeed -> nos 3 classes
    # mostly true / mixture of true and false -> nuanced
    # mostly false / no factual content -> fake
    buzzfeed_map = {
        "mostly true"              : "real",
        "mixture of true and false": "nuanced",
        "mostly false"             : "fake",
        "no factual content"       : "fake",
    }
    # Labels binaires simples
    binary_map = {
        "real": "real",
        "fake": "fake",
        "true": "real",
        "false": "fake",
        "1": "real",
        "0": "fake",
    }

    raw_labels = df["label"].str.lower().str.strip()
    df["label_3class"] = raw_labels.map(buzzfeed_map).fillna(
        raw_labels.map(binary_map)
    )

    # Supprimer les lignes dont le label n'a pas pu être mappé
    before = len(df)
    df = df.dropna(subset=["label_3class", "statement"])
    after = len(df)
    if before != after:
        print(f"BuzzFeed : {before - after} lignes supprimées (label ou texte manquant)")

    df["label_encoded"] = df["label_3class"].map(LABEL_ENCODE)
    print(f"BuzzFeed chargé — {len(df)} lignes")
    print(df["label_3class"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Nettoyage du texte
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Nettoyage standard d'une déclaration politique :
    - lowercase
    - suppression de la ponctuation et des chiffres isolés
    - suppression des stopwords
    - normalisation des espaces

    On NE fait PAS de stemming/lemmatisation pour préserver
    le sens des négations (important pour la détection de fake news).

    Args:
        text : chaîne brute

    Returns:
        texte nettoyé
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)          # URLs
    text = re.sub(r"[^a-z\s']", " ", text)            # ponctuation (garde apostrophe)
    text = re.sub(r"\b\d+\b", "", text)               # chiffres isolés
    text = re.sub(r"\s+", " ", text).strip()          # espaces multiples

    # Suppression des stopwords mot par mot
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Regroupement des labels
# ---------------------------------------------------------------------------

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute deux colonnes au DataFrame LIAR :
    - label_3class : 'fake' / 'nuanced' / 'real'
    - label_encoded : 0 / 1 / 2

    Args:
        df : DataFrame LIAR avec colonne 'label' originale

    Returns:
        df enrichi (modification en place + retour)
    """
    df = df.copy()
    df["label_3class"] = df["label"].map(LABEL_MAP)
    df["label_encoded"] = df["label_3class"].map(LABEL_ENCODE)

    # Vérification : aucun label non mappé
    unmapped = df["label_3class"].isna().sum()
    if unmapped > 0:
        print(f"Attention : {unmapped} labels non mappés (labels inconnus)")
    return df


# ---------------------------------------------------------------------------
# Gestion des valeurs manquantes
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit les valeurs manquantes sur les colonnes utiles.
    On ne supprime aucune ligne pour ne pas perdre de données.

    Stratégie :
    - Texte vide -> chaîne vide (sera ignorée par le vectoriseur)
    - Speaker / party inconnus -> 'unknown'
    - Counts manquants -> 0 (le speaker n'a pas d'historique)

    Args:
        df : DataFrame LIAR brut

    Returns:
        df nettoyé
    """
    df = df.copy()
    df["statement"] = df["statement"].fillna("")
    df["context"]   = df["context"].fillna("")
    df["subject"]   = df["subject"].fillna("")
    df["speaker"]   = df["speaker"].fillna("unknown")
    df["party"]     = df["party"].fillna("unknown")
    df["job_title"] = df["job_title"].fillna("unknown")

    for col in HISTORY_COLS:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Features de métadonnées
# ---------------------------------------------------------------------------

def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des features numériques à partir des métadonnées.
    Ces features viennent en complément du texte dans les modèles classiques.

    Features créées :
    - credibility_score : ratio (mostly_true + half_true) / total_history
      Mesure la fiabilité historique du speaker sur PolitiFact.
      ⚠ Biais : juge le speaker autant que la déclaration.

    - lie_rate : ratio (false + pants_on_fire) / total_history
      Proportion de mensonges avérés dans l'historique.

    - history_total : nombre total de déclarations vérifiées dans l'historique.
      Un historique court est moins fiable statistiquement.

    - is_politician : 1 si le party est republican ou democrat, 0 sinon.

    Args:
        df : DataFrame avec colonnes HISTORY_COLS et 'party'

    Returns:
        df enrichi avec les nouvelles colonnes
    """
    df = df.copy()

    total = df[HISTORY_COLS].sum(axis=1).replace(0, 1)  # évite /0

    df["credibility_score"] = (
        (df["mostly_true_counts"] + df["half_true_counts"]) / total
    ).round(4)

    df["lie_rate"] = (
        (df["false_counts"] + df["pants_on_fire_counts"]) / total
    ).round(4)

    df["history_total"] = df[HISTORY_COLS].sum(axis=1)

    df["is_politician"] = df["party"].isin(
        ["republican", "democrat"]
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Features étendues
# ---------------------------------------------------------------------------

# Top sujets LIAR (par fréquence dans le train set)
TOP_SUBJECTS = [
    "health-care", "taxes", "immigration", "elections", "education",
    "candidates-biography", "economy", "guns", "federal-budget", "jobs",
]

# Mots à langage absolu — indicateurs de claims extrêmes (souvent fake/real)
ABSOLUTE_WORDS = {
    "never", "always", "all", "every", "none", "everyone", "nobody",
    "entirely", "completely", "absolutely", "totally", "only", "best",
    "worst", "biggest", "largest", "smallest", "first", "last", "most",
    "least", "entire", "whole", "zero", "100",
}

# Mots à langage de couverture — indicateurs de claims nuancées
HEDGING_WORDS = {
    "approximately", "about", "roughly", "around", "nearly", "almost",
    "reportedly", "possibly", "perhaps", "maybe", "could", "might",
    "seems", "appear", "suggests", "according", "estimated", "roughly",
    "somewhat", "partly", "partially",
}


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features dérivées du texte de la déclaration.

    Features créées :
    - stmt_word_count : longueur en mots (les claims courtes tendent vers les extrêmes)
    - has_absolute_lang : 1 si la déclaration contient un mot absolu (never, always, all…)
      Signal souvent associé aux claims fake ou real tranchées.
    - has_hedging : 1 si la déclaration contient un mot de nuance (approximately, maybe…)
      Signal souvent associé aux claims nuanced.
    - subject_<topic> : indicateurs binaires pour les 10 sujets les plus fréquents.
      Le sujet abordé a un effet fort sur la véracité attendue.

    Args:
        df : DataFrame avec colonnes 'statement' et 'subject'

    Returns:
        df enrichi
    """
    df = df.copy()
    stmts = df["statement"].fillna("").str.lower()

    # Longueur
    df["stmt_word_count"] = stmts.str.split().str.len().fillna(0).astype(int)

    # Langage absolu
    df["has_absolute_lang"] = stmts.apply(
        lambda t: int(bool(set(t.split()) & ABSOLUTE_WORDS))
    )

    # Langage de couverture
    df["has_hedging"] = stmts.apply(
        lambda t: int(bool(set(t.split()) & HEDGING_WORDS))
    )

    # Sujets (multi-label → indicateurs binaires)
    subjects = df["subject"].fillna("").str.lower()
    for subj in TOP_SUBJECTS:
        col_name = "subj_" + subj.replace("-", "_")
        df[col_name] = subjects.str.contains(subj, regex=False).astype(int)

    return df


def add_party_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode le parti politique du speaker en indicateurs binaires.
    Plus informatif que le simple is_politician : 'none' et 'organization'
    ont des profils de véracité distincts de republican/democrat.

    Features créées :
    - party_republican, party_democrat, party_none, party_org

    Args:
        df : DataFrame avec colonne 'party'

    Returns:
        df enrichi
    """
    df = df.copy()
    party = df["party"].fillna("unknown").str.lower()
    df["party_republican"] = (party == "republican").astype(int)
    df["party_democrat"]   = (party == "democrat").astype(int)
    df["party_none"]       = (party == "none").astype(int)
    df["party_org"]        = (party == "organization").astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def preprocess_liar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet pour un split LIAR :
    1. Gestion des valeurs manquantes
    2. Nettoyage du texte -> colonne 'statement_clean'
    3. Regroupement des labels -> 'label_3class', 'label_encoded'
    4. Calcul des features de métadonnées
    5. Features textuelles étendues (longueur, langage, sujets)
    6. Features de parti politique

    Args:
        df : DataFrame LIAR brut (train, valid ou test)

    Returns:
        df prêt pour la modélisation
    """
    df = handle_missing(df)
    df["statement_clean"] = df["statement"].apply(clean_text)
    df = map_labels(df)
    df = add_metadata_features(df)
    df = add_text_features(df)
    df = add_party_features(df)
    return df


def preprocess_buzzfeed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de nettoyage pour le dataset BuzzFeed.
    Plus simple : pas de métadonnées, juste le texte.

    Args:
        df : DataFrame BuzzFeed (issu de load_buzzfeed)

    Returns:
        df avec colonne 'statement_clean' ajoutée
    """
    df = df.copy()
    df["statement_clean"] = df["statement"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# Résumé après preprocessing
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, nom: str = "dataset") -> None:
    """
    Affiche un résumé de la distribution des labels après preprocessing.

    Args:
        df  : DataFrame préprocessé (doit avoir 'label_3class')
        nom : nom du split pour l'affichage
    """
    print(f"\n=== Résumé — {nom} ({len(df)} lignes) ===")
    counts = df["label_3class"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {label:<10} : {count:>5}  ({pct:.1f}%)")
    print(f"   Texte vide : {(df['statement_clean'] == '').sum()} lignes")
