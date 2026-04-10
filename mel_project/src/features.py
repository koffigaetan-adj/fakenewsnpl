"""
features.py
-----------
Construction des représentations vectorielles du texte et combinaison
avec les features de métadonnées.

Approches implémentées :
  1. TF-IDF  (baseline rapide, interprétable)
  2. Word2Vec / GloVe (embeddings statiques, moyenne sur les tokens)
  3. Combinaison TF-IDF + métadonnées (meilleur compromis vitesse/perf)

Utilisé par : notebooks/02_models.ipynb, notebooks/03_generalization.ipynb
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ---------------------------------------------------------------------------
# Colonnes de métadonnées numériques à inclure dans les modèles classiques
# ---------------------------------------------------------------------------

META_COLS = [
    "credibility_score",
    "lie_rate",
    "history_total",
    "is_politician",
]

# Features étendues : inclut les features textuelles et le parti politique
# Utiliser META_COLS_EXTENDED pour les modèles améliorés
META_COLS_EXTENDED = META_COLS + [
    # Texte
    "stmt_word_count",
    "has_absolute_lang",
    "has_hedging",
    # Parti
    "party_republican",
    "party_democrat",
    "party_none",
    "party_org",
    # Sujets (top 10)
    "subj_health_care",
    "subj_taxes",
    "subj_immigration",
    "subj_elections",
    "subj_education",
    "subj_candidates_biography",
    "subj_economy",
    "subj_guns",
    "subj_federal_budget",
    "subj_jobs",
]


# ---------------------------------------------------------------------------
# 1. TF-IDF
# ---------------------------------------------------------------------------

class TfidfFeatures:
    """
    Vectorisation TF-IDF du texte nettoyé.

    Paramètres choisis :
    - ngram_range=(1,2) : unigrammes + bigrammes pour capturer les négations
      ("not true", "never said") absentes avec des unigrammes seuls.
    - max_features=20000 : compromis entre expressivité et dimensionnalité.
    - sublinear_tf=True  : log(tf) atténue l'effet des mots très fréquents.
    - min_df=3           : ignore les mots qui apparaissent dans < 3 docs
      (réduit le bruit et la taille du vocabulaire).

    Limite principale : ne capture pas le sens ni l'ordre des mots.
    "Tax cuts help families" et "Tax cuts hurt families" ont des vecteurs proches.
    """

    def __init__(self, max_features: int = 20_000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
            token_pattern=r"\b[a-zA-Z']{2,}\b",
        )
        self.fitted = False

    def fit_transform(self, texts: pd.Series) -> csr_matrix:
        """
        Entraîne le vectoriseur sur le corpus d'entraînement et transforme.

        Args:
            texts : Serie pandas de textes nettoyés (train set uniquement)

        Returns:
            matrice sparse (n_docs, n_features)
        """
        X = self.vectorizer.fit_transform(texts.fillna(""))
        self.fitted = True
        print(f"TF-IDF fit — vocabulaire : {X.shape[1]} features, "
              f"{X.shape[0]} documents")
        return X

    def transform(self, texts: pd.Series) -> csr_matrix:
        """
        Transforme de nouveaux textes avec le vocabulaire appris.
        À utiliser sur valid, test et données externes.

        Args:
            texts : Serie pandas de textes nettoyés

        Returns:
            matrice sparse (n_docs, n_features)
        """
        if not self.fitted:
            raise RuntimeError("Appelle fit_transform() sur le train avant transform().")
        return self.vectorizer.transform(texts.fillna(""))

    def get_feature_names(self) -> np.ndarray:
        """Retourne les noms des features (tokens) pour SHAP / analyse."""
        return self.vectorizer.get_feature_names_out()

    def save(self, path: str = "models/tfidf_vectorizer.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"TF-IDF sauvegardé -> {path}")

    def load(self, path: str = "models/tfidf_vectorizer.pkl") -> None:
        self.vectorizer = joblib.load(path)
        self.fitted = True
        print(f"TF-IDF chargé depuis {path}")


# ---------------------------------------------------------------------------
# 2. Word2Vec / GloVe (embeddings statiques, moyenne sur les tokens)
# ---------------------------------------------------------------------------

class EmbeddingFeatures:
    """
    Représentation d'un document par la moyenne des vecteurs de ses tokens.

    Stratégie : pour chaque déclaration, on calcule le vecteur moyen
    des embeddings de chaque mot présent dans le vocabulaire pré-entraîné.
    Les mots absents du vocabulaire sont ignorés.

    Avantage par rapport à TF-IDF : capture la similarité sémantique
    ("healthcare" et "medicine" sont proches dans l'espace vectoriel).

    Limite : "not true" et "true" ont des moyennes proches car les embeddings
    statiques ne tiennent pas compte du contexte local.

    Utilisation recommandée :
        from features import EmbeddingFeatures
        emb = EmbeddingFeatures()
        emb.load_glove("data/glove.6B.100d.txt")   # ou load_word2vec(...)
        X_train = emb.transform(train["statement_clean"])
    """

    def __init__(self, dim: int = 100):
        self.dim = dim
        self.vectors: dict = {}

    def load_glove(self, path: str) -> None:
        """
        Charge les vecteurs GloVe pré-entraînés depuis un fichier texte.
        Fichier recommandé : glove.6B.100d.txt (840 MB, 400k mots)
        Téléchargement : https://nlp.stanford.edu/projects/glove/

        Args:
            path : chemin vers le fichier GloVe
        """
        print(f"Chargement GloVe depuis {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if len(vec) == self.dim:
                        self.vectors[word] = vec
                except ValueError:
                    continue
        print(f"GloVe chargé — {len(self.vectors)} vecteurs de dim {self.dim}")

    def load_word2vec(self, path: str) -> None:
        """
        Charge les vecteurs Word2Vec au format gensim KeyedVectors (binary ou txt).

        Args:
            path : chemin vers le fichier .bin ou .txt Word2Vec
        """
        try:
            from gensim.models import KeyedVectors
            print(f"Chargement Word2Vec depuis {path}...")
            kv = KeyedVectors.load_word2vec_format(
                path, binary=path.endswith(".bin")
            )
            self.vectors = {word: kv[word] for word in kv.key_to_index}
            self.dim = kv.vector_size
            print(f"Word2Vec chargé — {len(self.vectors)} vecteurs de dim {self.dim}")
        except ImportError:
            raise ImportError("gensim requis : pip install gensim")

    def _doc_to_vec(self, text: str) -> np.ndarray:
        """
        Calcule le vecteur moyen d'un document.
        Retourne un vecteur nul si aucun mot du document n'est dans le vocabulaire.
        """
        tokens = text.split() if isinstance(text, str) else []
        vecs = [self.vectors[t] for t in tokens if t in self.vectors]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transforme une serie de textes en matrice d'embeddings.

        Args:
            texts : Serie pandas de textes nettoyés

        Returns:
            np.ndarray de shape (n_docs, dim)
        """
        if not self.vectors:
            raise RuntimeError(
                "Charge d'abord les vecteurs avec load_glove() ou load_word2vec()."
            )
        X = np.vstack([self._doc_to_vec(t) for t in texts.fillna("")])
        coverage = np.sum(X.any(axis=1)) / len(X) * 100
        print(f"Embeddings — shape: {X.shape}, "
              f"couverture vocab: {coverage:.1f}%")
        return X


# ---------------------------------------------------------------------------
# 3. Combinaison texte + métadonnées
# ---------------------------------------------------------------------------

class CombinedFeatures:
    """
    Combine les features textuelles (TF-IDF ou embeddings) avec
    les features numériques de métadonnées (credibility_score, lie_rate…).

    L'idée : les métadonnées apportent un signal sur QUI parle,
    le texte apporte un signal sur CE QUI est dit.
    Ensemble, ils sont complémentaires.

    ⚠ Biais à documenter : inclure credibility_score revient à évaluer
    le speaker, pas seulement la déclaration. Voir analyse_biais.

    Utilisation :
        combined = CombinedFeatures(text_features=tfidf, scaler=scaler)
        X_train = combined.fit_transform(train)
        X_valid = combined.transform(valid)
    """

    def __init__(
        self,
        text_features,           # TfidfFeatures ou EmbeddingFeatures
        meta_cols: list = META_COLS,
        scale_meta: bool = True,
    ):
        self.text_features = text_features
        self.meta_cols = meta_cols
        self.scale_meta = scale_meta
        self.scaler = StandardScaler() if scale_meta else None
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray | csr_matrix:
        """
        Entraîne et transforme sur le train set.
        Le text_features doit déjà avoir été fit (fit_transform appelé avant).

        Args:
            df : DataFrame préprocessé avec 'statement_clean' et META_COLS

        Returns:
            matrice combinée (sparse si TF-IDF, dense si embeddings)
        """
        # Features textuelles (déjà fittées)
        X_text = self.text_features.transform(df["statement_clean"])

        # Features de métadonnées
        X_meta = df[self.meta_cols].fillna(0).values.astype(np.float32)
        if self.scale_meta:
            X_meta = self.scaler.fit_transform(X_meta)

        self.fitted = True
        return self._combine(X_text, X_meta)

    def transform(self, df: pd.DataFrame) -> np.ndarray | csr_matrix:
        """
        Transforme valid, test ou données externes.

        Args:
            df : DataFrame préprocessé

        Returns:
            matrice combinée
        """
        if not self.fitted:
            raise RuntimeError("Appelle fit_transform() sur le train d'abord.")

        X_text = self.text_features.transform(df["statement_clean"])
        X_meta = df[self.meta_cols].fillna(0).values.astype(np.float32)
        if self.scale_meta:
            X_meta = self.scaler.transform(X_meta)

        return self._combine(X_text, X_meta)

    def _combine(self, X_text, X_meta: np.ndarray):
        """
        Concatène les features textuelles et les métadonnées.
        Gère le cas sparse (TF-IDF) vs dense (embeddings).
        """
        X_meta_sparse = csr_matrix(X_meta)
        if hasattr(X_text, "toarray"):
            # TF-IDF -> sparse + sparse
            return hstack([X_text, X_meta_sparse])
        else:
            # Embeddings -> dense + dense
            return np.hstack([X_text, X_meta])

    def get_feature_names(self) -> list:
        """
        Retourne les noms de toutes les features pour SHAP.
        Fonctionne uniquement si text_features est un TfidfFeatures.
        """
        if hasattr(self.text_features, "get_feature_names"):
            text_names = list(self.text_features.get_feature_names())
        else:
            text_names = [f"emb_{i}" for i in range(self.text_features.dim)]
        return text_names + self.meta_cols


# ---------------------------------------------------------------------------
# Utilitaire : préparer X et y depuis un DataFrame préprocessé
# ---------------------------------------------------------------------------

def get_X_y(df: pd.DataFrame, feature_builder, fit: bool = False):
    """
    Raccourci pour extraire X (features) et y (labels encodés) d'un DataFrame.

    Args:
        df              : DataFrame préprocessé (issu de preprocessing.preprocess_liar)
        feature_builder : instance de TfidfFeatures, EmbeddingFeatures ou CombinedFeatures
        fit             : True uniquement sur le train set

    Returns:
        tuple (X, y)
        - X : matrice de features
        - y : np.ndarray de labels encodés (0, 1, 2)
    """
    y = df["label_encoded"].values

    if fit:
        if isinstance(feature_builder, TfidfFeatures):
            X = feature_builder.fit_transform(df["statement_clean"])
        elif isinstance(feature_builder, CombinedFeatures):
            # Pour CombinedFeatures, le TF-IDF interne doit être fitté d'abord
            feature_builder.text_features.fit_transform(df["statement_clean"])
            X = feature_builder.fit_transform(df)
        else:
            X = feature_builder.transform(df["statement_clean"])
    else:
        if isinstance(feature_builder, CombinedFeatures):
            X = feature_builder.transform(df)
        elif isinstance(feature_builder, TfidfFeatures):
            X = feature_builder.transform(df["statement_clean"])
        else:
            X = feature_builder.transform(df["statement_clean"])

    print(f"X shape: {X.shape}, y shape: {y.shape}, "
          f"classes: {np.unique(y, return_counts=True)}")
    return X, y
