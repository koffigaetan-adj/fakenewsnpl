"""
train.py
--------
Entraînement des modèles de classification de fake news.

Modèles implémentés :
  1. Logistic Regression  (baseline interprétable)
  2. Random Forest        (non-linéaire, robuste)
  3. XGBoost              (gradient boosting, souvent meilleur sur tabular+texte)
  4. BERT / DistilBERT    (fine-tuning, modèle avancé)

Apple Silicon (M4 Pro) : BERT utilise le backend MPS automatiquement.
"""

import os
import numpy as np
import joblib
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost non installé — pip install xgboost")


# ---------------------------------------------------------------------------
# 1. Modèles classiques
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, C: float = 1.0) -> LogisticRegression:
    """
    Logistic Regression multiclasse avec régularisation L2.

    Choix méthodologiques :
    - class_weight='balanced' : compense le déséquilibre entre classes.
    - solver='lbfgs' : efficace pour les problèmes multiclasses sparse.
    - max_iter=1000  : suffisant pour converger sur LIAR avec TF-IDF.
    - C=1.0          : régularisation par défaut.

    Avantage principal : coefficients directement interprétables,
    compatibles avec SHAP LinearExplainer.
    """
    print("Entraînement Logistic Regression...")
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        # multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  LR — terminé")
    return model


def train_random_forest(
    X_train, y_train,
    n_estimators: int = 300,
    max_depth: int = None,
) -> RandomForestClassifier:
    """
    Random Forest multiclasse.

    Choix méthodologiques :
    - n_estimators=300 : bon compromis vitesse/variance.
    - class_weight='balanced_subsample' : recalcule les poids par bootstrap,
      plus robuste que 'balanced' pour les forêts.
    - max_features='sqrt' : standard pour la classification.

    Avantage : capture les interactions non-linéaires entre features.
    Limite sur TF-IDF pur : haute dimensionnalité sparse → meilleures
    performances avec les combined features.
    """
    print("Entraînement Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    print("  RF — terminé")
    return model


def train_xgboost(X_train, y_train) -> Any:
    """
    XGBoost multiclasse avec gestion du déséquilibre par sample_weight.

    Choix méthodologiques :
    - objective='multi:softprob' : sorties en probabilités pour les 3 classes.
    - tree_method='hist'         : algorithme rapide compatible Apple Silicon.
    - device MPS non encore stable dans XGBoost → on reste CPU + n_jobs=-1.

    Note : XGBoost préfère les matrices denses — conversion automatique
    si la matrice d'entrée est sparse (TF-IDF).
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost non installé — pip install xgboost")

    print("Entraînement XGBoost...")

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weight = np.array([weights[y] for y in y_train])

    # Conversion sparse -> dense si nécessaire
    if hasattr(X_train, "toarray"):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_dense, y_train, sample_weight=sample_weight)
    print("  XGBoost — terminé")
    return model


def save_classical_model(model, name: str, save_dir: str = "models") -> None:
    """Sauvegarde un modèle sklearn/xgboost avec joblib."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Sauvegardé -> {path}")


def load_classical_model(name: str, save_dir: str = "models") -> Any:
    """Charge un modèle sauvegardé."""
    path = os.path.join(save_dir, f"{name}.pkl")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 2. Fine-tuning BERT (Apple Silicon MPS)
# ---------------------------------------------------------------------------

def get_device():
    """
    Détecte automatiquement le meilleur device disponible :
    - MPS  : Apple Silicon (M1/M2/M3/M4)
    - CUDA : GPU NVIDIA
    - CPU  : fallback
    """
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device : Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device : CPU")
    return device


class LIARDataset:
    """
    Dataset PyTorch pour le fine-tuning BERT sur LIAR.

    Note : on tokenise le texte ORIGINAL (pas statement_clean) car
    BERT bénéficie de la ponctuation, de la casse et des majuscules
    pour mieux comprendre le contexte des déclarations politiques.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_bert(
    train_df,
    valid_df,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    save_dir: str = "models/bert_finetuned",
    max_length: int = 128,
):
    """
    Fine-tuning d'un modèle BERT/DistilBERT pour la classification 3 classes.

    Choix méthodologiques :
    - distilbert-base-uncased : 40% plus léger que BERT-base, performances
      proches, bien adapté au M4 Pro. Remplacer par 'roberta-base' pour
      de meilleures performances (mais 2x plus lent).
    - max_length=128  : suffisant pour LIAR (~18 mots en moyenne).
    - learning_rate=2e-5 : valeur standard pour le fine-tuning BERT.
      Trop élevée → catastrophic forgetting, trop basse → sous-apprentissage.
    - warmup_ratio=0.1 : 10% des steps en warmup pour stabiliser le début.
    - load_best_model_at_end=True : on garde le meilleur checkpoint (F1 macro).

    MPS : os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" active le fallback
    CPU pour les rares opérateurs non encore supportés par MPS.

    Args:
        train_df     : DataFrame préprocessé train
        valid_df     : DataFrame préprocessé valid
        model_name   : identifiant HuggingFace du modèle de base
        num_epochs   : nombre d'époques (3 suffit pour fine-tuning)
        batch_size   : taille des batchs (réduire à 16 si OOM)
        learning_rate: taux d'apprentissage
        save_dir     : dossier de sauvegarde
        max_length   : longueur max de tokenisation en tokens

    Returns:
        tuple (trainer, model)
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from sklearn.metrics import f1_score, accuracy_score
    except ImportError:
        raise ImportError(
            "Installe les dépendances : pip install transformers torch"
        )

    device = get_device()

    print(f"\nChargement : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "fake", 1: "nuanced", 2: "real"},
        label2id={"fake": 0, "nuanced": 1, "real": 2},
    )
    model.to(device)

    print("Tokenisation...")
    train_dataset = LIARDataset(
        train_df["statement"].tolist(),
        torch.tensor(train_df["label_encoded"].values, dtype=torch.long),
        tokenizer,
        max_length=max_length,
    )
    valid_dataset = LIARDataset(
        valid_df["statement"].tolist(),
        torch.tensor(valid_df["label_encoded"].values, dtype=torch.long),
        tokenizer,
        max_length=max_length,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy"   : accuracy_score(labels, preds),
            "f1_macro"   : f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    # use_mps_device supprimé dans transformers >= 4.38
    # Le Trainer détecte MPS automatiquement via accelerate
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning ({num_epochs} époques, batch={batch_size}, device={device})...")
    trainer.train()

    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModèle sauvegardé -> {save_dir}")

    return trainer, model


def predict_bert(texts, save_dir: str = "models/bert_finetuned", batch_size: int = 64):
    """
    Inférence avec le modèle BERT fine-tuné sauvegardé.
    Utilisé pour l'évaluation out-of-domain (BuzzFeed).

    Args:
        texts     : liste ou Serie de textes bruts
        save_dir  : dossier du modèle sauvegardé
        batch_size: taille des batchs d'inférence

    Returns:
        tuple (predictions, probabilities)
        - predictions   : np.ndarray int (0, 1, 2)
        - probabilities : np.ndarray float (n, 3)
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    all_preds, all_probs = [], []
    texts_list = list(texts)

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.vstack(all_probs)
