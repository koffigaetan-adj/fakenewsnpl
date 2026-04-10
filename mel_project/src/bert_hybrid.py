"""
bert_hybrid.py
--------------
Modèle hybride DistilBERT + métadonnées pour la classification de fake news.

Idée : le DistilBERT standard n'utilise que le texte.
Les modèles classiques (RF) l'écrasent car ils exploitent credibility_score
et lie_rate. Ce modèle combine les deux :

  [CLS token (768 dim)] + [métadonnées normalisées (4 dim)]
       → tête de classification → 3 classes

Architecture de la tête :
  Dropout(0.1) → Linear(772, 256) → GELU → Dropout(0.1) → Linear(256, 3)

Compatibilité : Apple Silicon MPS, CUDA, CPU.
Utilise l'API HuggingFace Trainer pour simplifier la boucle d'entraînement.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ---------------------------------------------------------------------------
# Colonnes de métadonnées injectées dans le modèle hybride
# ---------------------------------------------------------------------------

HYBRID_META_COLS = [
    "credibility_score",
    "lie_rate",
    "history_total",
    "is_politician",
]


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class LIARHybridDataset(Dataset):
    """
    Dataset pour le modèle hybride.
    Contient à la fois les encodings textuels et les features de métadonnées.

    Args:
        texts         : liste / Serie de déclarations BRUTES (pas statement_clean)
                        → BERT bénéficie de la casse et de la ponctuation
        meta_features : np.ndarray (n, len(HYBRID_META_COLS)) déjà normalisé
        labels        : torch.Tensor de labels (0, 1, 2)
        tokenizer     : tokenizer HuggingFace
        max_length    : longueur max de tokenisation
    """

    def __init__(self, texts, meta_features, labels, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.meta   = torch.tensor(meta_features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["meta_features"] = self.meta[idx]
        item["labels"]        = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Modèle hybride
# ---------------------------------------------------------------------------

class DistilBertHybrid(nn.Module):
    """
    DistilBERT avec tête de classification hybride (texte + métadonnées).

    Forward :
        1. Passe le texte dans DistilBERT → récupère le token [CLS] (dim 768)
        2. Concatène avec les features de métadonnées (dim = n_meta)
        3. Passe le vecteur combiné dans la tête de classification

    Retourne un SequenceClassifierOutput compatible HuggingFace Trainer.

    Args:
        model_name   : identifiant HuggingFace du backbone (ex: distilbert-base-uncased)
        n_meta       : nombre de features de métadonnées
        num_labels   : nombre de classes (3 pour fake / nuanced / real)
        dropout_prob : taux de dropout dans la tête
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_meta: int = 4,
        num_labels: int = 3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        hidden_size    = self.bert.config.hidden_size          # 768 pour DistilBERT
        combined_size  = hidden_size + n_meta                  # 772

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_labels),
        )

        self.num_labels = num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        meta_features=None,
        labels=None,
        **kwargs,
    ):
        # 1. Encodage textuel → token CLS
        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_out.last_hidden_state[:, 0, :]   # shape : (batch, 768)

        # 2. Concaténation avec métadonnées
        if meta_features is not None:
            combined = torch.cat([cls_output, meta_features], dim=1)  # (batch, 772)
        else:
            combined = cls_output

        # 3. Classification
        logits = self.classifier(combined)   # (batch, 3)

        # 4. Calcul de la loss si labels fournis
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Normalisation des métadonnées
# ---------------------------------------------------------------------------

def fit_meta_scaler(train_df, meta_cols=HYBRID_META_COLS):
    """Entraîne un StandardScaler sur les métadonnées du train set."""
    scaler = StandardScaler()
    scaler.fit(train_df[meta_cols].fillna(0).values)
    return scaler


def get_meta_array(df, scaler, meta_cols=HYBRID_META_COLS):
    """Retourne un np.ndarray normalisé des métadonnées."""
    return scaler.transform(df[meta_cols].fillna(0).values).astype(np.float32)


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_bert_hybrid(
    train_df,
    valid_df,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    save_dir: str = "models/bert_hybrid",
    max_length: int = 128,
    meta_cols: list = HYBRID_META_COLS,
):
    """
    Fine-tuning du modèle hybride DistilBERT + métadonnées.

    Différences clés vs le DistilBERT standard :
    - learning_rate=3e-5 (vs 2e-5) : légèrement plus élevé car la tête custom
      a besoin d'apprendre plus vite que le backbone pré-entraîné.
    - num_epochs=4 (vs 3) : une époque de plus pour que la tête converge.
    - warmup_ratio=0.06 (vs 0.10) : warmup plus court car lr plus élevé.

    Args:
        train_df     : DataFrame préprocessé train (doit avoir META_COLS)
        valid_df     : DataFrame préprocessé valid
        model_name   : backbone HuggingFace
        num_epochs   : nombre d'époques
        batch_size   : taille des batchs
        learning_rate: taux d'apprentissage
        save_dir     : dossier de sauvegarde
        max_length   : longueur max de tokenisation
        meta_cols    : colonnes de métadonnées à injecter

    Returns:
        tuple (trainer, model, scaler)
    """
    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device : Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device : CPU")

    # --- Tokenizer & Scaler ---
    print(f"\nChargement tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Normalisation des métadonnées...")
    scaler = fit_meta_scaler(train_df, meta_cols)
    meta_train = get_meta_array(train_df, scaler, meta_cols)
    meta_valid = get_meta_array(valid_df, scaler, meta_cols)

    # --- Datasets ---
    print("Tokenisation...")
    y_train = torch.tensor(train_df["label_encoded"].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df["label_encoded"].values, dtype=torch.long)

    train_dataset = LIARHybridDataset(
        train_df["statement"].tolist(), meta_train, y_train, tokenizer, max_length
    )
    valid_dataset = LIARHybridDataset(
        valid_df["statement"].tolist(), meta_valid, y_valid, tokenizer, max_length
    )

    # --- Modèle ---
    print(f"\nInitialisation DistilBertHybrid (backbone={model_name}, n_meta={len(meta_cols)})")
    model = DistilBertHybrid(
        model_name=model_name,
        n_meta=len(meta_cols),
        num_labels=3,
    )
    model.to(device)

    # --- Métriques ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy"   : accuracy_score(labels, preds),
            "f1_macro"   : f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=0.06,
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

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning ({num_epochs} époques, batch={batch_size}, lr={learning_rate})...")
    trainer.train()

    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModèle sauvegardé -> {save_dir}")

    return trainer, model, scaler


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

def predict_bert_hybrid(
    texts,
    meta_df,
    scaler,
    save_dir: str = "models/bert_hybrid",
    meta_cols: list = HYBRID_META_COLS,
    batch_size: int = 64,
    max_length: int = 128,
):
    """
    Inférence avec le modèle hybride sauvegardé.

    Args:
        texts     : liste / Serie de textes bruts
        meta_df   : DataFrame avec les colonnes META_COLS (non normalisées)
        scaler    : StandardScaler ajusté sur le train
        save_dir  : dossier du modèle sauvegardé
        meta_cols : colonnes de métadonnées
        batch_size: taille des batchs
        max_length: longueur max de tokenisation

    Returns:
        tuple (predictions, probabilities)
        - predictions   : np.ndarray int (0, 1, 2)
        - probabilities : np.ndarray float (n, 3)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    # Recharge le modèle hybride
    model = DistilBertHybrid(n_meta=len(meta_cols))
    state = torch.load(
        os.path.join(save_dir, "pytorch_model.bin"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    meta_array = get_meta_array(meta_df, scaler, meta_cols)
    texts_list = list(texts)

    all_preds, all_probs = [], []
    import torch.nn.functional as F

    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i:i + batch_size]
        batch_meta  = torch.tensor(meta_array[i:i + batch_size], dtype=torch.float32).to(device)

        inputs = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                meta_features=batch_meta,
            )
            probs = F.softmax(output.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.vstack(all_probs)
