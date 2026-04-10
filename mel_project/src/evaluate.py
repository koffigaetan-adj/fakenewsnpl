"""
evaluate.py
-----------
Évaluation des modèles, visualisation des métriques,
interprétabilité (SHAP, LIME) et comparaison in-domain / out-of-domain.

Utilisé par : notebooks/03_models.ipynb, notebooks/04_generalization.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour la sauvegarde

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os


LABEL_NAMES = ["fake", "nuanced", "real"]
COULEURS    = ["#d73027", "#fdae61", "#1a9850"]


# ---------------------------------------------------------------------------
# 1. Métriques de base
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, model_name: str = "modèle") -> dict:
    """
    Calcule et affiche les métriques principales pour un modèle.

    Métriques choisies :
    - Accuracy   : proportion de prédictions correctes (toutes classes).
    - F1 macro   : moyenne non pondérée du F1 par classe.
                   Pénalise les mauvaises performances sur les classes rares.
                   Métrique PRINCIPALE pour ce projet (recommandée en cas
                   de déséquilibre entre classes).
    - F1 weighted: F1 pondéré par le support de chaque classe.
                   Plus optimiste que macro si une grande classe est bien prédite.
    - Rapport complet : précision, rappel, F1 par classe.

    Args:
        y_true     : labels réels (np.ndarray)
        y_pred     : labels prédits (np.ndarray)
        model_name : nom du modèle pour l'affichage

    Returns:
        dict avec les métriques principales
    """
    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 macro    : {f1_macro:.4f}   ← métrique principale")
    print(f"  F1 weighted : {f1_weighted:.4f}")
    print(f"\n  F1 par classe :")
    for name, score in zip(LABEL_NAMES, f1_per_class):
        print(f"    {name:<10} : {score:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES, labels=[0, 1, 2], zero_division=0)}")

    return {
        "model"       : model_name,
        "accuracy"    : round(acc, 4),
        "f1_macro"    : round(f1_macro, 4),
        "f1_weighted" : round(f1_weighted, 4),
        "f1_fake"     : round(f1_per_class[0], 4),
        "f1_nuanced"  : round(f1_per_class[1], 4),
        "f1_real"     : round(f1_per_class[2], 4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Génère un tableau comparatif des modèles évalués.

    Args:
        results : liste de dicts retournés par evaluate_model()

    Returns:
        DataFrame trié par F1 macro décroissant
    """
    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df = df.reset_index(drop=True)
    print("\n=== Comparaison des modèles (triés par F1 macro) ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 2. Matrice de confusion
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true, y_pred,
    model_name: str = "modèle",
    save_path: str = None,
    normalize: bool = True,
) -> None:
    """
    Affiche et sauvegarde la matrice de confusion.

    On normalise par ligne (normalize='true') pour voir le taux d'erreur
    par classe indépendamment du support. Un modèle qui prédit toujours
    'nuanced' aurait une ligne parfaite pour nuanced mais 0 partout ailleurs.

    Args:
        y_true     : labels réels
        y_pred     : labels prédits
        model_name : titre du graphique
        save_path  : chemin de sauvegarde (.png), None = pas de sauvegarde
        normalize  : True = pourcentages, False = comptages bruts
    """
    norm = "true" if normalize else None
    cm   = confusion_matrix(y_true, y_pred, normalize=norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES
    )
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(f"Matrice de confusion — {model_name}", pad=14, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Matrice sauvegardée -> {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 3. Comparaison in-domain vs out-of-domain
# ---------------------------------------------------------------------------

def plot_domain_comparison(
    results_in: dict,
    results_out: dict,
    model_name: str = "modèle",
    save_path: str = None,
) -> None:
    """
    Visualise la chute de performance entre le dataset LIAR (in-domain)
    et le dataset BuzzFeed (out-of-domain).

    C'est la visualisation centrale de l'analyse du domain shift.
    Un écart important indique que le modèle a sur-appris les
    caractéristiques spécifiques de LIAR (style PolitiFact, speakers, etc.).

    Args:
        results_in  : dict de métriques in-domain (from evaluate_model)
        results_out : dict de métriques out-of-domain
        model_name  : nom du modèle
        save_path   : chemin de sauvegarde
    """
    metrics = ["accuracy", "f1_macro", "f1_fake", "f1_nuanced", "f1_real"]
    labels  = ["Accuracy", "F1 macro", "F1 fake", "F1 nuanced", "F1 real"]

    x = np.arange(len(metrics))
    width = 0.35

    vals_in  = [results_in.get(m, 0)  for m in metrics]
    vals_out = [results_out.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_in  = ax.bar(x - width/2, vals_in,  width, label="In-domain (LIAR)",    color="#378ADD", alpha=0.85)
    bars_out = ax.bar(x + width/2, vals_out, width, label="Out-of-domain (BuzzFeed)", color="#D85A30", alpha=0.85)

    # Annotations des valeurs
    for bar in bars_in + bars_out:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8
        )

    ax.set_xlabel("Métrique")
    ax.set_ylabel("Score")
    ax.set_title(f"In-domain vs Out-of-domain — {model_name}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique sauvegardé -> {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 4. Interprétabilité — SHAP
# ---------------------------------------------------------------------------

def explain_with_shap(
    model,
    X_train,
    X_explain,
    feature_names: list,
    model_type: str = "linear",
    n_samples: int = 200,
    save_path: str = None,
) -> None:
    """
    Calcule et visualise les valeurs SHAP pour comprendre les décisions du modèle.

    SHAP (SHapley Additive exPlanations) mesure la contribution de chaque
    feature à la prédiction. Contrairement aux coefficients d'une LR,
    SHAP prend en compte les interactions entre features.

    Types d'explainer selon le modèle :
    - 'linear'  : LinearExplainer → Logistic Regression (exact, rapide)
    - 'tree'    : TreeExplainer  → Random Forest, XGBoost (exact pour arbres)

    Visualisation : summary_plot montre les features les plus importantes
    globalement, avec la direction de l'impact (positif/négatif par classe).

    Args:
        model        : modèle sklearn entraîné
        X_train      : données d'entraînement (pour le background SHAP)
        X_explain    : données à expliquer (subset du valid/test)
        feature_names: noms des features (from CombinedFeatures.get_feature_names())
        model_type   : 'linear' ou 'tree'
        n_samples    : nombre d'exemples à expliquer (200 suffit pour le summary)
        save_path    : chemin de sauvegarde du plot
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP non installé — pip install shap")

    print(f"Calcul des valeurs SHAP ({model_type} explainer, {n_samples} samples)...")

    # Sous-échantillonnage pour la vitesse
    if hasattr(X_explain, "toarray"):
        X_bg  = shap.sample(X_train, 100)
        X_exp = X_explain[:n_samples]
    else:
        X_bg  = X_train[:100]
        X_exp = X_explain[:n_samples]

    if model_type == "linear":
        explainer   = shap.LinearExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_exp)
    elif model_type == "tree":
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_exp)
    else:
        raise ValueError("model_type doit être 'linear' ou 'tree'")

    # Summary plot : top 20 features les plus importantes globalement
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_exp,
        feature_names=feature_names,
        class_names=LABEL_NAMES,
        max_display=20,
        show=False,
    )
    plt.title("SHAP — Features les plus importantes (top 20)", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  SHAP plot sauvegardé -> {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 5. Interprétabilité — LIME
# ---------------------------------------------------------------------------

def explain_with_lime(
    model,
    vectorizer,
    text: str,
    label_idx: int = None,
    n_features: int = 10,
) -> None:
    """
    Explique la prédiction d'une déclaration spécifique avec LIME.

    LIME (Local Interpretable Model-agnostic Explanations) perturbe
    localement l'entrée et observe comment la prédiction change.
    Contrairement à SHAP, LIME explique UNE décision, pas le modèle global.

    Cas d'usage typique : analyser pourquoi le modèle se trompe
    sur un exemple précis — utile pour la section "analyse des erreurs".

    Args:
        model      : modèle sklearn avec predict_proba()
        vectorizer : TfidfVectorizer entraîné (from TfidfFeatures)
        text       : déclaration brute à expliquer
        label_idx  : index de la classe à expliquer (None = classe prédite)
        n_features : nombre de features à afficher dans l'explication
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        raise ImportError("LIME non installé — pip install lime")

    def predict_proba_pipeline(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=LABEL_NAMES)

    # Prédiction sur le texte
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label_to_explain = label_idx if label_idx is not None else pred

    print(f"\nTexte : {text[:120]}...")
    print(f"Prédiction : {LABEL_NAMES[pred]} "
          f"(fake={proba[0]:.2f}, nuanced={proba[1]:.2f}, real={proba[2]:.2f})")
    print(f"Classe expliquée : {LABEL_NAMES[label_to_explain]}")

    exp = explainer.explain_instance(
        text,
        predict_proba_pipeline,
        num_features=n_features,
        labels=[label_to_explain],
    )
    exp.show_in_notebook(text=True)


# ---------------------------------------------------------------------------
# 6. Analyse des erreurs
# ---------------------------------------------------------------------------

def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_examples: int = 10,
) -> pd.DataFrame:
    """
    Identifie et affiche les exemples mal classifiés.

    L'analyse des erreurs est une étape clé attendue dans le rapport :
    elle permet de comprendre POURQUOI le modèle se trompe (manque de contexte,
    ambiguïté sémantique, biais de speaker, etc.).

    Args:
        df         : DataFrame préprocessé avec 'statement' et 'speaker'
        y_true     : labels réels
        y_pred     : labels prédits
        n_examples : nombre d'exemples d'erreurs à afficher

    Returns:
        DataFrame des erreurs avec colonnes utiles pour le rapport
    """
    errors_mask = y_true != y_pred
    errors_df = df[errors_mask].copy()
    errors_df["label_true"] = [LABEL_NAMES[i] for i in y_true[errors_mask]]
    errors_df["label_pred"] = [LABEL_NAMES[i] for i in y_pred[errors_mask]]

    print(f"\n=== Analyse des erreurs : {errors_mask.sum()} / {len(y_true)} "
          f"({errors_mask.mean()*100:.1f}%) ===")

    # Distribution des types d'erreurs
    error_types = errors_df.groupby(["label_true", "label_pred"]).size()
    print("\nTypes d'erreurs (vrai -> prédit) :")
    print(error_types.to_string())

    # Exemples concrets
    print(f"\n--- {n_examples} exemples d'erreurs ---")
    cols = ["statement", "speaker", "party", "label_true", "label_pred"]
    cols = [c for c in cols if c in errors_df.columns]

    sample = errors_df[cols].head(n_examples)
    for _, row in sample.iterrows():
        print(f"\n  Texte    : {str(row['statement'])[:100]}...")
        if "speaker" in row:
            print(f"  Speaker  : {row['speaker']} ({row.get('party', '?')})")
        print(f"  Vrai     : {row['label_true']}  |  Prédit : {row['label_pred']}")

    return errors_df[cols]


# ---------------------------------------------------------------------------
# 7. Sauvegarde des résultats
# ---------------------------------------------------------------------------

def save_results(results: list[dict], path: str = "outputs/results.csv") -> None:
    """
    Sauvegarde le tableau comparatif des modèles en CSV.

    Args:
        results : liste de dicts retournés par evaluate_model()
        path    : chemin de sortie
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df.to_csv(path, index=False)
    print(f"Résultats sauvegardés -> {path}")
