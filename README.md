# TruthScope - Fake News Detection NLP Pipeline

TruthScope is a comprehensive Natural Language Processing (NLP) project focused on the automatic detection of political fake news. It leverages the LIAR dataset (PolitiFact) for training and evaluation, and incorporates the WELFake dataset for Out-of-Domain (OOD) generalization testing.

The project encompasses a complete pipeline: from initial data exploration and preprocessing to training classical machine learning models (Logistic Regression, Random Forest, XGBoost), fine-tuning transformer models (DistilBERT/BERT), and deploying an interactive Streamlit dashboard.

## Project Objectives

1.  **Analyze and understand** the linguistic and metadata characteristics of political statements using the LIAR dataset.
2.  **Evaluate different machine learning approaches** for fake news classification (TF-IDF with classical models vs. Deep Learning with BERT).
3.  **Investigate the impact of class definition** (6-class ordinal vs. binary classification).
4.  **Assess model robustness** and generalization capabilities on external data (Out-of-Domain evaluation using WELFake).
5.  **Provide an intuitive, professional dashboard** for data visualization, model interpretability, and live testing.

## Datasets

### 1. LIAR Dataset (In-Domain)
*   **Source:** PolitiFact API (collected by Wang, ACL 2017).
*   **Content:** 12,836 short political statements from various contexts (TV debates, speeches, tweets, etc.).
*   **Classes:** 6 ordinal labels (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`).
*   **Features:** Statement text, subject, speaker, speaker's job title, state, party affiliation, historical credibility constraints, and context.

### 2. WELFake Dataset (Out-of-Domain)
*   **Source:** Combined dataset of news articles.
*   **Content:** 72,134 full-length news articles (significantly longer than LIAR statements).
*   **Classes:** 2 labels (Fake/Real).
*   **Usage:** Used strictly for testing how well the models trained on short political statements generalize to long-form general news articles.

## Project Structure (Notebooks)

The analytical and modeling work is structured across 7 sequential notebooks:

*   **01_exploration.ipynb:** Comprehensive Exploratory Data Analysis (EDA). Analyzes label distributions, speaker biases, statement lengths, political party trends, and metadata credibility scores.
*   **02_preprocessing.ipynb:** Text cleaning pipeline (lowercasing, punctuation removal, stopward removal, lemmatization) and TF-IDF vectorization setup.
*   **03_models.ipynb:** Baseline models training (Logistic Regression, Random Forest, XGBoost) on the 6-class problem. Includes performance metrics and confusion matrices.
*   **04_improved_features.ipynb:** Feature engineering experiments. Tests the inclusion of metadata (party, subject) and linguistic features directly into the text or alongside TF-IDF vectors to improve accuracy.
*   **05_bert_improved.ipynb:** Fine-tuning of transformer models (DistilBERT and BERT-base) for nuanced semantic understanding.
*   **06_binary_vs_multiclass.ipynb:** Methodological shift from a 6-class problem to a binary classification (`fake` vs `real`), discarding the ambiguous `half-true` class, resulting in significant performance gains.
*   **07_outdomain_evaluation.ipynb:** Final evaluation of the binary models on the WELFake dataset, highlighting the challenges of cross-domain fake news detection and the drop in performance when losing metadata context.

## TruthScope Dashboard

The interactive Streamlit dashboard (`dashboard.py`) provides a premium, dark-themed interface to explore the project's findings.

### Features
*   **Exploration EDA:** Interactive Plotly charts showing dataset statistics, metadata influence, and missing values.
*   **Model Comparison:** Results and metrics for all trained models, including detailed confusion matrices and SHAP feature importance for explainability.
*   **Configuration Analysis:** Visual comparisons between different feature engineering approaches and the impact of binary vs. multi-class problem framing.
*   **Live Demo:** A real-time inference engine allowing users to:
    *   Input manual text (optimized for short US political statements).
    *   Test pre-selected LIAR examples.
    *   Sample and predict on random articles from the WELFake dataset.

## Installation and Usage

### Prerequisites
*   Python 3.8+
*   pip

### Setup Instructions

1.  **Clone the repository or download the project files.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Placement:** Ensure the data files are located in one of the following directory structures relative to `dashboard.py`:
    *   `data/raw/train.tsv`, `data/raw/valid.tsv`, `data/raw/test.tsv`
    *   `data/train.tsv`, `data/valid.tsv`, `data/test.tsv`
    *   For the OOD demo: `data/external/WELFake_Dataset.csv`
4.  **Run the Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

## Team Team (Epitech Digital - 2026)

*   Melvyn CHARLES
*   Koffi ADJAI
*   FANAINGAMAMPIANDRA S. Jo
*   RAMANDIMBISOA Heliaritiana
*   Antoine BENECH
