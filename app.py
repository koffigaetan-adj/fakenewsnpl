# ============================================================
# TRUTHSCOPE - DASHBOARD STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import base64
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="TruthScope",
    page_icon="",
    layout="wide"
)

# ============================================================
# IMAGE BASE64
# ============================================================

def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

hero_img = get_base64_image("assets/hero.avif")

# ============================================================
# CSS GLOBAL
# ============================================================

st.markdown(f"""
<style>
    /* Reset global */
    * {{ box-sizing: border-box; }}

    /* Background principal */
    .stApp {{
        background-color: #0f0f0f !important;
        color: #f1f5f9 !important;
    }}

    /* Ligne orange animée en haut — fine */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #fb923c, #fed7aa, #fb923c);
        background-size: 200% 100%;
        animation: shimmer 2s infinite linear;
        z-index: 9999;
    }}
    @keyframes shimmer {{
        0%   {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #0a0a0a !important;
        border-right: 1px solid #2a2a2a !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #e2e8f0 !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: #2a2a2a !important;
        opacity: 1 !important;
    }}

    /* Contenu principal */
    .main .block-container {{
        background-color: #0f0f0f !important;
        padding-top: 2rem !important;
    }}

    /* Titres */
    h1 {{ color: #ffffff !important; font-weight: 700 !important; }}
    h2 {{ color: #f1f5f9 !important; font-weight: 600 !important; }}
    h3 {{ color: #e2e8f0 !important; font-weight: 600 !important; }}
    p  {{ color: #94a3b8 !important; }}
    label {{ color: #94a3b8 !important; }}

    /* KPI cards */
    .kpi-card {{
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #fb923c, #fed7aa);
        border-radius: 14px 14px 0 0;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(251,146,60,0.15);
        border-color: #fb923c;
    }}
    .kpi-icon {{
        font-size: 28px;
        margin-bottom: 8px;
        display: block;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        display: block;
        line-height: 1.2;
    }}
    .kpi-label {{
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
        display: block;
    }}
    .kpi-sub {{
        font-size: 11px;
        color: #fb923c;
        margin-top: 6px;
        display: block;
    }}
    /* Radio button sélectionné → orange */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div p {{
    color: #fb923c !important;
    font-weight: 700 !important;
}}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] {{
    background-color: rgba(251,146,60,0.15) !important;
    border-color: #fb923c !important;
}}

/* Hover sur les options */
[data-testid="stSidebar"] .stRadio label:hover p {{
    color: #fb923c !important;
}}

    /* Cards */
    .card {{
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(251,146,60,0.1);
        border-color: #fb923c;
    }}
    .card h3 {{
        color: #ffffff !important;
        margin-top: 0 !important;
        font-size: 16px !important;
    }}
    .card p {{
        color: #94a3b8 !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
        margin: 4px 0 !important;
    }}

    /* Card interprétation */
    .card-interpretation {{
        background: #141414;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #fb923c;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 8px 0;
    }}
    .card-interpretation h4 {{
        color: #fb923c !important;
        margin-top: 0 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700 !important;
    }}
    .card-interpretation p {{
        color: #94a3b8 !important;
        font-size: 13px !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }}

    /* Hero */
    .hero {{
        position: relative;
        width: 100%;
        height: 300px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 32px;
        border: 1px solid #2a2a2a;
    }}
    .hero img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: brightness(0.3);
    }}
    .hero-overlay {{
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(251,146,60,0.15), transparent);
    }}
    .hero-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        width: 80%;
    }}
    .hero-text h1 {{
        color: #ffffff !important;
        font-size: 40px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }}
    .hero-text p {{
        color: #94a3b8 !important;
        font-size: 16px !important;
        margin-top: 10px !important;
    }}
    .hero-badge {{
        background: rgba(251,146,60,0.15);
        color: #fb923c !important;
        border: 1px solid rgba(251,146,60,0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 14px;
        letter-spacing: 0.04em;
    }}

    /* Badge sidebar */
    .badge {{
        background: rgba(251,146,60,0.15);
        color: #fb923c !important;
        border: 1px solid rgba(251,146,60,0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* Tags équipe */
    .team-tag {{
        background: rgba(251,146,60,0.08);
        border: 1px solid rgba(251,146,60,0.2);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 13px;
        color: #fb923c !important;
        font-weight: 500;
        display: inline-block;
        margin: 4px;
        transition: background 0.2s;
    }}
    .team-tag:hover {{
        background: rgba(251,146,60,0.15);
    }}

    /* Bouton */
    .stButton button {{
        background-color: #fb923c !important;
        color: #0f0f0f !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }}
    .stButton button:hover {{
        background-color: #ea7c20 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(251,146,60,0.3) !important;
    }}

    /* Tags résultat */
    .tag-fake {{
        background: rgba(239,68,68,0.1);
        color: #fca5a5 !important;
        padding: 8px 24px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
        border: 1px solid rgba(239,68,68,0.3);
    }}
    .tag-true {{
        background: rgba(34,197,94,0.1);
        color: #86efac !important;
        padding: 8px 24px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
        border: 1px solid rgba(34,197,94,0.3);
    }}

    /* Inputs */
    .stSelectbox > div > div {{
        background-color: #1a1a1a !important;
        color: #f1f5f9 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }}
    .stTextArea textarea {{
        background-color: #1a1a1a !important;
        color: #f1f5f9 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }}
    .stTextArea textarea:focus {{
        border-color: #fb923c !important;
        box-shadow: 0 0 0 2px rgba(251,146,60,0.15) !important;
    }}

    /* Tableau */
    .stDataFrame {{ border-radius: 12px !important; overflow: hidden !important; }}

    /* Divider */
    hr {{ border-color: #1e1e1e !important; margin: 1.5rem 0 !important; }}

    /* Transition page */
    .main {{ animation: fadeIn 0.35s ease; }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTION GRAPHIQUES DARK
# ============================================================

def dark_layout(fig, title=""):
    fig.update_layout(
        title=title,
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font_color="#94a3b8",
        title_font_color="#ffffff",
        legend_font_color="#94a3b8",
        xaxis=dict(
            gridcolor="#2a2a2a",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="#2a2a2a",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8")
        ),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "<h1 style='color:#fb923c; font-size:32px; font-weight:800; "
    "letter-spacing:-0.02em; margin-bottom:4px;'>TruthScope</h1>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<span class='badge'>v1.0</span>"
    "<span style='color:#4a4a4a; font-size:11px; margin-left:8px;'>Fake News Detector</span>",
    unsafe_allow_html=True
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Analyse EDA", "Modèles", "Demo Live"],
    format_func=lambda x: {
        "Accueil"     : "⌂  Accueil",
        "Analyse EDA" : "⊞  Analyse EDA",
        "Modèles"     : "◈  Modèles",
        "Demo Live"   : "◎  Demo Live"
    }[x]
)

st.sidebar.divider()
st.sidebar.markdown(
    "<p style='color:#fb923c; font-size:11px; font-weight:700; "
    "text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Équipe</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("""
<p style='color:#fb923c; font-size:12px; line-height:2; margin:0;'>
    Melvyn CHARLES<br>
    Koffi Adjaï<br>
    FANAINGAMAMPIANDRA S. Jo<br>
    RAMANDIMBISOA Heliaritiana<br>
    Antoine Benech
</p>
<p style='color:#4a4a4a; font-size:11px; margin-top:16px; text-align:center;'>
    Epitech Digital School — 2026
</p>
""", unsafe_allow_html=True)

# ============================================================
# DONNÉES & MODÈLES
# ============================================================

ORDRE_LABELS = ["pants-fire","false","barely-true","half-true","mostly-true","true"]
COULEURS     = ["#ef4444","#f97316","#fbbf24","#facc15","#84cc16","#22c55e"]
FAKE_LABELS  = ["pants-fire","false","barely-true"]
STOPWORDS    = set(stopwords.words("english"))

@st.cache_data
def load_data():
    COLONNES = [
        "id","label","statement","subject","speaker","job_title",
        "state","party","barely_true_counts","false_counts",
        "half_true_counts","mostly_true_counts","pants_on_fire_counts","context"
    ]
    train = pd.read_csv("data/train.tsv", sep="\t", header=None, names=COLONNES)
    valid = pd.read_csv("data/valid.tsv", sep="\t", header=None, names=COLONNES)
    for df in [train, valid]:
        df["context"]      = df["context"].fillna("")
        df["speaker"]      = df["speaker"].fillna("unknown")
        df["party"]        = df["party"].fillna("unknown")
        df["subject"]      = df["subject"].fillna("")
        df["nb_mots"]      = df["statement"].str.split().str.len()
        df["label_binary"] = df["label"].apply(lambda x: "fake" if x in FAKE_LABELS else "true")
    return train, valid

def preprocess_text(text):
    text   = text.lower().replace("-", " ")
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

@st.cache_resource
def train_models(_train, _valid):
    train_clean = _train["statement"].apply(preprocess_text)
    valid_clean = _valid["statement"].apply(preprocess_text)
    tfidf   = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
    X_train = tfidf.fit_transform(train_clean)
    X_valid = tfidf.transform(valid_clean)
    y_train = _train["label_binary"]
    y_valid = _valid["label_binary"]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return tfidf, lr, rf, X_valid, y_valid

train, valid    = load_data()
tfidf, lr, rf, X_valid, y_valid = train_models(train, valid)
y_pred_lr = lr.predict(X_valid)
y_pred_rf = rf.predict(X_valid)

# ============================================================
# PAGE ACCUEIL
# ============================================================

if page == "Accueil":

    # Hero
    if hero_img:
        st.markdown(f"""
        <div class='hero'>
            <img src='data:image/avif;base64,{hero_img}'/>
            <div class='hero-overlay'></div>
            <div class='hero-text'>
                <span class='hero-badge'>EPITECH DIGITAL SCHOOL — 2026</span>
                <h1>TruthScope</h1>
                <p>Détection automatique de fake news politiques par NLP</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.title("TruthScope")

    # KPI cards custom
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("◎", "10 240", "Déclarations", "LIAR Dataset"),
        ("◈", "6",      "Modèles testés", "LR · RF · XGB"),
        ("⊞", "61.53%", "Meilleure accuracy", "RF + TF-IDF"),
        ("⌘", "2",      "Vectorisations", "TF-IDF · Word2Vec"),
    ]
    for col, (icon, val, label, sub) in zip([col1,col2,col3,col4], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <span class='kpi-icon'>{icon}</span>
                <span class='kpi-value'>{val}</span>
                <span class='kpi-label'>{label}</span>
                <span class='kpi-sub'>{sub}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>A propos du projet</h3>
            <p>TruthScope est un système de détection automatique
            de fake news politiques basé sur le LIAR Dataset de PolitiFact.
            On classifie les déclarations en deux catégories : FAKE ou TRUE.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>Dataset LIAR</h3>
            <p><b style='color:#fb923c;'>12 800</b> déclarations politiques américaines annotées par PolitiFact.</p>
            <p><b style='color:#fb923c;'>Classification</b> : binaire — fake / true</p>
            <p><b style='color:#fb923c;'>Source</b> : William Yang Wang (ACL 2017)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Résultats des modèles")

    col1, col2, col3 = st.columns(3)
    modeles = [
        ("TF-IDF", [("LR", "61.37%"), ("RF", "61.53% ✓"), ("XGB", "58.72%")]),
        ("Word2Vec", [("LR", "60.05%"), ("RF", "58.72%"), ("XGB", "59.89%")]),
        ("BERT", None),
    ]
    for col, (nom, resultats) in zip([col1,col2,col3], modeles):
        with col:
            if resultats:
                rows = "".join([
                    f"<p><span style='color:#fb923c; font-weight:600;'>{m}</span>"
                    f"&nbsp;→&nbsp;<span style='color:#e2e8f0;'>{v}</span></p>"
                    for m, v in resultats
                ])
                st.markdown(f"""
                <div class='card'>
                    <h3>{nom}</h3>
                    {rows}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='card'>
                    <h3>{nom}</h3>
                    <p style='color:#fb923c !important; font-weight:600;'>
                    En cours — Google Colab GPU</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Équipe projet")
    membres = [
        "Melvyn CHARLES", "Koffi Adjaï",
        "FANAINGAMAMPIANDRA S. Jo",
        "RAMANDIMBISOA Heliaritiana", "Antoine Benech"
    ]
    tags = "".join([f"<span class='team-tag'>{m}</span>" for m in membres])
    st.markdown(f"""
    <div class='card'>
        <div style='display:flex; gap:8px; flex-wrap:wrap;'>{tags}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE EDA
# ============================================================
elif page == "Analyse EDA":

    st.markdown("# Analyse Exploratoire des Données")
    st.markdown(
        "<p style='color:#64748b;'>LIAR Dataset — Train set · 10 240 déclarations</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # 1. Distribution des labels
    st.markdown("### Distribution des labels")
    counts = train["label"].value_counts().reindex(ORDRE_LABELS)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig1 = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_sequence=COULEURS,
            hole=0.55
        )
        fig1.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont_color="#ffffff",
            marker=dict(line=dict(color="#0f0f0f", width=2))
        )
        fig1 = dark_layout(fig1)
        fig1.update_layout(
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(
                text="Labels", x=0.5, y=0.5,
                font=dict(size=14, color="#ffffff"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig1, use_container_width=True, key="eda_fig1")

    with col2:
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>Classes relativement équilibrées entre 16% et 20%.<br><br>
            <b style='color:#fb923c;'>pants-fire</b> est sous-représenté
            à 8.2% — deux fois moins que les autres.<br><br>
            On utilisera <b style='color:#fb923c;'>class_weight="balanced"</b>
            pour corriger ce déséquilibre.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 2. Speakers + Partis
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 speakers")
        top_spk = train["speaker"].value_counts().head(10).reset_index()
        top_spk.columns = ["speaker", "count"]
        fig2 = px.bar(
            top_spk, x="count", y="speaker",
            orientation="h", color="count",
            color_continuous_scale=[[0,"#2a2a2a"],[1,"#fb923c"]],
            text="count",
            labels={"count":"Déclarations","speaker":""}
        )
        fig2.update_traces(textposition="outside", textfont_color="#ffffff")
        fig2 = dark_layout(fig2)
        fig2.update_layout(
            yaxis={"categoryorder":"total ascending"},
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True, key="eda_fig2")
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>Obama domine avec 488 déclarations — risque de sur-apprentissage.
            "chain-email" représente des emails viraux anonymes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Distribution par parti")
        top_par = train["party"].value_counts().head(8).reset_index()
        top_par.columns = ["party", "count"]
        fig3 = px.treemap(
            top_par,
            path=["party"],
            values="count",
            color="count",
            color_continuous_scale=[[0,"#1a1a1a"],[0.5,"#7c3000"],[1,"#fb923c"]]
        )
        fig3.update_traces(
            textfont=dict(color="#ffffff", size=13),
            marker=dict(line=dict(width=2, color="#0f0f0f"))
        )
        fig3 = dark_layout(fig3)
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True, key="eda_fig3")
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>Republicans (44%) et Democrats (33%) dominent.
            Les autres partis sont très minoritaires — biais potentiel.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 3. Véracité par parti
    st.markdown("### Véracité par parti politique (top 5)")
    top5 = train["party"].value_counts().head(5).index
    df_g = (
        train[train["party"].isin(top5)]
        .groupby(["party","label"]).size().reset_index(name="count")
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        fig4 = px.bar(
            df_g, x="party", y="count",
            color="label",
            barmode="stack",
            category_orders={"label": ORDRE_LABELS},
            color_discrete_sequence=COULEURS,
            labels={"count":"Déclarations","party":"Parti","label":"Label"}
        )
        fig4 = dark_layout(fig4)
        fig4.update_layout(
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig4, use_container_width=True, key="eda_fig4")
    with col2:
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>Les républicains ont plus de
            <b style='color:#ef4444;'>pants-fire</b>
            et <b style='color:#f97316;'>false</b>.<br><br>
            Cela peut refléter un biais de PolitiFact
            qui aurait vérifié davantage de déclarations
            républicaines controversées.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 4. Longueur
    st.markdown("### Longueur des déclarations")
    col1, col2 = st.columns(2)
    with col1:
        fig5 = px.histogram(
            train, x="nb_mots", nbins=50,
            color_discrete_sequence=["#fb923c"],
            labels={"nb_mots":"Nombre de mots","count":"Fréquence"}
        )
        fig5.add_vline(
            x=train["nb_mots"].mean(),
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"Moy. {train['nb_mots'].mean():.1f} mots",
            annotation_font_color="#ef4444"
        )
        fig5 = dark_layout(fig5, "Distribution de la longueur")
        st.plotly_chart(fig5, use_container_width=True, key="eda_fig5")

    with col2:
        moy = (
            train.groupby("label")["nb_mots"]
            .mean().reindex(ORDRE_LABELS).round(1).reset_index()
        )
        moy.columns = ["label", "moy"]
        fig6 = px.bar(
            moy, x="label", y="moy",
            color="label",
            color_discrete_sequence=COULEURS,
            text="moy",
            labels={"moy":"Mots en moyenne","label":"Label"}
        )
        fig6.update_traces(textposition="outside", textfont_color="#ffffff")
        fig6 = dark_layout(fig6, "Longueur moyenne par label")
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True, key="eda_fig6")

    st.markdown("""
    <div class='card-interpretation'>
        <h4>Interprétation</h4>
        <p>Déclarations très courtes en moyenne (18 mots).
        Les différences entre labels sont minimes (17–18.8 mots)
        — la longueur seule ne suffit pas à détecter les fake news.
        Un outlier à 467 mots est conservé.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE MODÈLES
# ============================================================

elif page == "Modèles":

    st.markdown("# Comparaison des modèles")
    st.markdown(
        "<p style='color:#64748b;'>Valid set · Classification binaire fake / true</p>",
        unsafe_allow_html=True
    )
    st.divider()

    df_res = pd.DataFrame({
        "Modèle"        : ["LR + TF-IDF","RF + TF-IDF","XGB + TF-IDF",
                           "LR + Word2Vec","RF + Word2Vec","XGB + Word2Vec"],
        "Vectorisation" : ["TF-IDF"]*3 + ["Word2Vec"]*3,
        "Accuracy"      : [0.6137, 0.6153, 0.5872, 0.6005, 0.5872, 0.5989],
        "F1-score"      : [0.61, 0.62, 0.59, 0.60, 0.59, 0.60]
    })

    st.markdown("### Tableau comparatif")
    def highlight_best(row):
        if row["Accuracy"] == df_res["Accuracy"].max():
            return ["background-color:#1f1a0e; color:#fb923c; font-weight:700"]*len(row)
        return [""]*len(row)

    st.dataframe(
        df_res.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Accuracy TF-IDF vs Word2Vec")
    col1, col2 = st.columns([2,1])
    with col1:
        fig_c = px.bar(
            df_res, x="Modèle", y="Accuracy",
            color="Vectorisation", barmode="group",
            text="Accuracy",
            color_discrete_sequence=["#fb923c","#64748b"],
            labels={"Accuracy":"Accuracy","Modèle":""}
        )
        fig_c.update_traces(textposition="outside", textfont_color="#ffffff")
        fig_c = dark_layout(fig_c)
        fig_c.update_layout(yaxis_range=[0,1])
        st.plotly_chart(fig_c, use_container_width=True)
    with col2:
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>TF-IDF performe mieux que Word2Vec.<br><br>
            Les textes très courts (18 mots) font perdre
            de l'info à Word2Vec lors de la moyenne.<br><br>
            <b style='color:#fb923c;'>Meilleur : RF + TF-IDF → 61.53%</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Matrice de confusion — RF + TF-IDF")
    col1, col2 = st.columns([2,1])
    with col1:
        cm = confusion_matrix(y_valid, y_pred_rf, labels=["fake","true"])
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Prédit : fake","Prédit : true"],
            y=["Réel : fake","Réel : true"],
            colorscale=[[0,"#1a1a1a"],[1,"#fb923c"]],
            showscale=True,
            font_colors=["#ffffff","#0f0f0f"]
        )
        fig_cm.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font_color="#94a3b8",
            margin=dict(t=30,b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Interprétation</h4>
            <p>Le modèle classe de manière équilibrée
            les deux catégories.<br><br>
            Erreurs symétriques entre faux positifs
            et faux négatifs — pas de biais vers
            une classe.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Classification Report — RF + TF-IDF")
    report = classification_report(
        y_valid, y_pred_rf,
        target_names=["fake","true"],
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose().round(2)
    st.dataframe(df_report, use_container_width=True)

# ============================================================
# PAGE DEMO LIVE
# ============================================================

elif page == "Demo Live":

    st.markdown("# Demo Live")
    st.markdown(
        "<p style='color:#64748b;'>Testez notre modèle sur une déclaration politique</p>",
        unsafe_allow_html=True
    )
    st.divider()

    col1, col2 = st.columns([2,1])

    with col1:
        exemples = [
            "Sélectionner un exemple...",
            "We send the EU £350 million a week.",
            "The economy has created 10 million jobs.",
            "Climate change is a hoax invented by scientists.",
            "Unemployment is at its lowest level in 50 years."
        ]
        exemple = st.selectbox("Choisir un exemple :", exemples)
        texte = st.text_area(
            "Déclaration à analyser :",
            value="" if exemple == "Sélectionner un exemple..." else exemple,
            height=120,
            placeholder="Ex: The unemployment rate is at a record low..."
        )

        if st.button("Analyser la déclaration"):
            if texte.strip() == "":
                st.warning("Veuillez entrer une déclaration.")
            else:
                clean    = preprocess_text(texte)
                vec      = tfidf.transform([clean])
                pred_lr  = lr.predict(vec)[0]
                pred_rf  = rf.predict(vec)[0]
                proba_lr = lr.predict_proba(vec)[0]
                proba_rf = rf.predict_proba(vec)[0]
                classes  = list(lr.classes_)
                fi, ti   = classes.index("fake"), classes.index("true")

                st.divider()
                st.markdown("### Résultats")
                ca, cb = st.columns(2)

                for col, pred, proba, nom in [
                    (ca, pred_lr, proba_lr, "Logistic Regression"),
                    (cb, pred_rf, proba_rf, "Random Forest")
                ]:
                    with col:
                        tag   = "tag-fake" if pred == "fake" else "tag-true"
                        label = "FAKE" if pred == "fake" else "TRUE"
                        st.markdown(f"""
                        <div class='card'>
                            <h3>{nom}</h3>
                            <p style='margin-bottom:10px;'>Prédiction :</p>
                            <span class='{tag}'>{label}</span>
                            <p style='margin-top:16px;'>
                                Confiance FAKE :
                                <b style='color:#ef4444;'>{proba[fi]*100:.1f}%</b><br>
                                Confiance TRUE :
                                <b style='color:#22c55e;'>{proba[ti]*100:.1f}%</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h3>Comment ça marche ?</h3>
            <p>1. Le texte est nettoyé et tokenisé</p>
            <p>2. TF-IDF transforme le texte en vecteur</p>
            <p>3. Les modèles prédisent FAKE ou TRUE</p>
            <p>4. La confiance indique la certitude</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='card-interpretation'>
            <h4>Attention</h4>
            <p>Modèle entraîné sur des déclarations
            politiques américaines. Les performances
            peuvent varier sur d'autres types de textes.</p>
        </div>
        """, unsafe_allow_html=True)