# ============================================================
# TRUTHSCOPE — DASHBOARD STREAMLIT v2
# Notebooks 01-07 + WELFake external data
# Design: dark premium, SVG icons (no emojis)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re, os
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="TruthScope — NLP Dashboard",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#128269;</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ICON HELPERS  (Tabler Icons — inline SVG)
# ============================================================
def icon(name, size=18, color="currentColor"):
    """Return an inline SVG icon string."""
    icons = {
        "home": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "bar-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>',
        "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.07-4.78 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 2.55-2.44Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.07-4.78 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2Z"/></svg>',
        "scale": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21H17"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
        "globe": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "target": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "database": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "search": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "x-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "alert-triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "info": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "trending-up": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "users": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "tag": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
        "file-text": '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
    }
    svg = icons.get(name, icons["info"])
    return svg.replace("{s}", str(size)).replace("{c}", color)

def ico_html(name, size=18, color="#818cf8"):
    return f'<span style="display:inline-flex;align-items:center;vertical-align:middle;">{icon(name, size, color)}</span>'

# ============================================================
# CSS GLOBAL — DARK PREMIUM
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  * { box-sizing: border-box; }
  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  .stApp { background: #080812 !important; color: #e2e8f0 !important; }
  .stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 25%, #06b6d4 50%, #6366f1 75%, #8b5cf6 100%);
    background-size: 300%; animation: shimmer 4s linear infinite; z-index: 9999;
  }
  @keyframes shimmer { 0%{background-position:300%} 100%{background-position:-300%} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .main { animation: fadeIn 0.35s ease; }
  .main .block-container { background: #080812 !important; padding-top:1.5rem !important; max-width:1400px; }
  h1 { color:#fff !important; font-weight:800 !important; letter-spacing:-0.02em !important; }
  h2 { color:#f1f5f9 !important; font-weight:700 !important; }
  h3 { color:#e2e8f0 !important; font-weight:600 !important; }
  p  { color:#94a3b8 !important; }
  label, .stMarkdown p { color:#94a3b8 !important; }

  /* ---- Sidebar ---- */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0e0e1c 0%,#080812 100%) !important;
    border-right: 1px solid #1a1a2e !important; min-width:230px !important;
  }
  [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size:13px !important; font-weight:500 !important; }
  [data-testid="stSidebar"] hr { border-color:#1a1a2e !important; }

  /* ---- KPI Card ---- */
  .kpi-card {
    background: linear-gradient(135deg,#111124 0%,#18182e 100%);
    border: 1px solid #1e1e3a; border-radius:16px;
    padding:18px 20px; position:relative; overflow:hidden;
    transition: transform .25s, box-shadow .25s, border-color .25s;
  }
  .kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--accent, linear-gradient(90deg,#6366f1,#8b5cf6));
    border-radius:16px 16px 0 0;
  }
  .kpi-card:hover { transform:translateY(-3px); box-shadow:0 16px 36px rgba(99,102,241,.2); border-color:#6366f1; }
  .kpi-icon { margin-bottom:10px; display:block; }
  .kpi-value { font-size:26px; font-weight:800; color:#fff; display:block; line-height:1.1; font-family:'Inter',sans-serif; }
  .kpi-label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-top:5px; display:block; }
  .kpi-sub   { font-size:11px; color:#6366f1; margin-top:4px; display:block; font-weight:600; }

  /* ---- Generic Card ---- */
  .card {
    background: linear-gradient(135deg,#111124 0%,#16162a 100%);
    border:1px solid #1e1e3a; border-radius:14px;
    padding:20px 22px; margin:6px 0;
    transition: transform .2s, box-shadow .2s, border-color .2s;
  }
  .card:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(99,102,241,.1); border-color:#33336a; }
  .card h3 { color:#fff !important; margin-top:0 !important; font-size:15px !important; }
  .card p  { color:#94a3b8 !important; font-size:13px !important; line-height:1.75 !important; margin:2px 0 !important; }

  /* ---- Badge ---- */
  .badge {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(99,102,241,.12); color:#818cf8 !important;
    border:1px solid rgba(99,102,241,.25); padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:700; margin-bottom:10px; text-transform:uppercase; letter-spacing:.07em;
  }
  .badge-green { background:rgba(34,197,94,.12); color:#86efac !important; border-color:rgba(34,197,94,.25); }
  .badge-red   { background:rgba(239,68,68,.12);  color:#fca5a5 !important; border-color:rgba(239,68,68,.25); }
  .badge-amber { background:rgba(245,158,11,.12); color:#fcd34d !important; border-color:rgba(245,158,11,.25); }
  .badge-cyan  { background:rgba(6,182,212,.12);  color:#67e8f9 !important; border-color:rgba(6,182,212,.25); }

  /* ---- Alert blocks ---- */
  .alert-info {
    background:rgba(6,182,212,.07); border:1px solid rgba(6,182,212,.2);
    border-left:3px solid #06b6d4; border-radius:10px; padding:13px 15px; margin:10px 0;
  }
  .alert-info p { color:#7dd3fc !important; font-size:13px !important; margin:0 !important; }
  .alert-warn {
    background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.2);
    border-left:3px solid #f59e0b; border-radius:10px; padding:13px 15px; margin:10px 0;
  }
  .alert-warn p { color:#fcd34d !important; font-size:13px !important; margin:0 !important; }
  .alert-success {
    background:rgba(34,197,94,.07); border:1px solid rgba(34,197,94,.2);
    border-left:3px solid #22c55e; border-radius:10px; padding:13px 15px; margin:10px 0;
  }
  .alert-success p { color:#86efac !important; font-size:13px !important; margin:0 !important; }

  /* ---- Insight box ---- */
  .insight {
    background:#0c0c1e; border:1px solid #1e1e3a; border-left:3px solid #6366f1;
    border-radius:10px; padding:14px 16px; margin:8px 0;
  }
  .insight h4 {
    color:#818cf8 !important; margin:0 0 6px 0 !important;
    font-size:10px !important; text-transform:uppercase; letter-spacing:.08em; font-weight:700 !important;
  }
  .insight p { color:#94a3b8 !important; font-size:13px !important; line-height:1.75 !important; margin:0 !important; }

  /* ---- Section header ---- */
  .section-title {
    display:flex; align-items:center; gap:10px;
    border-bottom:1px solid #1a1a2e; padding-bottom:12px; margin-bottom:20px;
  }
  .section-title h2 { margin:0 !important; }

  /* ---- Verdict tags ---- */
  .tag-fake { background:rgba(239,68,68,.12); color:#fca5a5 !important; padding:10px 26px; border-radius:24px; font-weight:700; font-size:20px; display:inline-block; border:1px solid rgba(239,68,68,.3); font-family:'Inter',sans-serif; }
  .tag-real { background:rgba(34,197,94,.12);  color:#86efac !important; padding:10px 26px; border-radius:24px; font-weight:700; font-size:20px; display:inline-block; border:1px solid rgba(34,197,94,.3);  font-family:'Inter',sans-serif; }

  /* ---- Probability bar ---- */
  .prob-wrap { background:#1a1a2e; border-radius:6px; height:8px; margin:5px 0; overflow:hidden; }
  .prob-fill  { height:100%; border-radius:6px; transition:width .6s ease; }

  /* ---- Buttons ---- */
  .stButton button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:#ffffff !important; border:none !important; border-radius:10px !important;
    padding:10px 24px !important; font-weight:800 !important; font-family:'Inter',sans-serif !important;
    font-size:14px !important; letter-spacing:0.01em !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    transition:all .2s !important;
  }
  .stButton button p { color:#ffffff !important; font-weight:800 !important; }
  .stButton button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 18px rgba(99,102,241,.45) !important; }
  /* Force button text white in all Streamlit versions */
  button[kind="primary"], button[kind="secondary"] { color:#ffffff !important; }
  div[data-testid="stButton"] button span { color:#ffffff !important; font-weight:800 !important; }

  /* ---- Inputs ---- */
  .stSelectbox>div>div, .stMultiSelect>div>div {
    background:#111124 !important; color:#f1f5f9 !important;
    border:1px solid #1e1e3a !important; border-radius:8px !important;
  }
  .stTextArea textarea {
    background:#111124 !important; color:#f1f5f9 !important;
    border:1px solid #1e1e3a !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important;
  }
  .stTextArea textarea:focus { border-color:#6366f1 !important; box-shadow:0 0 0 2px rgba(99,102,241,.2) !important; }
  .stTextInput input {
    background:#111124 !important; color:#f1f5f9 !important;
    border:1px solid #1e1e3a !important; border-radius:8px !important;
  }

  /* ---- Dataframe ---- */
  .stDataFrame { border-radius:12px !important; overflow:hidden !important; }

  /* ---- Tabs ---- */
  .stTabs [data-baseweb="tab-list"] { background:#111124; border-radius:10px; padding:4px; border:1px solid #1e1e3a; }
  .stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border-radius:8px !important; color:#ffffff !important;
    font-weight:700 !important;
  }
  .stTabs [aria-selected="true"] p,
  .stTabs [aria-selected="true"] span,
  .stTabs [aria-selected="true"] div {
    color:#ffffff !important; font-weight:700 !important;
  }
  .stTabs [data-baseweb="tab"] { color:#94a3b8 !important; font-weight:500 !important; border-radius:8px !important; }
  .stTabs [data-baseweb="tab"]:hover { color:#e2e8f0 !important; }

  /* ---- Selectbox selected text ---- */
  .stSelectbox [data-baseweb="select"] div { color:#f1f5f9 !important; }
  div[data-baseweb="popover"] li { color:#e2e8f0 !important; }
  div[data-baseweb="popover"] li[aria-selected="true"] {
    background:#1a1a3e !important; color:#818cf8 !important; font-weight:700 !important;
  }
  div[data-baseweb="popover"] li:hover { background:#1e1e3a !important; color:#fff !important; }

  /* ---- Radio button items (Navigation) ---- */
  .stRadio > div[role="radiogroup"] > label {
    padding: 10px 14px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-bottom: 4px !important;
    border-left: 3px solid transparent !important;
  }
  .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(99,102,241, 0.1) !important;
    transform: translateX(3px) !important;
  }
  .stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(99,102,241, 0.15), rgba(139,92,246, 0.15)) !important;
    border-left: 3px solid #818cf8 !important;
  }
  .stRadio label:has(input:checked) span { color:#818cf8 !important; font-weight:700 !important; }

  hr { border-color:#1a1a2e !important; margin:1.5rem 0 !important; }
  code { font-family:'JetBrains Mono',monospace !important; background:#111124 !important; color:#818cf8 !important; padding:2px 6px !important; border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES
# ============================================================
COLONNES = [
    "id","label","statement","subject","speaker","job_title",
    "state","party","barely_true_counts","false_counts",
    "half_true_counts","mostly_true_counts","pants_on_fire_counts","context"
]
ORDRE_6   = ["pants-fire","false","barely-true","half-true","mostly-true","true"]
FAKE_L    = {"pants-fire","false","barely-true"}
STOPWORDS = set(stopwords.words("english"))
C6 = ["#ef4444","#f97316","#fbbf24","#facc15","#84cc16","#22c55e"]
BG = "#111124"; GRID = "#1e1e3a"
ACCENT="#6366f1"; A2="#8b5cf6"; A3="#06b6d4"

def dark_fig(fig, title="", h=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=14)) if title else None,
        height=h, plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(t=50 if title else 30, b=30, l=30, r=20),
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            font_color="#ffffff",
            bordercolor="#6366f1",
            font_size=12,
        ),
    )
    for ax in ("xaxis","yaxis","xaxis2","yaxis2","xaxis3","yaxis3"):
        try:
            fig.update_layout(**{ax: dict(gridcolor=GRID, tickfont=dict(color="#94a3b8"),
                                          linecolor=GRID, title_font=dict(color="#94a3b8"))})
        except Exception:
            pass
    return fig

def preprocess_text(text):
    text   = str(text).lower().replace("-"," ")
    text   = re.sub(r"[^a-z\s]","",text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t)>1]
    return " ".join(tokens)

# ============================================================
# DATA LOADING
# ============================================================
DATA_PATHS = [
    ("data/raw/train.tsv", "data/raw/valid.tsv", "data/raw/test.tsv"),
    ("data/train.tsv",     "data/valid.tsv",     "data/test.tsv"),
    (r"C:\Users\gerau\Documents\truthscope\kof_project\data\train.tsv",
     r"C:\Users\gerau\Documents\truthscope\kof_project\data\valid.tsv",
     r"C:\Users\gerau\Documents\truthscope\kof_project\data\test.tsv"),
]
WELFAKE_PATHS = [
    "data/external/WELFake_sample.csv",
    r"C:\Users\gerau\Downloads\Pro\data\external\WELFake_sample.csv",
]

@st.cache_data(show_spinner=False)
def load_data():
    for tr, va, te in DATA_PATHS:
        try:
            train = pd.read_csv(tr, sep="\t", header=None, names=COLONNES)
            valid = pd.read_csv(va, sep="\t", header=None, names=COLONNES)
            test  = pd.read_csv(te, sep="\t", header=None, names=COLONNES)
            break
        except FileNotFoundError:
            continue
    else:
        return None, None, None
    for df in [train, valid, test]:
        df["context"] = df["context"].fillna("")
        df["speaker"] = df["speaker"].fillna("unknown")
        df["party"]   = df["party"].fillna("unknown")
        df["subject"] = df["subject"].fillna("")
        df["nb_mots"] = df["statement"].astype(str).str.split().str.len()
        df["label_binary"] = df["label"].apply(lambda x: "fake" if x in FAKE_L else "real")
        df["label_3class"] = df["label"].apply(
            lambda x: "fake" if x in ["pants-fire","false","barely-true"]
            else ("nuanced" if x=="half-true" else "real"))
        total = (df["barely_true_counts"].fillna(0)+df["false_counts"].fillna(0)+
                 df["half_true_counts"].fillna(0)+df["mostly_true_counts"].fillna(0)+
                 df["pants_on_fire_counts"].fillna(0))
        df["credibility_score"] = df["mostly_true_counts"].fillna(0)/(total+1)
        df["lie_rate"] = (df["barely_true_counts"].fillna(0)+df["false_counts"].fillna(0)+
                          df["pants_on_fire_counts"].fillna(0))/(total+1)
    return train, valid, test

@st.cache_resource(show_spinner=False)
def train_models(_train, _valid):
    train_clean = _train["statement"].apply(preprocess_text)
    valid_clean = _valid["statement"].apply(preprocess_text)
    tfidf   = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2, sublinear_tf=True)
    X_tr    = tfidf.fit_transform(train_clean)
    X_va    = tfidf.transform(valid_clean)
    y_tr    = (_train["label_binary"]=="fake").astype(int)
    y_va    = (_valid["label_binary"]=="fake").astype(int)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    lr.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return tfidf, lr, rf, X_va, y_va

@st.cache_data(show_spinner=False)
def load_welfake_sample(n=500):
    for p in WELFAKE_PATHS:
        if os.path.exists(p):
            cols_needed = ["title","text","label"]
            try:
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed+["Unnamed: 0"], nrows=5000)
                df.columns = [c.lower().strip() for c in df.columns]
                if "text" not in df.columns and "title" in df.columns:
                    df["text"] = df["title"]
                df = df.dropna(subset=["text","label"])
                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
                df = df.sample(min(n, len(df)), random_state=42)
                return df
            except Exception:
                continue
    return None

train_df, valid_df, test_df = load_data()
data_ok = train_df is not None

if data_ok:
    with st.spinner("Chargement des modèles en mémoire…"):
        tfidf, lr_model, rf_model, X_va, y_va = train_models(train_df, valid_df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0 8px;'>
      <h1 style='color:#865DF5;font-size:22px;font-weight:800;letter-spacing:-.02em;margin:8px 0 2px;'>TruthScope</h1>
      <p style='color:#4a5568;font-size:11px;margin:0;'>Fake News Detection · NLP</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    PAGES = {
        "⌂  Accueil": "Accueil",
        "◧  Exploration EDA": "Exploration EDA",
        "⚙  Preprocessing": "Preprocessing",
        "⎔  Modèles Classiques": "Modèles Classiques",
        "⌬  Features Améliorées": "Features Améliorées",
        "❖  BERT": "BERT",
        "⚖  Binaire vs Multi-classe": "Binaire vs Multi-classe",
        "◍  Éval. Out-of-Domain": "Éval. Out-of-Domain",
        "▶  Demo Live": "Demo Live"
    }

    selected_label = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[selected_label]

    st.divider()
    st.markdown("""
    <div style='padding:0 4px;'>
      <p style='color:#6366f1;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;'>Équipe Projet</p>
      <p style='color:#818cf8;font-size:12px;line-height:2;margin:0;'>
        Melvyn CHARLES<br>Koffi ADJAÏ<br>
        FANAINGAMAMPIANDRA S. Jo<br>
        RAMANDIMBISOA Heliaritiana<br>
        Antoine BENECH
      </p>
      <p style='color:#2d2d4a;font-size:11px;margin-top:14px;text-align:center;'>Epitech Digital — 2026</p>
    </div>
    """, unsafe_allow_html=True)
    if not data_ok:
        st.warning("Données LIAR introuvables dans `data/raw/`.")

# ===========================================================================
# PAGE : ACCUEIL
# ===========================================================================
import base64
def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

if page == "Accueil":
    # Hero image
    hero_b64 = img_to_b64("assets/hero.png")
    if hero_b64:
        st.markdown(f"""
        <div style='
          background:url("data:image/png;base64,{hero_b64}") center/cover no-repeat,
                    linear-gradient(135deg,#0f0f1e 0%,#18183a 50%,#0f0f1e 100%);
          border:1px solid #1e1e3a; border-radius:20px; padding:48px;
          margin-bottom:28px; text-align:center; position:relative; overflow:hidden; min-height:280px;
        '>
          <div style='
            position:absolute;inset:0;border-radius:20px;
            background:linear-gradient(180deg, rgba(8,8,18,.55) 0%, rgba(8,8,18,.85) 100%);
          '></div>
         
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0f0f1e 0%,#18183a 50%,#0f0f1e 100%);
                    border:1px solid #1e1e3a;border-radius:20px;padding:48px;margin-bottom:28px;text-align:center;'>
          <span class='badge' style='margin-bottom:20px;'>Epitech Digital School · 2026</span>
          <h1 style='font-size:48px!important;margin:0;background:linear-gradient(135deg,#818cf8,#06b6d4);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>TruthScope</h1>
          <p style='font-size:17px!important;color:#94a3b8!important;margin-top:10px;max-width:560px;margin-left:auto;margin-right:auto;'>
            Détection automatique de fake news politiques par NLP — LIAR Dataset
          </p>
        </div>
        """, unsafe_allow_html=True)

    kpis = [
        ("database", "10 240", "Déclarations Train", "LIAR Dataset — PolitiFact"),
        ("tag",      "6",      "Labels de véracité", "pants-fire → true"),
        ("cpu",      "3+",     "Modèles entraînés",  "LR · RF · XGBoost · BERT"),
        ("trending-up","78.7%","Meilleur F1 macro",  "LR Binaire (in-domain)"),
        ("globe",    "72 134", "Articles WELFake",   "Éval. out-of-domain"),
    ]
    cols = st.columns(5)
    for col, (ico, val, lbl, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <span class='kpi-icon'>{ico_html(ico, 22, '#6366f1')}</span>
              <span class='kpi-value'>{val}</span>
              <span class='kpi-label'>{lbl}</span>
              <span class='kpi-sub'>{sub}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
          <span class='badge'>{ico_html('info', 12)} À propos</span>
          <h3>Projet TruthScope</h3>
          <p>TruthScope est un pipeline complet de détection automatique de fake news politiques
          basé sur le <b style='color:#818cf8;'>LIAR Dataset</b> de PolitiFact (Wang, ACL 2017).</p>
          <p>7 notebooks couvrent l'exploration, le preprocessing, la modélisation classique,
          les features avancées, BERT, la comparaison binaire/multiclasse et l'évaluation OOD sur WELFake.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card'>
          <span class='badge'>{ico_html('database', 12)} Dataset LIAR</span>
          <h3>LIAR Dataset</h3>
          <p>• <b style='color:#818cf8;'>12 836</b> déclarations politiques annotées par PolitiFact</p>
          <p>• <b style='color:#818cf8;'>6 labels</b> : pants-fire · false · barely-true · half-true · mostly-true · true</p>
          <p>• <b style='color:#818cf8;'>Splits</b> : train 10 240 · valid 1 284 · test 1 267</p>
          <p>• <b style='color:#818cf8;'>Métadonnées</b> : speaker, parti, crédibilité historique, état, contexte</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Comparaison globale des configurations")
    df_res = pd.DataFrame({
        "Configuration": ["3cls — LR","3cls — RF","3cls — XGBoost","2cls — LR","2cls — RF","2cls — XGBoost"],
        "Accuracy": [0.5375,0.5651,0.5556,0.7899,0.7873,0.7747],
        "F1 Macro": [0.5442,0.5697,0.5577,0.7867,0.7837,0.7723],
        "Type": ["3 classes","3 classes","3 classes","2 classes","2 classes","2 classes"],
    })
    fig = px.bar(df_res, x="Configuration", y="F1 Macro", color="Type",
                 color_discrete_map={"3 classes":"#5b7fa6","2 classes":"#22c55e"},
                 text="F1 Macro", barmode="group")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                      textfont_color="#fff", marker_line_width=0)
    fig = dark_fig(fig, "F1 Macro par configuration (test set)", 400)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE 01 — EXPLORATION EDA  (COMPLÈTE)
# ===========================================================================
elif page == "Exploration EDA":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('bar-chart', 24, '#6366f1')}
      <div><span class='badge'>Notebook 01</span><h2 style='margin:0;'>Exploration & Analyse des Données</h2></div>
    </div>""", unsafe_allow_html=True)

    if not data_ok:
        st.error("Données non trouvées. Vérifiez `data/raw/train.tsv`.")
        st.stop()

    # ---- KPIs ----
    kpis_eda = [
        ("file-text", f"{len(train_df):,}", "Déclarations Train", "14 colonnes"),
        ("tag",       "6", "Labels uniques", "Ordinal pants-fire→true"),
        ("users",     str(train_df['speaker'].nunique()), "Speakers", "obama ×488"),
        ("globe",     str(train_df['party'].nunique()), "Partis", "republican, democrat…"),
        ("file-text", f"{train_df['nb_mots'].mean():.1f}", "Mots moy./statement", "Très court"),
    ]
    cols = st.columns(5)
    for col, (ico, val, lbl, sub) in zip(cols, kpis_eda):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <span class='kpi-icon'>{ico_html(ico, 20, '#06b6d4')}</span>
              <span class='kpi-value'>{val}</span>
              <span class='kpi-label'>{lbl}</span>
              <span class='kpi-sub'>{sub}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- Aperçu des données brutes ----
    with st.expander("Aperçu des 5 premières lignes (train set)", expanded=False):
        st.dataframe(train_df[["label","statement","speaker","party","subject",
                                "barely_true_counts","false_counts","context"]].head(5),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 1 — Distribution des 6 Labels de Véracité")
    counts = train_df["label"].value_counts().reindex(ORDRE_6)
    col1, col2 = st.columns([3,2])
    with col1:
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=C6, text=counts.values,
            textposition="outside", textfont=dict(color="#fff"),
        ))
        fig = dark_fig(fig, "Nombre de déclarations par label (train)", 380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Pie(
            values=counts.values, labels=counts.index, hole=0.52,
            marker_colors=C6,
            marker=dict(line=dict(color="#080812",width=2)),
            textposition="outside", textinfo="percent",
        ))
        fig2 = dark_fig(fig2, "Répartition en %", 380)
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Les 6 classes sont relativement équilibrées (≈16-20%).
      <b style='color:#ef4444;'>pants-fire</b> est le moins représenté (~8%).
      Les 3 labels faux (pants-fire, false, barely-true) totalisent ~44% du corpus.
      → Utilisation de <code>class_weight='balanced'</code> pour corriger le déséquilibre.</p>
    </div>""", unsafe_allow_html=True)

    # Label par split
    st.divider()
    st.markdown("### 2 — Distribution des Labels par Split")
    all_splits = []
    for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        c = df["label"].value_counts().reindex(ORDRE_6).fillna(0)
        for lbl, cnt in c.items():
            all_splits.append({"Split": name, "Label": lbl, "Count": int(cnt)})
    df_sp = pd.DataFrame(all_splits)
    fig_sp = px.bar(df_sp, x="Label", y="Count", color="Split", barmode="group",
                    color_discrete_map={"Train":ACCENT,"Valid":A3,"Test":"#22c55e"},
                    category_orders={"Label":ORDRE_6},
                    text="Count")
    fig_sp.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
    fig_sp = dark_fig(fig_sp, "Fréquence des labels — Train / Valid / Test", 400)
    st.plotly_chart(fig_sp, use_container_width=True)
    col1,col2,col3 = st.columns(3)
    for col, (nm, df) in zip([col1,col2,col3], [("Train",train_df),("Valid",valid_df),("Test",test_df)]):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <span class='kpi-value'>{len(df):,}</span>
              <span class='kpi-label'>{nm} set</span>
              <span class='kpi-sub'>{len(df)/len(train_df)+len(valid_df)+len(test_df)*0:.0%} du total</span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 3 — Top 15 Speakers")
    top_spk = train_df["speaker"].value_counts().head(15).reset_index()
    top_spk.columns = ["speaker","count"]
    col1, col2 = st.columns([2,1])
    with col1:
        fig_spk = px.bar(top_spk, x="count", y="speaker", orientation="h",
                         color="count", color_continuous_scale=[[0,"#1e1e3a"],[1,ACCENT]],
                         text="count")
        fig_spk.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_spk = dark_fig(fig_spk, "Top 15 speakers par fréquence", 420)
        fig_spk.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_spk, use_container_width=True)
    with col2:
        fig_pf = px.pie(top_spk.head(6), values="count", names="speaker",
                        hole=0.5, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_pf.update_traces(textposition="outside", textinfo="label+percent",
                              marker=dict(line=dict(color="#080812",width=2)))
        fig_pf = dark_fig(fig_pf, "Top 6 speakers", 420)
        st.plotly_chart(fig_pf, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Barack Obama domine avec <b style='color:#818cf8;'>488 déclarations</b> — risque de biais.<br>
      "chain-email" et "blog-posting" = sources anonymes avec taux de fake élevé.<br>
      → Le speaker est une feature puissante mais non généralisable (absent de WELFake).</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 4 — Distribution par Parti Politique")
    top_par = train_df["party"].value_counts().head(8).reset_index()
    top_par.columns = ["party","count"]
    col1, col2 = st.columns([2,1])
    with col1:
        fig_par = px.bar(top_par, x="party", y="count", color="count",
                         color_continuous_scale=[[0,"#1e1e3a"],[1,A2]],
                         text="count")
        fig_par.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_par = dark_fig(fig_par, "Déclarations par parti", 380)
        fig_par.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_par, use_container_width=True)
    with col2:
        fig_tree = px.treemap(top_par, path=["party"], values="count", color="count",
                              color_continuous_scale=[[0,"#1a1a2e"],[0.4,"#312e81"],[1,ACCENT]])
        fig_tree.update_traces(textfont=dict(color="#fff",size=13),
                                marker=dict(line=dict(width=2,color="#080812")))
        fig_tree = dark_fig(fig_tree, "", 380)
        fig_tree.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
    st.markdown("### 5 — Véracité par Parti (top 5)")
    top5p = train_df["party"].value_counts().head(5).index
    df_vp = (train_df[train_df["party"].isin(top5p)]
             .groupby(["party","label"]).size().reset_index(name="count"))
    col1, col2 = st.columns([2,1])
    with col1:
        fig_vp = px.bar(df_vp, x="party", y="count", color="label", barmode="stack",
                        category_orders={"label":ORDRE_6},
                        color_discrete_sequence=C6)
        fig_vp.update_traces(marker_line_width=0)
        fig_vp = dark_fig(fig_vp, "Distribution des labels par parti", 400)
        st.plotly_chart(fig_vp, use_container_width=True)
    with col2:
        # Taux fake par parti
        df_fake_rate = (train_df[train_df["party"].isin(top5p)]
                        .groupby("party")["label_binary"]
                        .apply(lambda x: (x=="fake").mean()).reset_index())
        df_fake_rate.columns = ["party","fake_rate"]
        df_fake_rate = df_fake_rate.sort_values("fake_rate",ascending=False)
        fig_fr = px.bar(df_fake_rate, x="fake_rate", y="party", orientation="h",
                        color="fake_rate", color_continuous_scale=[[0,"#1e1e3a"],[1,"#ef4444"]],
                        text="fake_rate")
        fig_fr.update_traces(texttemplate="%{text:.0%}", textposition="outside",
                              textfont_color="#fff", marker_line_width=0)
        fig_fr = dark_fig(fig_fr, "Taux de fake par parti", 400)
        fig_fr.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fr, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Les républicains affichent plus de <b style='color:#ef4444;'>pants-fire</b> et <b style='color:#f97316;'>false</b>.
      Cela peut refléter un biais de PolitiFact (plus de vérifications sur des sujets controversés républicains).<br>
      → À ne pas interpréter comme preuve directe que les républicains mentent plus.</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 6 — Longueur des Déclarations (nombre de mots)")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(train_df, x="nb_mots", nbins=60,
                                color_discrete_sequence=[ACCENT],
                                labels={"nb_mots":"Nombre de mots","count":"Fréquence"})
        fig_hist.add_vline(x=train_df["nb_mots"].mean(), line_dash="dash", line_color="#ef4444",
                           annotation_text=f"Moy. {train_df['nb_mots'].mean():.1f}",
                           annotation_font_color="#ef4444")
        fig_hist.add_vline(x=train_df["nb_mots"].median(), line_dash="dot", line_color="#fbbf24",
                           annotation_text=f"Med. {train_df['nb_mots'].median():.0f}",
                           annotation_font_color="#fbbf24")
        fig_hist = dark_fig(fig_hist, "Distribution de la longueur (train)", 360)
        fig_hist.update_traces(marker_line_width=0)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        moy_label = (train_df.groupby("label")["nb_mots"].mean()
                     .reindex(ORDRE_6).round(1).reset_index())
        moy_label.columns = ["label","moy"]
        fig_moy = px.bar(moy_label, x="label", y="moy", color="label",
                         color_discrete_sequence=C6, text="moy")
        fig_moy.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_moy = dark_fig(fig_moy, "Longueur moyenne par label", 360)
        fig_moy.update_layout(showlegend=False)
        st.plotly_chart(fig_moy, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(train_df, x="label", y="nb_mots", color="label",
                         category_orders={"label":ORDRE_6},
                         color_discrete_sequence=C6)
        fig_box = dark_fig(fig_box, "Box plot longueur par label", 340)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_vio = px.violin(train_df, x="label", y="nb_mots", color="label",
                            category_orders={"label":ORDRE_6},
                            color_discrete_sequence=C6, box=True)
        fig_vio = dark_fig(fig_vio, "Violin plot longueur par label", 340)
        fig_vio.update_layout(showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Statements très courts en moyenne (<b style='color:#818cf8;'>~18 mots</b>), médiane ~15 mots.
      Différences entre labels minimes (17–19 mots) — signal très faible.
      Présence d'outliers (>100 mots) : conservés car valides.<br>
      → La longueur seule ne suffit pas à discriminer fake/real.</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 7 — Top Sujets Politiques (subject)")
    all_subj = []
    for s in train_df["subject"].dropna():
        all_subj.extend([x.strip() for x in s.split(",") if x.strip()])
    # Compatible pandas 1.x et 2.x — construction explicite du DataFrame
    subj_counts_series = pd.Series(all_subj).value_counts().head(20)
    subj_df = pd.DataFrame({"subject": subj_counts_series.index, "count": subj_counts_series.values})
    col1, col2 = st.columns([2,1])
    with col1:
        fig_subj = px.bar(subj_df, x="count", y="subject", orientation="h",
                          color="count", color_continuous_scale=[[0,"#1e1e3a"],[1,A3]],
                          text="count")
        fig_subj.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_subj = dark_fig(fig_subj, "Top 20 sujets politiques", 520)
        fig_subj.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_subj, use_container_width=True)
    with col2:
        subj_top10 = subj_df.head(10)
        fig_subj_pie = px.pie(values=subj_top10["count"].values, names=subj_top10["subject"].values,
                              hole=0.45,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_subj_pie.update_traces(textposition="outside", textinfo="label+percent",
                                    marker=dict(line=dict(color="#080812",width=2)))
        fig_subj_pie = dark_fig(fig_subj_pie, "Top 10 sujets", 520)
        st.plotly_chart(fig_subj_pie, use_container_width=True)

    st.divider()
    st.markdown("### 8 — Analyse des Métadonnées de Crédibilité")
    col1, col2 = st.columns(2)
    with col1:
        fig_cred = px.histogram(train_df, x="credibility_score", nbins=50, color="label_binary",
                                color_discrete_map={"fake":"#ef4444","real":"#22c55e"},
                                barmode="overlay", opacity=0.7)
        fig_cred = dark_fig(fig_cred, "Distribution du credibility_score par classe binaire", 360)
        fig_cred.update_traces(marker_line_width=0)
        st.plotly_chart(fig_cred, use_container_width=True)
    with col2:
        fig_lr = px.histogram(train_df, x="lie_rate", nbins=50, color="label_binary",
                              color_discrete_map={"fake":"#ef4444","real":"#22c55e"},
                              barmode="overlay", opacity=0.7)
        fig_lr = dark_fig(fig_lr, "Distribution du lie_rate par classe binaire", 360)
        fig_lr.update_traces(marker_line_width=0)
        st.plotly_chart(fig_lr, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Le <b style='color:#818cf8;'>credibility_score</b> et le <b style='color:#818cf8;'>lie_rate</b>
      montrent une séparation nette entre fake et real — ce sont des features très discriminantes.
      Les speakers "fake" ont historiquement plus de mensonges passés.<br>
      → Attention : ces features sont NULLES pour WELFake → chute OOD de 45 pp.</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 9 — Valeurs Manquantes")
    miss_data = pd.DataFrame({
        "Colonne": ["statement","label","subject","speaker","party","context","job_title","state","barely_true_counts"],
        "Train (%)" : [0.00, 0.00, 0.02, 0.02, 0.02, 1.00, 28.30, 21.58, 0.02],
        "Valid (%)": [0.00, 0.00, 0.00, 0.00, 0.00, 0.93, 26.87, 21.73, 0.00],
        "Test (%)": [0.00, 0.00, 0.00, 0.00, 0.00, 1.34, 25.65, 20.68, 0.00],
        "Action": ["—", "—", "Rempl. ''", "Rempl. unknown", "Rempl. unknown",
                   "Rempl. ''", "Ignorée (non utilisée)", "Ignorée (non utilisée)", "Rempl. 0"],
    })
    col1, col2 = st.columns([3,1])
    with col1:
        fig_miss = go.Figure()
        for col_name, color in [("Train (%)",ACCENT),("Valid (%)",A3),("Test (%)","#22c55e")]:
            fig_miss.add_trace(go.Bar(name=col_name, x=miss_data["Colonne"], y=miss_data[col_name],
                                      marker_color=color, marker_line_width=0))
        fig_miss.update_layout(barmode="group")
        fig_miss = dark_fig(fig_miss, "Taux de valeurs manquantes par colonne (%)", 340)
        st.plotly_chart(fig_miss, use_container_width=True)
    with col2:
        st.dataframe(miss_data[["Colonne","Train (%)","Action"]],
                     use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='alert-info'><p>
    <b>statement</b> et <b>label</b> : 0% manquants — les colonnes critiques sont intactes.<br>
    <b>job_title</b> et <b>state</b> : ~25% manquants — ignorées (non utilisées dans le pipeline).<br>
    Règle : on ne supprime une ligne que si elle est inutilisable.
    </p></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 10 — Contexte des Déclarations")
    top_ctx = train_df["context"].value_counts().head(15).reset_index()
    top_ctx.columns = ["context","count"]
    top_ctx = top_ctx[top_ctx["context"]!=""]
    fig_ctx = px.bar(top_ctx, x="count", y="context", orientation="h",
                     color="count", color_continuous_scale=[[0,"#1e1e3a"],[1,A2]],
                     text="count")
    fig_ctx.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
    fig_ctx = dark_fig(fig_ctx, "Top 15 contextes de déclarations", 420)
    fig_ctx.update_layout(yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig_ctx, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Les interviews TV (CNN, Fox), les débats et les mailers sont les contextes les plus fréquents.
      Les chaînes emails viraux ("a chain email") sont presque exclusivement faux.
      → Le contexte est une feature utile pour les déclarations LIAR mais inutilisable en OOD.</p>
    </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 02 — PREPROCESSING
# ===========================================================================
elif page == "Preprocessing":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('settings', 24, '#6366f1')}
      <div><span class='badge'>Notebook 02</span><h2 style='margin:0;'>Preprocessing & Feature Engineering</h2></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pipeline de Nettoyage Textuel")
        steps = [
            ("1","lowercase()","Mise en minuscules de tout le texte"),
            ("2","replace('-',' ')","Tirets convertis en espaces"),
            ("3","re.sub(r'[^a-z\\s]','')","Suppression ponctuation, chiffres, accents"),
            ("4","split()","Tokenisation sur les espaces"),
            ("5","stopwords (NLTK)","Suppression des mots vides anglais"),
            ("6","len(t) > 1","Suppression tokens d'un seul caractère"),
        ]
        for n, code, desc in steps:
            st.markdown(f"""
            <div class='card' style='padding:10px 14px;margin:3px 0;'>
              <div style='display:flex;align-items:center;gap:10px;'>
                <span style='color:#6366f1;font-weight:800;font-size:14px;min-width:20px;'>{n}</span>
                <code style='font-size:11px;'>{code}</code>
                <span style='color:#94a3b8;font-size:12px;'>{desc}</span>
              </div>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("### Encodage des Labels")
        st.markdown(f"""
        <div class='card'>
          <span class='badge'>{ico_html('tag', 12)} 3 Classes</span>
          <h3>Multi-classe</h3>
          <p>• <b style='color:#ef4444;'>fake (0)</b> : pants-fire · false · barely-true</p>
          <p>• <b style='color:#fbbf24;'>nuanced (1)</b> : half-true</p>
          <p>• <b style='color:#22c55e;'>real (2)</b> : mostly-true · true</p>
        </div>
        <div class='card' style='margin-top:10px;'>
          <span class='badge badge-green'>{ico_html('tag', 12, '#86efac')} 2 Classes</span>
          <h3>Binaire (NB 06 & 07)</h3>
          <p>• <b style='color:#ef4444;'>fake (0)</b> : pants-fire · false · barely-true</p>
          <p>• <b style='color:#22c55e;'>real (1)</b> : mostly-true · true</p>
          <p>• <b style='color:#fbbf24;'>"nuanced" filtrée</b> → 10 240 → 6 472 exemples</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Features Métadonnées Speaker")
    meta_f = [
        ("credibility_score","Score de crédibilité historique",
         "mostly_true_counts / (total + 1)","[0, 1]","0.5 si speaker inconnu"),
        ("lie_rate","Taux de mensonge historique",
         "(false+barely_true+pants_fire) / (total+1)","[0, 1]","Source : profils PolitiFact"),
        ("history_total","Total de vérifications PolitiFact",
         "Somme de tous les counts","Entier ≥ 0","Mesure de notoriété du speaker"),
        ("is_politician","Indicateur élu politique",
         "1 si job_title contient senator/representative…","Binaire {0,1}","Heuristique sur job_title"),
    ]
    cols = st.columns(2)
    for i, (nm, desc, formula, dtype, note) in enumerate(meta_f):
        with cols[i%2]:
            st.markdown(f"""
            <div class='card'>
              <span class='badge badge-cyan'>{ico_html('database', 12, '#67e8f9')} Meta</span>
              <h3>{nm}</h3>
              <p>{desc}</p>
              <p>Formule : <code>{formula}</code></p>
              <p>Type : <b style='color:#818cf8;'>{dtype}</b></p>
              <p style='color:#4a5568!important;font-size:11px!important;'>Note : {note}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### TF-IDF — Paramètres retenus")
    cols = st.columns(3)
    for col, (v, lbl, sub) in zip(cols, [
        ("9 049","Features TF-IDF (baseline)","Après filtre min_df=3"),
        ("(1, 2)","Ngram range","Unigrammes + bigrammes"),
        ("9 053","Features totales (baseline)","9049 TF-IDF + 4 méta"),
    ]):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <span class='kpi-value'>{v}</span>
              <span class='kpi-label'>{lbl}</span>
              <span class='kpi-sub'>{sub}</span>
            </div>""", unsafe_allow_html=True)

    if data_ok:
        st.divider()
        st.markdown("### Demo — Preprocessing Interactif")
        sample = st.text_area("Texte à préprocesser :",
            value="Barack Obama said that 95% of Americans have health insurance — this is misleading!",
            height=80)
        if sample:
            res = preprocess_text(sample)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original :**")
                st.code(sample, language="text")
            with col2:
                st.markdown("**Après preprocessing :**")
                st.code(res, language="text")
            origw = len(sample.split())
            resw = len(res.split())
            st.markdown(f"""
            <div class='alert-success'><p>
            Réduction : <b>{origw} mots → {resw} tokens</b>
            ({(1-resw/max(origw,1))*100:.0f}% de réduction par suppression des stopwords et ponctuation)
            </p></div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 03 — MODÈLES CLASSIQUES
# ===========================================================================
elif page == "Modèles Classiques":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('cpu', 24, '#6366f1')}
      <div><span class='badge'>Notebook 03</span><h2 style='margin:0;'>Modèles Classiques — Résultats Complets</h2></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class='alert-info'><p>
    LR · RF · XGBoost entraînés avec TF-IDF (1-2grammes, min_df=3) + 4 métadonnées speaker.
    Évaluation sur le test set. Métrique principale : <b>F1 macro</b>.
    </p></div>""", unsafe_allow_html=True)

    # Pipeline cards
    col1, col2, col3 = st.columns(3)
    for col, (nm, params, color) in zip([col1,col2,col3], [
        ("Logistic Regression","C=1.0, max_iter=1000, class_weight='balanced'", ACCENT),
        ("Random Forest","n_estimators=300, class_weight='balanced'", A3),
        ("XGBoost","n_estimators=300, max_depth=6, lr=0.1, subsample=0.8", "#f59e0b"),
    ]):
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='--accent:linear-gradient(90deg,{color},{color}88);'>
              <span class='kpi-label'>{nm}</span>
              <p style='font-size:11px!important;color:#64748b!important;margin-top:8px!important;'>{params}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Résultats — 3 Classes (test set)")
    res3 = pd.DataFrame({
        "Modèle":    ["Logistic Regression","Random Forest","XGBoost"],
        "Accuracy":  [0.5375, 0.5651, 0.5556],
        "F1 Macro":  [0.5442, 0.5697, 0.5577],
        "F1 Fake":   [0.6465, 0.6431, 0.6239],
        "F1 Nuanced":[0.4705, 0.4874, 0.4824],
        "F1 Real":   [0.5156, 0.5786, 0.5668],
        "Precision Fake":[0.61,0.62,0.56],
        "Recall Fake":   [0.69,0.67,0.70],
    })
    st.dataframe(res3.style.highlight_max(subset=["F1 Macro","F1 Fake","F1 Nuanced","F1 Real"],
                                           props="background-color:#1a1a4e; color:white;"),
                 use_container_width=True, hide_index=True)

    color_map_m = {"Logistic Regression": ACCENT, "Random Forest": A3, "XGBoost": "#f59e0b"}
    tab1, tab2, tab3 = st.tabs(["F1 par classe", "Precision / Recall", "Radar"])
    with tab1:
        fig = go.Figure()
        metriques = ["F1 Macro","F1 Fake","F1 Nuanced","F1 Real"]
        for _, row in res3.iterrows():
            clr = color_map_m.get(row["Modèle"], ACCENT)
            fig.add_trace(go.Bar(name=row["Modèle"], x=metriques,
                                 y=[row[m] for m in metriques],
                                 marker_color=clr, marker_line_width=0,
                                 text=[f"{row[m]:.3f}" for m in metriques],
                                 textposition="outside", textfont=dict(color="#fff")))
        fig.update_layout(barmode="group")
        fig = dark_fig(fig, "F1 par classe et modèle — 3 classes", 420)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Precision Fake", x=res3["Modèle"],
                              y=res3["Precision Fake"], marker_color="#ef4444",
                              text=res3["Precision Fake"].map("{:.2f}".format),
                              textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig2.add_trace(go.Bar(name="Recall Fake", x=res3["Modèle"],
                              y=res3["Recall Fake"], marker_color="#f97316",
                              text=res3["Recall Fake"].map("{:.2f}".format),
                              textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig2.update_layout(barmode="group")
        fig2 = dark_fig(fig2, "Precision vs Recall — classe Fake", 380)
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        cats = ["F1 Macro","F1 Fake","F1 Nuanced","F1 Real","Accuracy"]
        fig3 = go.Figure()
        fill_colors = {"Logistic Regression": "rgba(99,102,241,.12)",
                       "Random Forest": "rgba(6,182,212,.12)",
                       "XGBoost": "rgba(245,158,11,.12)"}
        for _, row in res3.iterrows():
            vals = [row["F1 Macro"],row["F1 Fake"],row["F1 Nuanced"],row["F1 Real"],row["Accuracy"]]
            clr = color_map_m.get(row["Modèle"], ACCENT)
            fig3.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                           name=row["Modèle"], line=dict(color=clr, width=2),
                                           fill="toself",
                                           fillcolor=fill_colors.get(row["Modèle"],"rgba(99,102,241,.12)")))
        fig3.update_layout(polar=dict(radialaxis=dict(range=[0.4,0.75], gridcolor=GRID, tickfont=dict(color="#94a3b8")),
                                      angularaxis=dict(gridcolor=GRID, tickfont=dict(color="#94a3b8")),
                                      bgcolor=BG),
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig3 = dark_fig(fig3, "Radar — comparaison modèles 3 classes", 420)
        st.plotly_chart(fig3, use_container_width=True)

    # Matrices de confusion
    st.divider()
    st.markdown("### Matrices de Confusion (3 classes — test set)")
    cm_data = {
        "Logistic Regression": np.array([[236,66,39],[123,205,149],[82,128,239]]),
        "Random Forest":        np.array([[229,84,28],[113,214,150],[71,102,276]]),
        "XGBoost":             np.array([[240,60,41],[117,207,153],[90,103,256]]),
    }
    cls_names = ["Fake","Nuanced","Real"]
    m_sel = st.selectbox("Modèle :", list(cm_data.keys()))
    cm = cm_data[m_sel]
    cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    col1, col2 = st.columns([2,1])
    with col1:
        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm, x=cls_names, y=cls_names,
            colorscale=[[0,"#111124"],[0.5,"#312e81"],[1,ACCENT]],
            text=[[f"{cm_norm[i][j]:.0%}<br>({cm[i][j]})" for j in range(3)] for i in range(3)],
            texttemplate="%{text}", textfont=dict(color="#fff",size=12), showscale=True,
        ))
        fig_cm.update_layout(xaxis_title="Prédit", yaxis_title="Réel")
        fig_cm = dark_fig(fig_cm, f"Confusion Matrix — {m_sel}", 380)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        acc_per_class = np.diag(cm_norm)
        for cls, acc in zip(cls_names, acc_per_class):
            color = "#22c55e" if acc>0.6 else ("#fbbf24" if acc>0.45 else "#ef4444")
            st.markdown(f"""
            <div class='kpi-card' style='margin-top:8px;'>
              <span class='kpi-label'>{cls}</span>
              <span class='kpi-value' style='color:{color};'>{acc:.0%}</span>
              <span class='kpi-sub'>Recall (diagonale)</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight'>
      <h4>Lecture de la matrice</h4>
      <p>La diagonale = prédictions correctes.
      La classe <b style='color:#fbbf24;'>nuanced</b> est la plus confondue (frontière floue avec fake et real).
      Les modèles confondent peu <b>fake ↔ real</b> directement — les erreurs extrêmes restent rares.</p>
    </div>""", unsafe_allow_html=True)

    # SHAP features
    st.divider()
    st.markdown("### Top Features TF-IDF — Coefficient LR (classe Fake vs Real)")
    shap_fake = {"claim":0.089,"hoax":0.082,"false":0.079,"lie":0.071,"rumor":0.065,
                 "debunked":0.061,"incorrect":0.058,"wrong":0.054,"misled":0.049,"fake":0.046}
    shap_real = {"according":0.076,"data_shows":0.068,"study":0.063,"percent":0.059,
                 "research":0.055,"sources":0.051,"fact":0.048,"report":0.044,"official":0.041,"confirmed":0.038}
    col1, col2 = st.columns(2)
    with col1:
        fig_s1 = px.bar(x=list(shap_fake.values()), y=list(shap_fake.keys()),
                        orientation="h", color_discrete_sequence=["#ef4444"],
                        text=[f"+{v:.3f}" for v in shap_fake.values()])
        fig_s1.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_s1 = dark_fig(fig_s1, "Features favorisant FAKE", 360)
        fig_s1.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_s1, use_container_width=True)
    with col2:
        fig_s2 = px.bar(x=list(shap_real.values()), y=list(shap_real.keys()),
                        orientation="h", color_discrete_sequence=["#22c55e"],
                        text=[f"+{v:.3f}" for v in shap_real.values()])
        fig_s2.update_traces(textposition="outside", textfont_color="#fff", marker_line_width=0)
        fig_s2 = dark_fig(fig_s2, "Features favorisant REAL", 360)
        fig_s2.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_s2, use_container_width=True)

# ===========================================================================
# PAGE 04 — FEATURES AMÉLIORÉES (résultats réels NB04)
# ===========================================================================
elif page == "Features Améliorées":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('microscope', 24, '#6366f1')}
      <div><span class='badge'>Notebook 04</span><h2 style='margin:0;'>Features Améliorées — 4 Configurations</h2></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class='alert-info'><p>
    4 configurations comparées : Baseline · Extended (+17 features) · Piste 1 (sujet dans TF-IDF) · Piste 2 (min_df=2).
    </p></div>""", unsafe_allow_html=True)

    configs_desc = [
        ("Baseline","TF-IDF (1-2g, min_df=3) + 4 méta","9 053 features","Référence"),
        ("Extended","Baseline + 17 features (sujet, parti, linguistiques)","9 070 features","+17 features vs baseline"),
        ("Piste 1","Sujet concaténé au texte avant TF-IDF","9 962 features","health-care → token à part entière"),
        ("Piste 2","TF-IDF avec min_df=2 + 4 méta","16 471 features","Vocabulaire élargi +7 418 tokens"),
    ]
    cols = st.columns(4)
    for col, (nm, desc, feats, note) in zip(cols, configs_desc):
        with col:
            st.markdown(f"""
            <div class='card'>
              <span class='badge'>{nm}</span>
              <h3 style='font-size:13px!important;'>{desc}</h3>
              <span class='kpi-value' style='font-size:18px;'>{feats}</span>
              <p style='margin-top:6px!important;font-size:11px!important;'>{note}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()
    # Résultats NB04 exacts
    df_imp = pd.DataFrame({
        "Config": ["Baseline","Baseline","Baseline","Extended","Extended","Extended",
                   "Piste 1","Piste 1","Piste 1","Piste 2","Piste 2","Piste 2"],
        "Modèle": ["LR","RF","XGBoost"]*4,
        "Accuracy": [0.5375,0.5651,0.5556, 0.5375,0.5635,0.5406, 0.5304,0.5588,0.5501, 0.5375,0.5556,0.5320],
        "F1 Macro": [0.5442,0.5697,0.5577, 0.5437,0.5677,0.5431, 0.5368,0.5642,0.5520, 0.5454,0.5603,0.5338],
        "F1 Fake":  [0.6465,0.6431,0.6239, 0.6390,0.6325,0.6166, 0.6440,0.6461,0.6304, 0.6537,0.6402,0.6150],
        "F1 Nuanced":[0.4705,0.4874,0.4824,0.4658,0.4926,0.4507,0.4561,0.4813,0.4672,0.4727,0.4744,0.4407],
        "F1 Real":  [0.5156,0.5786,0.5668, 0.5262,0.5780,0.5620, 0.5103,0.5654,0.5584, 0.5098,0.5661,0.5458],
    })
    tab1, tab2 = st.tabs(["F1 Macro par config", "Tableau complet"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_cfg = px.bar(df_imp, x="Config", y="F1 Macro", color="Modèle", barmode="group",
                             color_discrete_map={"LR":ACCENT,"RF":A3,"XGBoost":"#f59e0b"},
                             text="F1 Macro")
            fig_cfg.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                   textfont_color="#fff", marker_line_width=0)
            fig_cfg = dark_fig(fig_cfg, "F1 Macro par configuration", 400)
            st.plotly_chart(fig_cfg, use_container_width=True)
        with col2:
            fig_heat = px.density_heatmap(df_imp, x="Config", y="Modèle", z="F1 Macro",
                                           color_continuous_scale=[[0,"#1e1e3a"],[1,ACCENT]],
                                           histfunc="avg")
            fig_heat = dark_fig(fig_heat, "Heatmap F1 Macro moyen", 400)
            st.plotly_chart(fig_heat, use_container_width=True)
    with tab2:
        st.dataframe(df_imp.style.highlight_max(subset=["F1 Macro","F1 Fake","F1 Real"],
                                                  props="background-color:#1a1a4e;color:white;"),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight'>
      <h4>Conclusions NB04</h4>
      <p>• <b>RF Baseline reste optimal</b> (F1 macro 0.570) — les configurations améliorées n'apportent pas de gain significatif.<br>
      • Piste 1 (sujet dans TF-IDF) : léger recul (-0.005 F1) — sujet trop dilué dans la matrice TF-IDF.<br>
      • Piste 2 (min_df=2) : résultats mitigés — XGBoost recule (-0.024), RF recule légèrement (-0.009).<br>
      • Extended features : RF recule de -0.002 — les 17 nouvelles features ajoutent du bruit.</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Features Linguistiques — Distribution par Classe")
    ling = pd.DataFrame({
        "Feature": ["has_absolute_lang","has_hedging","stmt_word_count"],
        "fake":    [0.210, 0.089, 17.029],
        "nuanced": [0.198, 0.096, 18.515],
        "real":    [0.263, 0.109, 18.251],
    })
    fig_ling = go.Figure()
    for cls, clr in [("fake","#ef4444"),("nuanced","#fbbf24"),("real","#22c55e")]:
        fig_ling.add_trace(go.Bar(name=cls, x=ling["Feature"], y=ling[cls],
                                   marker_color=clr, marker_line_width=0,
                                   text=[f"{v:.3f}" for v in ling[cls]],
                                   textposition="outside", textfont=dict(color="#fff")))
    fig_ling.update_layout(barmode="group")
    fig_ling = dark_fig(fig_ling, "Valeurs moyennes des features linguistiques par classe", 380)
    st.plotly_chart(fig_ling, use_container_width=True)
    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>Le langage absolu (<code>has_absolute_lang</code>) est plus fréquent dans les déclarations <b>réelles</b> (0.26 vs 0.21).
      Le hedging est légèrement plus présent dans le réel. Les différences sont faibles — signal limité.</p>
    </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 05 — BERT
# ===========================================================================
elif page == "BERT":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('brain', 24, '#6366f1')}
      <div><span class='badge'>Notebook 05</span><h2 style='margin:0;'>Fine-tuning BERT / DistilBERT</h2></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
          <span class='badge'>DistilBERT</span>
          <h3>distilbert-base-uncased</h3>
          <p>• max_length=128 · padding · truncation</p>
          <p>• Epochs : 5 | Batch : 32 | LR : 2e-5</p>
          <p>• Warmup : 10% | Device : MPS (Apple Silicon)</p>
          <p>• Classes : 3 (fake / nuanced / real)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        bert_r = pd.DataFrame({
            "Modèle": ["DistilBERT (3cls)","BERT-base (3cls)","LR TF-IDF+méta","RF TF-IDF+méta","XGBoost TF-IDF+méta"],
            "F1 Macro": [0.541, 0.558, 0.544, 0.570, 0.578],
            "Accuracy": [0.534, 0.549, 0.537, 0.565, 0.576],
        })
        st.dataframe(bert_r.style.highlight_max(subset=["F1 Macro"],
                                                  props="background-color:#1a1a4e;color:white;"),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Courbes d'Apprentissage DistilBERT")
    epochs = [1,2,3,4,5]
    fig_lr = make_subplots(rows=1, cols=2, subplot_titles=["Loss (train vs val)","F1 Macro (val)"])
    fig_lr.add_trace(go.Scatter(x=epochs, y=[0.891,0.754,0.698,0.661,0.638], name="Train Loss",
                                line=dict(color=ACCENT,width=2), mode="lines+markers"), 1, 1)
    fig_lr.add_trace(go.Scatter(x=epochs, y=[0.812,0.789,0.781,0.778,0.776], name="Val Loss",
                                line=dict(color="#ef4444",width=2), mode="lines+markers"), 1, 1)
    fig_lr.add_trace(go.Scatter(x=epochs, y=[0.481,0.512,0.528,0.534,0.541], name="F1 Val",
                                line=dict(color="#22c55e",width=2), mode="lines+markers"), 1, 2)
    fig_lr.update_layout(height=360, plot_bgcolor=BG, paper_bgcolor=BG, font_color="#94a3b8",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          margin=dict(t=40,b=20,l=20,r=20))
    for ax in ["xaxis","yaxis","xaxis2","yaxis2"]:
        fig_lr.update_layout(**{ax:dict(gridcolor=GRID, tickfont=dict(color="#94a3b8"))})
    st.plotly_chart(fig_lr, use_container_width=True)

    # Comparaison radar
    radar_models = ["DistilBERT","BERT-base","LR+méta","RF+méta","XGBoost+méta"]
    f1_vals = [0.541, 0.558, 0.544, 0.570, 0.578]
    acc_vals = [0.534, 0.549, 0.537, 0.565, 0.576]
    col1, col2 = st.columns(2)
    with col1:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="F1 Macro", x=radar_models, y=f1_vals,
                                  marker_color=[A2,A2,ACCENT,A3,"#f59e0b"], marker_line_width=0,
                                  text=[f"{v:.3f}" for v in f1_vals],
                                  textposition="outside", textfont=dict(color="#fff")))
        fig_cmp = dark_fig(fig_cmp, "BERT vs Modèles Classiques — F1 Macro", 360)
        st.plotly_chart(fig_cmp, use_container_width=True)
    with col2:
        col1i, col2i = st.columns(2)
        with col1i:
            st.markdown(f"""
            <div class='card'>
              <span class='badge badge-green'>{ico_html('check-circle', 12, '#86efac')} Avantages BERT</span>
              <p>Meilleur que LR pur (sans méta)</p>
              <p>Gère les nuances sémantiques</p>
              <p>Robuste aux variations orthographiques</p>
              <p>BERT-base > DistilBERT</p>
            </div>""", unsafe_allow_html=True)
        with col2i:
            st.markdown(f"""
            <div class='card'>
              <span class='badge badge-red'>{ico_html('x-circle', 12, '#fca5a5')} Limites BERT</span>
              <p>Textes trop courts (~18 mots)</p>
              <p>Pas d'accès aux métadonnées</p>
              <p>10x plus lent à entraîner</p>
              <p>Inférieur à XGBoost+méta</p>
            </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 06 — BINAIRE vs MULTI
# ===========================================================================
elif page == "Binaire vs Multi-classe":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('scale', 24, '#6366f1')}
      <div><span class='badge'>Notebook 06</span><h2 style='margin:0;'>Classification Binaire vs Multi-classe</h2></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
          <span class='badge'>3 classes</span>
          <h3>Multi-classe (baseline)</h3>
          <p>• fake (0) · nuanced (1) · real (2)</p>
          <p>• Train : <b style='color:#818cf8;'>10 240 exemples</b></p>
          <p>• Distribution : fake 27.7% · nuanced 36.8% · real 35.5%</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card'>
          <span class='badge badge-green'>2 classes</span>
          <h3>Binaire — sans "nuanced"</h3>
          <p>• fake (0) · real (1)</p>
          <p>• Train : <b style='color:#22c55e;'>6 472 exemples</b> (-3 768 nuanced)</p>
          <p>• Distribution : fake 43.8% · real 56.2%</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    res_comp = pd.DataFrame({
        "Modèle":       ["LR","RF","XGBoost"],
        "F1 (3 cls)":   [0.5442,0.5697,0.5577],
        "F1 (2 cls)":   [0.7867,0.7837,0.7723],
        "Acc (3 cls)":  [0.5375,0.5651,0.5556],
        "Acc (2 cls)":  [0.7899,0.7873,0.7747],
        "Gain F1 (pp)": [24.25, 21.40, 21.46],
    })

    col1, col2 = st.columns([3,2])
    with col1:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="3 classes", x=res_comp["Modèle"],
                                  y=res_comp["F1 (3 cls)"], marker_color="#5b7fa6",
                                  text=[f"{v:.3f}" for v in res_comp["F1 (3 cls)"]],
                                  textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig_cmp.add_trace(go.Bar(name="2 classes", x=res_comp["Modèle"],
                                  y=res_comp["F1 (2 cls)"], marker_color="#22c55e",
                                  text=[f"{v:.3f}" for v in res_comp["F1 (2 cls)"]],
                                  textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig_cmp.update_layout(barmode="group")
        fig_cmp = dark_fig(fig_cmp, "F1 Macro : 3 classes vs 2 classes", 400)
        st.plotly_chart(fig_cmp, use_container_width=True)
    with col2:
        # Gain — couleur valide rgba (pas de hex 8 chiffres)
        fig_gain = px.bar(res_comp, x="Modèle", y="Gain F1 (pp)", color="Gain F1 (pp)",
                           color_continuous_scale=[[0,"rgba(34,197,94,0.3)"],[1,"rgba(34,197,94,1)"]],
                           text="Gain F1 (pp)")
        fig_gain.update_traces(texttemplate="+%{text:.2f} pp", textposition="outside",
                                textfont_color="#fff", marker_line_width=0)
        fig_gain = dark_fig(fig_gain, "Gain en F1 Macro (en pp)", 400)
        fig_gain.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_gain, use_container_width=True)

    st.dataframe(res_comp.style.background_gradient(subset=["Gain F1 (pp)"], cmap="Greens"),
                 use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='insight'>
      <h4>Résultats — Hypothèse confirmée</h4>
      <p>+21 à +24 pp de F1 macro en passant de 3 à 2 classes.
      La classe "nuanced" (half-true) est intrinsèquement ambiguë — impossible à discriminer fiablement.
      Les 3 modèles bénéficient également du passage au binaire.
      → Configuration binaire retenue pour l'évaluation OOD (NB07).</p>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### F1 par classe — Détail LR binaire")
    c2_lr = pd.DataFrame({
        "Classe":    ["Fake","Real"],
        "Precision": [0.77, 0.80],
        "Recall":    [0.75, 0.82],
        "F1":        [0.7608, 0.8126],
        "Support":   [341, 449],
    })
    col1, col2 = st.columns(2)
    with col1:
        fig_c2 = go.Figure()
        for m, clr in [("Precision","#6366f1"),("Recall","#06b6d4"),("F1","#22c55e")]:
            fig_c2.add_trace(go.Bar(name=m, x=c2_lr["Classe"], y=c2_lr[m],
                                     marker_color=clr, marker_line_width=0,
                                     text=[f"{v:.3f}" for v in c2_lr[m]],
                                     textposition="outside", textfont=dict(color="#fff")))
        fig_c2.update_layout(barmode="group")
        fig_c2 = dark_fig(fig_c2, "LR binaire — Precision / Recall / F1", 360)
        st.plotly_chart(fig_c2, use_container_width=True)
    with col2:
        cm_bin = np.array([[255, 86],[80, 369]])
        cm_bin_n = cm_bin.astype(float)/cm_bin.sum(axis=1,keepdims=True)
        fig_cm2 = go.Figure(go.Heatmap(
            z=cm_bin_n, x=["Fake","Real"], y=["Fake","Real"],
            colorscale=[[0,"#111124"],[0.5,"#312e81"],[1,ACCENT]],
            text=[[f"{cm_bin_n[i][j]:.0%}<br>({cm_bin[i][j]})" for j in range(2)] for i in range(2)],
            texttemplate="%{text}", textfont=dict(color="#fff",size=13), showscale=False,
        ))
        fig_cm2.update_layout(xaxis_title="Prédit", yaxis_title="Réel")
        fig_cm2 = dark_fig(fig_cm2, "Confusion Matrix — LR Binaire", 360)
        st.plotly_chart(fig_cm2, use_container_width=True)

# ===========================================================================
# PAGE 07 — ÉVAL OUT-OF-DOMAIN
# ===========================================================================
elif page == "Éval. Out-of-Domain":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('globe', 24, '#6366f1')}
      <div><span class='badge'>Notebook 07</span><h2 style='margin:0;'>Évaluation Out-of-Domain : LIAR → WELFake</h2></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    | | **LIAR** (entraînement) | **WELFake** (évaluation OOD) |
    |---|---|---|
    | **Source** | PolitiFact | Kaggle · McIntire · Reuters · BuzzFeed |
    | **Type** | Déclaration ~18 mots | Article complet ~540 mots |
    | **Labels** | Binaire fake=0 / real=1 | Binaire fake=0 / real=1 |
    | **Taille entraînement** | 6 472 (après filtre nuanced) | 72 134 articles |
    | **Métadonnées speaker** | Oui | Non |
    """)

    st.divider()
    ind = pd.DataFrame({
        "Modèle":    ["LR + méta","RF + méta","XGBoost + méta","RF texte seul"],
        "Domain":    ["in"]*4,
        "Accuracy":  [0.7899,0.7873,0.7747,0.6291],
        "F1 Macro":  [0.7867,0.7837,0.7723,0.6151],
        "F1 Fake":   [0.7608,0.7558,0.7493,0.5415],
        "F1 Real":   [0.8126,0.8117,0.7954,0.6886],
    })
    ood = pd.DataFrame({
        "Modèle":    ["LR + méta","RF + méta","XGBoost + méta","RF texte seul"],
        "Domain":    ["ood"]*4,
        "Accuracy":  [0.5000,0.5000,0.5004,0.4683],
        "F1 Macro":  [0.3333,0.3333,0.3344,0.4644],
        "F1 Fake":   [0.0000,0.0000,0.0020,0.4185],
        "F1 Real":   [0.6667,0.6667,0.6668,0.5103],
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Résultats In-Domain (référence)")
        st.dataframe(ind[["Modèle","Accuracy","F1 Macro","F1 Fake","F1 Real"]]
                     .style.highlight_max(subset=["F1 Macro"], props="background-color:#1a1a4e;color:white;"),
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### Résultats Out-of-Domain (WELFake)")
        st.dataframe(ood[["Modèle","Accuracy","F1 Macro","F1 Fake","F1 Real"]]
                     .style.highlight_max(subset=["F1 Macro"], props="background-color:#0a2e0a;color:white;"),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='alert-warn'><p>
    Les modèles avec métadonnées prédisent <b>systématiquement "real"</b> sur WELFake
    (F1 fake ≈ 0, F1 real ≈ 0.67). Cause : credibility_score=0 et lie_rate=0 pour tous les articles
    WELFake → le modèle assimile tous les speakers à des "inconnus fiables".
    </p></div>""", unsafe_allow_html=True)

    st.divider()
    chute = pd.DataFrame({
        "Modèle":        ["LR + méta","RF + méta","XGBoost + méta","RF texte seul"],
        "In-Domain":     [0.7867,0.7837,0.7723,0.6151],
        "Out-of-Domain": [0.3333,0.3333,0.3344,0.4644],
        "Chute (pp)":    [-45.34,-45.04,-43.79,-15.07],
    })
    col1, col2 = st.columns([3,2])
    with col1:
        fig_chute = go.Figure()
        fig_chute.add_trace(go.Bar(name="In-Domain (LIAR)", x=chute["Modèle"],
                                    y=chute["In-Domain"], marker_color="#378Add",
                                    text=[f"{v:.3f}" for v in chute["In-Domain"]],
                                    textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig_chute.add_trace(go.Bar(name="Out-of-Domain (WELFake)", x=chute["Modèle"],
                                    y=chute["Out-of-Domain"], marker_color="#D85A30",
                                    text=[f"{v:.3f}" for v in chute["Out-of-Domain"]],
                                    textposition="outside", textfont=dict(color="#fff"), marker_line_width=0))
        fig_chute.update_layout(barmode="group")
        fig_chute = dark_fig(fig_chute, "F1 Macro — In-Domain vs Out-of-Domain", 420)
        st.plotly_chart(fig_chute, use_container_width=True)
    with col2:
        fig_chute2 = px.bar(chute, x="Chute (pp)", y="Modèle", orientation="h",
                             color="Chute (pp)", color_continuous_scale=[[0,"#ef4444"],[1,"#fbbf24"]],
                             text="Chute (pp)")
        fig_chute2.update_traces(texttemplate="%{text:.1f} pp", textposition="outside",
                                  textfont_color="#fff", marker_line_width=0)
        fig_chute2 = dark_fig(fig_chute2, "Chute de performance (pp)", 420)
        fig_chute2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_chute2, use_container_width=True)

    st.markdown("""
    <div class='insight'>
      <h4>Interprétation</h4>
      <p>
      Modèles avec méta : chute de <b style='color:#ef4444;'>-45 pp</b> en OOD.<br>
      RF texte seul : chute de seulement <b style='color:#fbbf24;'>-15 pp</b> — bien meilleure généralisation.<br>
      Écart de distribution : statements courts (18 mots) vs articles longs (540 mots) → densité TF-IDF 2.7x supérieure sur WELFake.<br>
      Couverture du vocabulaire LIAR sur WELFake : <b style='color:#818cf8;'>99.7%</b> — le problème n'est pas le vocabulaire mais la distribution.
      </p>
    </div>""", unsafe_allow_html=True)

    # Stats WELFake
    st.divider()
    st.markdown("### Statistiques WELFake")
    col1,col2,col3,col4 = st.columns(4)
    for col, (v,lbl,sub) in zip([col1,col2,col3,col4], [
        ("72 134","Articles WELFake","~540 mots / article"),
        ("50.5%","Fake (label=0)","36 184 articles"),
        ("49.5%","Real (label=1)","35 950 articles"),
        ("99.7%","Couverture vocab","LIAR → WELFake"),
    ]):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <span class='kpi-value'>{v}</span>
              <span class='kpi-label'>{lbl}</span>
              <span class='kpi-sub'>{sub}</span>
            </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE : DEMO LIVE
# ===========================================================================
elif page == "Demo Live":
    st.markdown(f"""
    <div class='section-title'>
      {ico_html('target', 24, '#6366f1')}
      <div><span class='badge'>Demo Interactive</span><h2 style='margin:0;'>Testez les Modèles sur vos Textes</h2></div>
    </div>""", unsafe_allow_html=True)

    if not data_ok:
        st.error("Données LIAR nécessaires. Vérifiez `data/raw/`.")
        st.stop()

    # Sélection source de texte
    st.markdown("### Source du texte")
    source_tab = st.radio("", ["Saisie manuelle", "Exemples LIAR", "Dataset WELFake (externe)"],
                           horizontal=True)

    st.divider()
    col_left, col_right = st.columns([1,1])

    with col_left:
        model_choice = st.selectbox("Modèle :", ["Logistic Regression", "Random Forest"])

        if source_tab == "Saisie manuelle":
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(6,182,212,.06));
                        border:1px solid rgba(99,102,241,.25); border-left:3px solid #6366f1;
                        border-radius:10px; padding:12px 15px; margin-bottom:12px;'>
              <div style='display:flex;align-items:flex-start;gap:10px;'>
                <div style='margin-top:2px;flex-shrink:0;'>{ico_html('alert-triangle', 16, '#818cf8')}</div>
                <div>
                  <p style='color:#818cf8!important;font-size:11px!important;font-weight:700!important;
                             text-transform:uppercase;letter-spacing:.07em;margin:0 0 5px!important;'>
                    Contexte du modele - Politique americaine uniquement
                  </p>
                  <p style='color:#94a3b8!important;font-size:12px!important;line-height:1.7!important;margin:0!important;'>
                    Modele entraine exclusivement sur le <b style='color:#c7d2fe;'>LIAR Dataset</b>
                    (declarations politiques US verifiees par <b style='color:#c7d2fe;'>PolitiFact</b>, 2007-2017).<br>
                    Optimal pour du texte <b style='color:#c7d2fe;'>court en anglais</b> (~18 mots)
                    sur des sujets <b style='color:#c7d2fe;'>politiques americains</b>
                    (economie, sante, immigration, fiscalite, elections...).<br>
                    <span style='color:#64748b;font-size:11px;'>
                    Textes en francais, hors politique US ou articles longs : resultats peu fiables.
                    </span>
                  </p>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            user_text = st.text_area("Entrez votre declaration (en anglais) :", height=110,
                placeholder="Ex: The United States has the highest corporate tax rate in the world.")
            source_label = None

        elif source_tab == "Exemples LIAR":
            exemples = {
                "FAKE — pants-fire : Obamacare biggest tax increase":
                    ("Obamacare is the biggest tax increase in the history of the United States.", "pants-fire"),
                "REAL — mostly-true : 8/10 Americans health care priority":
                    ("More than 8 in 10 Americans say reducing health care costs is a top priority for Congress.", "mostly-true"),
                "FAKE — false : Crime rising all over the country":
                    ("Crime is rising all over the country, especially in big cities. Things are out of control.", "false"),
                "REAL — true : US 50 states, 330 million":
                    ("The United States has 50 states and a population of around 330 million people.", "true"),
                "FAKE — barely-true : Vaccines autism multiple studies":
                    ("Multiple studies have suggested that vaccines may be linked to autism in children.", "barely-true"),
                "REAL — mostly-true : ACA expanded Medicaid":
                    ("The Affordable Care Act expanded Medicaid coverage to millions of low-income Americans.", "mostly-true"),
            }
            ex_key = st.selectbox("Choisir un exemple :", list(exemples.keys()))
            user_text, source_label = exemples[ex_key]
            st.text_area("Texte de l'exemple :", value=user_text, height=100, disabled=True)

        else:  # WELFake
            st.markdown("""
            <div class='alert-info'><p>
            Chargement d'un échantillon aléatoire depuis <b>WELFake_sample.csv</b> (version allégée pour le cloud).
            Les articles sont longs (~540 mots) — le modèle (entraîné sur LIAR ~18 mots) peut se tromper davantage.
            </p></div>""", unsafe_allow_html=True)
            wf = load_welfake_sample(500)
            if wf is None:
                st.warning("Echantillon WELFake non trouvé dans `data/external/WELFake_sample.csv`.")
                user_text, source_label = "", None
            else:
                wf_row = wf.sample(1, random_state=np.random.randint(0,999)).iloc[0]
                user_text = str(wf_row.get("text",""))[:1000]
                source_label = "real" if wf_row["label"]==1 else "fake"
                st.text_area(f"Article WELFake (label réel: {source_label.upper()}) :",
                              value=user_text, height=130, disabled=True)
                if st.button("Charger un autre article WELFake"):
                    st.rerun()

    with col_right:
        if st.button("Analyser le texte", use_container_width=True):
            if not str(user_text).strip():
                st.warning("Entrez un texte.")
            else:
                with st.spinner("Analyse…"):
                    processed = preprocess_text(str(user_text))
                    X_inp = tfidf.transform([processed])
                    model = lr_model if model_choice=="Logistic Regression" else rf_model
                    pred  = model.predict(X_inp)[0]
                    proba = model.predict_proba(X_inp)[0]
                    is_fake = (pred == 1)
                    p_fake = proba[1]
                    p_real = proba[0]

                st.divider()
                # Verdict
                if is_fake:
                    st.markdown("<div style='text-align:center;padding:16px 0;'><span class='tag-fake'>FAKE NEWS</span></div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align:center;padding:16px 0;'><span class='tag-real'>VRAIE INFO</span></div>",
                                unsafe_allow_html=True)

                # Ground truth si dispo
                if source_label and source_tab != "Saisie manuelle":
                    gt_is_fake = "fake" in source_label.lower() or source_label=="fake"
                    correct = (gt_is_fake == is_fake)
                    color = "#22c55e" if correct else "#ef4444"
                    verdict_txt = "Correct" if correct else "Incorrect"
                    st.markdown(f"""
                    <div class='alert-{"success" if correct else "warn"}'><p>
                    Label réel : <b>{source_label.upper()}</b> — Prédiction : <b>{verdict_txt}</b>
                    </p></div>""", unsafe_allow_html=True)

                # Probabilités
                c_f = "#ef4444" if p_fake>0.5 else "#64748b"
                c_r = "#22c55e" if p_real>0.5 else "#64748b"
                st.markdown(f"""
                <div class='card' style='padding:16px;'>
                  <p style='font-weight:600;color:#e2e8f0!important;margin:0 0 6px!important;'>
                    Fake News : <b style='color:{c_f};'>{p_fake:.1%}</b>
                  </p>
                  <div class='prob-wrap'><div class='prob-fill' style='width:{p_fake*100:.1f}%;background:{c_f};'></div></div>
                  <p style='font-weight:600;color:#e2e8f0!important;margin:12px 0 6px!important;'>
                    Vraie Info : <b style='color:{c_r};'>{p_real:.1%}</b>
                  </p>
                  <div class='prob-wrap'><div class='prob-fill' style='width:{p_real*100:.1f}%;background:{c_r};'></div></div>
                </div>""", unsafe_allow_html=True)

                # Stats
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class='kpi-card' style='margin-top:8px;'>
                      <span class='kpi-value'>{max(p_fake, p_real):.0%}</span>
                      <span class='kpi-label'>Confiance</span>
                      <span class='kpi-sub'>{model_choice}</span>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class='kpi-card' style='margin-top:8px;'>
                      <span class='kpi-value'>{len(processed.split())}</span>
                      <span class='kpi-label'>Tokens après preprocessing</span>
                      <span class='kpi-sub'>{len(str(user_text).split())} mots orig.</span>
                    </div>""", unsafe_allow_html=True)

                # Top tokens
                st.markdown("**Top tokens les plus discriminants :**")
                fn = tfidf.get_feature_names_out()
                vec = X_inp.toarray()[0]
                nz = np.where(vec>0)[0]
                if len(nz):
                    top_idx = nz[np.argsort(vec[nz])[::-1]][:8]
                    tok_df = pd.DataFrame({"Token":fn[top_idx],"TF-IDF":vec[top_idx].round(4)})
                    st.dataframe(tok_df.style.background_gradient(subset=["TF-IDF"], cmap="Blues"),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun token du vocabulaire LIAR trouvé.")

                # Text preview
                st.markdown("**Texte après preprocessing :**")
                st.code(processed[:400]+"…" if len(processed)>400 else processed, language="text")

    st.divider()
    # Note OOD
    st.markdown("""
    <div class='alert-warn'><p>
    <b>Note sur la généralisation :</b> Ce modèle est entraîné sur des déclarations politiques LIAR (~18 mots).
    Les textes WELFake sont des articles longs (~540 mots) — la distribution TF-IDF est très différente.
    La chute de performance attendue est de -15 pp (RF texte) à -45 pp (modèles avec méta) — cf. NB07.
    </p></div>""", unsafe_allow_html=True)
