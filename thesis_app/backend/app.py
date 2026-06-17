"""
Narrative Similarity - Academic Thesis Demo
Alexandru Ioan Cuza University · Faculty of Computer Science · 2026

Ana Ciobanu | Supervisor: Diana Trandabat
"Latent and Explicit Narrative Representations for Multilingual Narrative Similarity"

Run:
    pip install streamlit plotly pandas sentence-transformers scipy
    streamlit run app.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import concurrent.futures
import time
import urllib.error
import urllib.request

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Narrative Similarity · Thesis",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== ACADEMIC DESIGN SYSTEM ========================
# Visual identity: deep navy + aged parchment + vermillion accent
# Typography: academic serif for display, humanist sans for UI, mono for data
# Signature element: the three narrative aspects rendered as glowing "facet chips"
# with animated underscore lines on hover - a callback to academic annotation marks.

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --navy:       #0f1e35;
    --navy-mid:   #1a2f4a;
    --navy-light: #243d5e;
    --parchment:  #f4f0e8;
    --parchment2: #ede8dc;
    --cream:      #faf8f4;
    --ink:        #0f1e35;
    --ink-muted:  #4a5d72;
    --ink-faint:  #8a9db0;
    --vermillion: #c23b2a;
    --gold:       #b8913a;
    --gold-light: #d4aa5a;
    --sage:       #3d6b58;
    --teal:       #2a5f72;
    --surface:    #ffffff;
    --border:     #d8d2c6;
    --border-l:   #e8e2d8;

    --coa-color:  #2a5f72;
    --out-color:  #3d6b58;
    --thm-color:  #6b4a7a;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: var(--cream) !important;
    color: var(--ink) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: var(--parchment) !important; }
[data-testid="stSidebar"] .stRadio label { color: var(--parchment) !important; }
[data-testid="stSidebar"] hr { border-color: var(--navy-light) !important; opacity: 1; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem !important; }

/* Radio buttons in sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 6px 10px !important;
    border-radius: 4px !important;
    transition: background 0.15s;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--navy-light) !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-weight: 500 !important;
    color: var(--navy) !important;
    letter-spacing: -0.01em;
}

h1 { font-size: 2.4rem !important; line-height: 1.15; }
h2 { font-size: 1.8rem !important; margin-bottom: 0.2rem; }
h3 { font-size: 1.3rem !important; }

.thesis-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--vermillion);
    margin-bottom: 6px;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8em;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border-l);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    color: var(--ink);
}
.card-navy {
    background: var(--navy);
    border: 1px solid var(--navy-light);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    color: var(--parchment);
}
.card-parchment {
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    color: var(--ink);
}

.card-accent-coa  { border-left: 3px solid var(--coa-color);  background: #f0f6f8; }
.card-accent-out  { border-left: 3px solid var(--out-color);  background: #f0f6f3; }
.card-accent-thm  { border-left: 3px solid var(--thm-color);  background: #f5f0f7; }
.card-accent-red  { border-left: 3px solid var(--vermillion); background: #fdf5f4; }
.card-accent-gold { border-left: 3px solid var(--gold);       background: #fdf9f0; }

/* ── Metric boxes ── */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border-l);
    border-radius: 6px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--navy);
    line-height: 1;
}
.metric-val-accent { color: var(--vermillion); }
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.67rem;
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 5px;
}

/* ── Aspect facet chips ── */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 5px;
    margin-bottom: 4px;
}
.pill-coa { background: #deedf2; color: var(--coa-color); border: 1px solid #b0d0dc; }
.pill-out { background: #deeee8; color: var(--out-color); border: 1px solid #b0d0c4; }
.pill-thm { background: #ede5f0; color: var(--thm-color); border: 1px solid #c4b0cc; }
.pill-match { background: #e0f0e8; color: var(--sage); border: 1px solid #b0d0bc; }
.pill-miss  { background: #fce8e6; color: var(--vermillion); border: 1px solid #f0b8b4; }
.pill-gold  { background: #fdf4e0; color: var(--gold); border: 1px solid #e0cc90; }

/* ── Story block ── */
.story-block {
    background: var(--parchment);
    border: 1px solid var(--border);
    border-left: 3px solid var(--navy-light);
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--ink);
}

/* ── Section headers ── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-faint);
    margin-bottom: 0.6rem;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-l);
}

/* ── RQ boxes ── */
.rq-box {
    background: var(--parchment);
    border: 1px solid var(--border);
    border-left: 3px solid var(--vermillion);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.87rem;
    line-height: 1.5;
}
.rq-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--vermillion);
    margin-bottom: 3px;
}

/* ── Result highlight ── */
.result-highlight {
    background: var(--navy);
    color: var(--parchment);
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.75rem;
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem;
    line-height: 1.6;
}
.result-highlight strong { color: var(--gold-light); }

/* ── Finding box ── */
.finding-box {
    background: var(--cream);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.87rem;
    line-height: 1.55;
    color: var(--ink);
}

/* ── Tables ── */
[data-testid="stDataFrame"] { background: var(--surface) !important; border-radius: 6px !important; }
.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78em !important; }

/* ── Misc ── */
hr { border-color: var(--border-l) !important; opacity: 1 !important; }
.baseline-note {
    font-size: 0.75rem;
    color: var(--ink-muted);
    font-style: italic;
    margin-top: 4px;
    line-height: 1.5;
}

/* ── Sidebar thesis identity block ── */
.sidebar-title {
    font-family: 'EB Garamond', serif;
    font-size: 1.35rem;
    font-weight: 500;
    line-height: 1.25;
    color: var(--parchment);
    margin-bottom: 2px;
}
.sidebar-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: var(--ink-faint);
    line-height: 1.5;
}
.sidebar-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #6a8aab;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Streamlit override for wider tables */
.stDataFrame { width: 100% !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ======================== PATH RESOLUTION ========================
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent

def resolve_data_path(relative_path: str, fallback_filename: str | None = None) -> Path:
    candidates = [
        SCRIPT_DIR / relative_path,
        Path(__file__).resolve().parent / Path(relative_path).name,
    ]
    if fallback_filename:
        candidates.append(Path("/mnt/data") / fallback_filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

DATA_PATHS = {
    "dev":        resolve_data_path("narrative_nlp/dataset/dev_track_a.jsonl", "dev_track_a.jsonl"),
    "dev_labels": resolve_data_path("narrative_nlp/dataset/dev_track_a_labels.jsonl", "dev_track_a_labels.jsonl"),
    "aspects_v1": resolve_data_path("narrative_nlp/dataset/aspects_cache_v1.json", "aspects_cache_v1.json"),
    "aspects_v2": resolve_data_path("narrative_nlp/dataset/aspects_cache_v2.json", "aspects_cache_v2.json"),
    "aspects_v3": resolve_data_path("narrative_nlp/dataset/aspects_cache_v3.json", "aspects_cache_v3.json"),
    "test_a":     resolve_data_path("narrative_nlp/dataset/test_track_a.jsonl", "test_track_a.jsonl"),
    "test_b":     resolve_data_path("narrative_nlp/dataset/test_track_b.jsonl", "test_track_b.jsonl"),
    "synth":      resolve_data_path("narrative_nlp/dataset/synthetic_data_for_classification.jsonl", "synthetic_data_for_classification.jsonl"),
    "synth_new":  resolve_data_path("narrative_nlp/dataset/synthetic_data_new.jsonl", "synthetic_data_new.jsonl"),
    "trans_quality": resolve_data_path("narrative_nlp/dataset/romanian_narrative_similarity_dataset/translation_quality_report.json", "translation_quality_report.json"),
    "pred_track_a": resolve_data_path("narrative_nlp/G2_condition_predictions/Condition_G2_512_full_train_track_a.jsonl",
                                   "Condition_G2_512_full_train_track_a.jsonl"),
    "test_a_labels": resolve_data_path("narrative_nlp/dataset/test_track_a_labels.jsonl",
                                    "test_track_a_labels.jsonl"),
}

def norm_text(text: str) -> str:
    return " ".join(str(text).split())

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text)))

def char_count(text: str) -> int:
    return len(str(text))

# ======================== DATA LOADING ========================
@st.cache_data
def load_jsonl_file(path: Path):
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

@st.cache_data
def load_dev_triples():   return load_jsonl_file(DATA_PATHS["dev"])
@st.cache_data
def load_test_a_rows():   return load_jsonl_file(DATA_PATHS["test_a"])
@st.cache_data
def load_test_b_rows():   return load_jsonl_file(DATA_PATHS["test_b"])
@st.cache_data
def load_synth_rows():    return load_jsonl_file(DATA_PATHS["synth"])
@st.cache_data
def load_synth_new_rows(): return load_jsonl_file(DATA_PATHS["synth_new"])
@st.cache_data
def load_dev_labels():    return load_jsonl_file(DATA_PATHS["dev_labels"])

@st.cache_data
def load_aspects_cache(version: int) -> Dict:
    key = f"aspects_v{version}"
    path = DATA_PATHS.get(key)
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {norm_text(k): v for k, v in raw.items()}

def load_all_aspect_caches():
    return {1: load_aspects_cache(1), 2: load_aspects_cache(2), 3: load_aspects_cache(3)}

# ======================== THESIS TABLES (precomputed from thesis) ========================
DF_ROBERTA_200 = pd.DataFrame([
    ["V1 (narrative prose)", "Full text", 57.5, 0.0271, 0.0069, "✓"],
    ["V1 (narrative prose)", "CoA",       62.5, 0.0270, 0.0061, "✓"],
    ["V1 (narrative prose)", "Outcomes",  51.5, 0.0137, 0.0805, "-"],
    ["V1 (narrative prose)", "Theme",     51.0, 0.0013, 0.8888, "-"],
    ["V2 (role-label steps)", "Full text", 57.5, 0.0271, 0.0069, "✓"],
    ["V2 (role-label steps)", "CoA",       55.0, 0.0178, 0.0197, "✓"],
    ["V2 (role-label steps)", "Outcomes",  57.0, 0.0174, 0.0399, "✓"],
    ["V2 (role-label steps)", "Theme",     53.5, 0.0140, 0.1962, "-"],
    ["V3 (compact phrases)", "Full text",  57.5, 0.0271, 0.0069, "✓"],
    ["V3 (compact phrases)", "CoA",        53.5, 0.0114, 0.2439, "-"],
    ["V3 (compact phrases)", "Outcomes",   54.5, 0.0174, 0.1749, "-"],
    ["V3 (compact phrases)", "Theme",      56.5, 0.0146, 0.2168, "-"],
], columns=["Version", "Aspect", "% pos>neg", "Mean diff.", "p-value", "Sig."])

DF_BGEM3_200 = pd.DataFrame([
    ["V1 (narrative prose)", "Full text", 60.5, 0.0092, 0.0020, "✓"],
    ["V1 (narrative prose)", "CoA",       59.5, 0.0067, 0.0364, "✓"],
    ["V1 (narrative prose)", "Outcomes",  57.0, 0.0071, 0.0125, "✓"],
    ["V1 (narrative prose)", "Theme",     53.5, 0.0020, 0.3993, "-"],
    ["V2 (role-label steps)", "Full text", 60.5, 0.0092, 0.0020, "✓"],
    ["V2 (role-label steps)", "CoA",       63.0, 0.0110, 0.0007, "✓"],
    ["V2 (role-label steps)", "Outcomes",  54.0, 0.0076, 0.0191, "✓"],
    ["V2 (role-label steps)", "Theme",     52.5, 0.0084, 0.0548, "-"],
    ["V3 (compact phrases)", "Full text",  60.5, 0.0092, 0.0020, "✓"],
    ["V3 (compact phrases)", "CoA",        57.5, 0.0068, 0.0273, "✓"],
    ["V3 (compact phrases)", "Outcomes",   57.0, 0.0054, 0.2797, "-"],
    ["V3 (compact phrases)", "Theme",      55.5, 0.0094, 0.0248, "✓"],
], columns=["Version", "Aspect", "% pos>neg", "Mean diff.", "p-value", "Sig."])

DF_166_GOLD = pd.DataFrame([
    ["RoBERTa", "V1", "CoA",      63.3, 0.0308, 0.221, "[+0.0096, +0.0523]", 0.0049, "**"],
    ["RoBERTa", "V2", "Outcomes", 57.8, 0.0194, 0.160, "[+0.0012, +0.0382]", 0.0408, "*"],
    ["BGE-M3",  "V2", "CoA",      64.5, 0.0107, 0.236, "[+0.0039, +0.0176]", 0.0028, "**"],
    ["BGE-M3",  "V1", "Outcomes", 58.4, 0.0075, 0.186, "[+0.0014, +0.0135]", 0.0177, "*"],
    ["BGE-M3",  "V3", "Theme",    56.6, 0.0117, 0.199, "[+0.0026, +0.0205]", 0.0111, "*"],
], columns=["Model", "Ver.", "Aspect", "%pos>neg", "Diff.", "Cohen's d", "95% CI", "p", "Sig."])

DF_CORRELATIONS = pd.DataFrame([
    ["RoBERTa", "V1 (narrative prose)",  "CoA–Full",      0.2719, 0.0004, "***"],
    ["RoBERTa", "V1 (narrative prose)",  "Outcomes–Full", 0.3943, 0.0000, "***"],
    ["RoBERTa", "V1 (narrative prose)",  "CoA–Outcomes",  0.2356, 0.0022, "**"],
    ["BGE-M3",  "V2 (role-label steps)", "CoA–Full",      0.2057, 0.0078, "**"],
    ["BGE-M3",  "V2 (role-label steps)", "Outcomes–Full", 0.0486, 0.5338, "n.s."],
    ["BGE-M3",  "V2 (role-label steps)", "CoA–Outcomes",  0.2120, 0.0061, "**"],
    ["BGE-M3",  "V3 (compact phrases)",  "Theme–Full",    0.0267, 0.7328, "n.s."],
], columns=["Model", "Version", "Pair", "r", "p-value", "Sig."])

DF_INPUT_LENGTH = pd.DataFrame([
    [128, 68.25, 68.23, 62.25, 62.21],
    [256, 65.25, 65.15, 66.75, 66.74],
    [384, 67.25, 67.16, 69.00, 68.98],
    [512, 66.25, 66.17, 69.25, 69.23],
], columns=["max_len", "Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"])

DF_SYNTHETIC_EFFECT = pd.DataFrame([
    ["Baseline", "Organiser only",   512, 66.25, 66.17, 69.25, 69.23],
    ["Baseline", "Organiser + extra",512, 64.50, 64.46, 61.00, 61.00],
    ["G2",       "Organiser only",   512, 70.00, 69.98, 64.75, 64.70],
    ["G2",       "Organiser + extra",512, 69.00, 69.00, 71.75, 71.72],
], columns=["Model", "Synthetic data", "max_len", "Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"])

DF_BEST_MODELS = pd.DataFrame([
    ["Baseline (best Track A)", 128, "Organiser only",   68.25, 68.23, 62.25, 62.21],
    ["Baseline (best Track B)", 512, "Organiser only",   66.25, 66.17, 69.25, 69.23],
    ["G2 (best overall)",       512, "Organiser + extra",69.00, 69.00, 71.75, 71.72],
    ["Ensemble (G2 + Qwen)",    512, "Organiser + extra","-",   "-",   72.00, 71.98],
], columns=["Model", "max_len", "Synthetic data", "Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"])

DF_ASPECTS_PERF = pd.DataFrame([
    ["G2 (reference)",          "Full text + two latent heads",                   69.00, 69.00, 71.75, 71.72],
    ["G2 COA",                  "Course of action only + latent heads",            61.75, 61.71, 62.00, 61.94],
    ["E",                       "CoA + outcomes + theme (concatenated)",           62.50, 62.50, 60.25, 60.12],
    ["N",                       "Full text + CoA & outcomes + latent heads",       69.50, 69.50, 70.00, 69.92],
    ["H",                       "Full text appended with CoA & outcomes",          68.00, 67.00, 61.00, 60.90],
    ["P* (CoA+Outcomes views)", "Multi-view: full, CoA, outcomes (max_len=384)",   66.75, 66.72, 63.25, 63.20],
    ["P (all views)",           "Multi-view: full, CoA, outcomes, theme (max_len=384)", 67.00, 66.97, 59.75, 59.74],
], columns=["Condition", "Architecture / Input", "Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"])

DF_ENSEMBLE = pd.DataFrame([
    ["G2 (best single)",        71.75, 71.72],
    ["Qwen3 zero-shot",         66.00, 66.00],
    ["Ensemble (G2+Qwen 0.90/0.10)", 72.00, 71.98],
], columns=["Model", "Track B Acc. (%)", "Track B F1 (%)"])

DF_MULTILINGUAL = pd.DataFrame([
    ["multilingual-e5-base",                   "English",     70.00, 69.95, 68.25, 68.23],
    ["multilingual-e5-base",                   "Romanian MT", 66.25, 66.17, 63.50, 63.44],
    ["paraphrase-multilingual-mpnet-base-v2",  "English",     65.00, 65.00, 63.00, 62.94],
    ["paraphrase-multilingual-mpnet-base-v2",  "Romanian MT", 62.00, 62.00, 59.00, 58.99],
], columns=["Model", "Language", "Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"])

# ======================== HELPER FUNCTIONS ========================
def word_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text)))

def char_count(text: str) -> int:
    return len(str(text))

@st.cache_data
def build_synth_model_stats():
    synth = load_synth_rows()
    if not synth:
        return pd.DataFrame()
    model_counts = Counter(r.get("model_name", "unknown") for r in synth)
    return pd.DataFrame(model_counts.most_common(), columns=["Model", "Stories Generated"])

@st.cache_data
def build_dev_label_analysis():
    labels = load_dev_labels()
    if not labels:
        return pd.DataFrame()
    records = []
    for row in labels:
        records.append({
            "coa_match_a":      row.get("course_of_actions", [False, False])[0],
            "coa_match_b":      row.get("course_of_actions", [False, False])[1],
            "outcomes_match_a": row.get("outcomes", [False, False])[0],
            "outcomes_match_b": row.get("outcomes", [False, False])[1],
            "theme_match_a":    row.get("abstract_theme", [False, False])[0],
            "theme_match_b":    row.get("abstract_theme", [False, False])[1],
        })
    return pd.DataFrame(records)

# ======================== PLOTLY THEME ========================
def navy_fig(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f9f7f3",
        font=dict(family="Inter, sans-serif", color="#0f1e35", size=11),
        margin=dict(l=10, r=10, t=44, b=30),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor="#e8e2d8", linecolor="#d0cac0"),
        yaxis=dict(gridcolor="#e8e2d8", linecolor="#d0cac0"),
        title=dict(font=dict(family="EB Garamond, Georgia, serif", size=15, color="#0f1e35")),
    )
    return fig

PALETTE_A = ["#0f1e35", "#2a5f72", "#3d6b58", "#6b4a7a", "#b8913a", "#c23b2a"]

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("""
<div class="sidebar-title">Latent & Explicit<br>Narrative Representations</div>
<div style="margin: 6px 0 12px 0; font-family: Inter, sans-serif; font-size: 0.72rem; color: #6a8aab; line-height: 1.6;">
  Ana Ciobanu<br>
  <span style="opacity:0.7;">Supervisor: Diana Trandabat</span><br>
  Alexandru Ioan Cuza University<br>Faculty of Computer Science · 2026
</div>
<hr>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "📋 Thesis Overview",
            "🗂 Dataset & EDA",
            "🔬 Aspect Extraction Versions",
            "📈 Aspect Informativeness",
            "📊 Experimental Results",
            "🌍 Multilingual Comparison",
            "🌐 Translation Explorer",
            "🔍 Prediction Browser",
            "⚡ Live Aspect Extraction",
        ],
        label_visibility="collapsed",
    )
    page = page.split(" ", 1)[1]

    st.markdown("""
<hr>
<div style="font-family: JetBrains Mono, monospace; font-size: 0.6rem; color: #3a5a7a; line-height: 1.7;">
  RQ1 · Input length<br>
  RQ2 · Latent heads (G2)<br>
  RQ3 · Explicit aspects<br>
  RQ4 · Synthetic data<br>
  RQ5 · Multilingual transfer<br>
  RQ6 · Interactive app
</div>
""", unsafe_allow_html=True)

# ======================== PAGE: Thesis Overview ========================
if page == "Thesis Overview":
    # Hero
    st.markdown("<div class='thesis-eyebrow'>Master Dissertation · July 2026</div>", unsafe_allow_html=True)
    st.markdown("# Latent and Explicit Narrative Representations for Multilingual Narrative Similarity")
    st.markdown("""
<div style="font-family: Inter, sans-serif; font-size: 0.9rem; color: #4a5d72; margin-bottom: 1.5rem; line-height: 1.7;">
Ana Ciobanu &nbsp;·&nbsp; Supervisor: Diana Trandabat &nbsp;·&nbsp;
Faculty of Computer Science, Alexandru Ioan Cuza University, Iași &nbsp;·&nbsp; July 2026
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card card-parchment" style="font-family: 'EB Garamond', Georgia, serif; font-size: 1.05rem; line-height: 1.8; border-left: 3px solid #0f1e35;">
<strong>Abstract.</strong> Compared to text similarity where we look for overlap of words, narrative similarity is more
challenging because stories can align in plot, outcomes, or abstract meaning even when their characters,
settings, and wording differ. This thesis evaluates neural methods for narrative similarity assessment
using the <strong>SemEval-2026 Task 4</strong> benchmark, focusing on how modelling decisions affect both direct
triplet comparison (Track A) and embedding-based story representation (Track B). The central finding is
that flexible latent representations learned from full text are more effective for predictive accuracy,
while explicit narrative aspects remain valuable for interpretability and error analysis.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Key results summary
    st.markdown("### Key Results at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-box">
          <div class="metric-val">72.0%</div>
          <div class="metric-label">Best Track B Accuracy<br>(Ensemble)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-box">
          <div class="metric-val">69.0%</div>
          <div class="metric-label">Best Track A Accuracy<br>(G2 model)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-box">
          <div class="metric-val metric-val-accent">+2.5pp</div>
          <div class="metric-label">G2 gain over baseline<br>(Track B)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-box">
          <div class="metric-val">62.5%</div>
          <div class="metric-label">Best aspect signal<br>V1 CoA, RoBERTa</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Research questions
    st.markdown("### Research Questions")
    rqs = [
        ("RQ1", "How does maximum input length influence performance in narrative similarity assessment?",
         "Track B accuracy increases from 62.25% at 128 tokens to 69.25% at 512 tokens. Track A is less sensitive. Longer inputs are critical for standalone story embeddings."),
        ("RQ2", "Does a full-text model with generic latent projection heads (G2) improve over a plain full-text baseline?",
         "Yes. G2 achieves 71.75% Track B accuracy vs. 69.25% for the best baseline - a +2.5pp gain - with latent heads learning task-relevant narrative subspaces."),
        ("RQ3", "Do explicit narrative aspects improve predictive accuracy, or are they mainly useful for interpretation?",
         "Mainly for interpretation. The best aspect-aware variant (Condition N) reaches 70.00% Track B, still 1.75pp behind G2. Theme extraction is especially noisy."),
        ("RQ4", "What is the effect of additional synthetic data on narrative similarity models?",
         "It harms the baseline (Track B: 69.25%→61.00%) but substantially helps G2 (64.75%→71.75%). Latent heads regularise the model against diverse synthetic examples."),
        ("RQ5", "How does the approach behave when translated into Romanian via machine translation?",
         "Performance drops 3–5pp. multilingual-e5-base is more robust (68.25% EN → 63.50% RO). The experiment confirms viability but highlights sensitivity to translation noise."),
        ("RQ6", "How can an interactive application support dataset exploration, aspect inspection, and result visualisation?",
         "This Streamlit application addresses RQ6 directly - supporting live aspect extraction, dataset EDA, and complete result visualisation."),
    ]
    for code, q, a in rqs:
        st.markdown(f"""<div class="rq-box">
          <div class="rq-label">{code}</div>
          <div style="font-weight:500; margin-bottom:4px;">{q}</div>
          <div style="color:#3d6b58; font-size:0.85rem;">✦ {a}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Contributions
    st.markdown("### Main Contributions")
    contribs = [
        ("Reproducible baseline", "Two-stage contrastive pipeline (Stage 1: synthetic triplets, Stage 2: dev fine-tuning) with fixed seeds and deterministic CUDA settings."),
        ("Input length ablation", "Systematic evaluation at 128 / 256 / 384 / 512 tokens revealing the asymmetric sensitivity of Track A and Track B."),
        ("G2 latent-head architecture", "Two generic 256-dim projection heads over the full-text encoder, trained with a head triplet loss regulariser (weight 0.3)."),
        ("Three-version aspect extraction pipeline", "Verbose prose (V1), structured role-label (V2), and compact phrase (V3) variants, with a quantitative informativeness analysis."),
        ("Synthetic data experiments", "LLM-generated additional triplets that harm the baseline but substantially benefit the G2 model via latent regularisation."),
        ("English–Romanian multilingual comparison", "NLLB-200 machine-translated dataset evaluated with two multilingual encoders (E5 and MPNet) in a controlled within-model comparison."),
        ("Embedding ensemble for Track B", "Task-tuned G2 embeddings combined with zero-shot Qwen3-Embedding-0.6B (0.90/0.10 weight), achieving 72.00% Track B accuracy."),
        ("Interactive Streamlit application", "This application - supporting live extraction, EDA, and result visualisation."),
    ]
    for i, (title, desc) in enumerate(contribs, 1):
        st.markdown(f"""<div class="finding-box">
          <span style="font-family: JetBrains Mono, monospace; font-size: 0.68rem; color: #b8913a; font-weight:600;">{i:02d}</span>
          &nbsp;<strong>{title}</strong><br>
          <span style="font-size:0.84rem; color: #4a5d72;">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dissertation Structure")
    chapters = [
        ("Chapter 1", "Introduction", "Motivation, research questions, and contributions."),
        ("Chapter 2", "Background & Related Work", "Narrative similarity, dense embeddings, contrastive learning, explicit aspects, SemEval-2026 Task 4, multilingual representations."),
        ("Chapter 3", "Data, Resources & Exploratory Tools", "Dataset composition, synthetic data, Romanian translation, aspect extraction pipeline, EDA, and this application."),
        ("Chapter 4", "Methodology", "Baseline model, reproducibility setup, input length configuration, G2 architecture, aspect-based variants, multilingual setup, embedding ensemble."),
        ("Chapter 5", "Experiments, Results & Discussion", "All six RQ experiments with results, error analysis, and interpretation."),
        ("Chapter 6", "Conclusions", "Summary of findings, limitations, and future work."),
    ]
    cols = st.columns(3)
    for i, (ch, title, desc) in enumerate(chapters):
        with cols[i % 3]:
            st.markdown(f"""<div class="card" style="min-height: 110px;">
              <div style="font-family: JetBrains Mono, monospace; font-size: 0.65rem; color: #b8913a; font-weight:600;">{ch}</div>
              <div style="font-weight:600; font-size: 0.9rem; margin: 3px 0 5px;">{title}</div>
              <div style="font-size:0.78rem; color:#4a5d72; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ======================== PAGE: Dataset & EDA ========================
elif page == "Dataset & EDA":
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.1 & 3.5</div>", unsafe_allow_html=True)
    st.markdown("## Dataset & Exploratory Data Analysis")
    st.markdown("SemEval-2026 Task 4 narrative similarity benchmark - story summaries from the *Tell Me Again!* corpus linked via Wikidata identifiers.")

    dev   = load_dev_triples()
    test_a = load_test_a_rows()
    test_b = load_test_b_rows()
    synth  = load_synth_rows()
    synth_new = load_synth_new_rows()

    if not dev:
        st.warning("`dev_track_a.jsonl` not found - data files not loaded.")
        st.stop()

    def get_texts_from_triples(rows):
        texts = []
        for r in rows:
            for f in ["anchor_text", "text_a", "text_b"]:
                t = r.get(f, "")
                if t: texts.append(t)
        return texts

    dev_texts  = get_texts_from_triples(dev)
    ta_texts   = get_texts_from_triples(test_a)
    tb_texts   = [r.get("text","") for r in test_b if r.get("text","")]
    syn_texts  = get_texts_from_triples(synth)
    synn_texts = get_texts_from_triples(synth_new)

    # ── Overview metrics ──
    st.markdown("### Corpus Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    def m(col, val, label):
        col.markdown(f"""<div class="metric-box">
          <div class="metric-val" style="font-size:1.9rem;">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    n_unique_dev = len(set(dev_texts))
    m(m1, len(dev),   "Dev triples")
    m(m2, n_unique_dev, "Unique dev stories")
    m(m3, len(test_a), "Test-A triples")
    m(m4, len(test_b), "Test-B stories")
    m(m5, len(synth)+len(synth_new), "Synth. triples (total)")

    st.markdown("---")

    # ── Table 3.5: Dataset composition ──
    st.markdown("### Table 3.5 · Dataset Composition and Label Balance")

    def row_stats(name, texts, rows, labelled=True):
        wl = [word_count(t) for t in texts]
        row = {
            "Dataset": name,
            "Rows": len(rows),
            "Unique texts": len(set(texts)),
            "Mean words": round(np.mean(wl), 1) if wl else 0,
            "Median words": int(np.median(wl)) if wl else 0,
            "Max words": int(np.max(wl)) if wl else 0,
        }
        if labelled and rows and "text_a_is_closer" in rows[0]:
            pos = sum(bool(r.get("text_a_is_closer")) for r in rows)
            row["text_a closer (%)"] = round(pos/len(rows)*100, 1)
        else:
            row["text_a closer (%)"] = "-"
        return row

    summary_df = pd.DataFrame([
        row_stats("Dev set (Track A)", dev_texts, dev),
        row_stats("Test set (Track A)", ta_texts, test_a, labelled=False),
        row_stats("Test set (Track B)", tb_texts, test_b, labelled=False),
        row_stats("Organiser synthetic", syn_texts, synth),
        row_stats("Additional synthetic", synn_texts, synth_new),
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.markdown("<div class='baseline-note'>Label balance close to 50% across all labelled splits prevents positional strategies from inflating accuracy.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Figure 3.1: token length analysis ──
    st.markdown("### Figure 3.1 · Tokenizer Length Analysis (Table 3.6)")
    st.markdown("Texts exceeding each maximum length under the `all-roberta-large-v1` tokenizer.")

    # Approximate table 3.6 from thesis
    tok_df = pd.DataFrame([
        ["Organiser synthetic", 5691, 190.65, 193.0, 367, 93.06, 3.58, 0.00],
        ["Additional synthetic", 3291, 220.50, 219.0, 312, 100.00, 9.75, 0.00],
        ["Development set",      479,  156.18, 150.0, 390, 64.30,  6.89, 0.42],
        ["Track A test",         848,  155.12, 148.0, 626, 61.91,  6.01, 0.47],
        ["Track B test",         849,  154.06, 149.0, 436, 64.19,  5.54, 0.47],
    ], columns=["Dataset", "Unique texts", "Mean tokens", "Median tokens", "Max tokens", ">128 (%)", ">256 (%)", ">384 (%)"])
    st.dataframe(tok_df, hide_index=True, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        pct_df = pd.DataFrame({
            "max_len": [128, 256, 384],
            "Organiser synthetic": [93.06, 3.58, 0.00],
            "Additional synthetic": [100.00, 9.75, 0.00],
            "Development set": [64.30, 6.89, 0.42],
            "Track A test": [61.91, 6.01, 0.47],
            "Track B test": [64.19, 5.54, 0.47],
        })
        pct_melted = pct_df.melt("max_len", var_name="Dataset", value_name="Texts truncated (%)")
        fig = px.line(pct_melted, x="max_len", y="Texts truncated (%)", color="Dataset",
                      markers=True, title="Texts exceeding each max_len",
                      color_discrete_sequence=PALETTE_A)
        fig = navy_fig(fig)
        fig.update_xaxes(title="Max sequence length (tokens)")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        ret_df = pd.DataFrame({
            "max_len": [128, 256, 384, 512],
            "Organiser synthetic": [0.69, 0.99, 1.00, 1.00],
            "Additional synthetic": [0.59, 0.97, 1.00, 1.00],
            "Development set": [0.82, 0.99, 1.00, 1.00],
            "Track A / B test": [0.82, 0.99, 1.00, 1.00],
        })
        ret_melted = ret_df.melt("max_len", var_name="Dataset", value_name="Mean retained fraction")
        fig2 = px.line(ret_melted, x="max_len", y="Mean retained fraction", color="Dataset",
                       markers=True, title="Mean retained token fraction",
                       color_discrete_sequence=PALETTE_A)
        fig2 = navy_fig(fig2)
        fig2.update_xaxes(title="Max sequence length (tokens)")
        fig2.update_yaxes(range=[0.5, 1.02])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='baseline-note'>128 tokens removes too much context (especially synthetic stories). 384–512 tokens retain nearly all narrative information - motivating the input-length ablation in §5.2.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Word-length distributions ──
    st.markdown("### Word-Length Distributions by Dataset")
    dist_data = []
    for texts, name in [(dev_texts, "Dev"), (ta_texts, "Test A"), (tb_texts, "Test B"), (syn_texts, "Synth Org."), (synn_texts, "Synth Add.")]:
        for t in texts:
            dist_data.append({"Dataset": name, "Words": word_count(t)})
    dist_df = pd.DataFrame(dist_data)

    fig3 = px.box(dist_df, x="Dataset", y="Words",
                  title="Word-count distribution by dataset",
                  color="Dataset", color_discrete_sequence=PALETTE_A,
                  category_orders={"Dataset": ["Dev","Test A","Test B","Synth Org.","Synth Add."]})
    fig3.update_layout(showlegend=False)
    navy_fig(fig3, height=360)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Dev set label analysis ──
    st.markdown("### Development Set Annotation Analysis")
    label_df = build_dev_label_analysis()
    if not label_df.empty:
        coa_a   = (label_df["coa_match_a"] == label_df["coa_match_b"]).sum()/len(label_df)*100
        out_a   = (label_df["outcomes_match_a"] == label_df["outcomes_match_b"]).sum()/len(label_df)*100
        theme_a = (label_df["theme_match_a"] == label_df["theme_match_b"]).sum()/len(label_df)*100

        lc1, lc2, lc3 = st.columns(3)
        for col, v, lbl in [(lc1, coa_a,"CoA agreement"), (lc2, out_a,"Outcomes agreement"), (lc3, theme_a,"Theme agreement")]:
            col.markdown(f"""<div class="metric-box">
              <div class="metric-val" style="font-size:2rem;">{v:.1f}%</div>
              <div class="metric-label">{lbl}<br>across candidates</div>
            </div>""", unsafe_allow_html=True)

        agree_df = pd.DataFrame({"Aspect": ["CoA","Outcomes","Theme"], "Agreement %": [coa_a, out_a, theme_a]})
        coa_rate   = (label_df["coa_match_a"].sum()+label_df["coa_match_b"].sum()) / (len(label_df)*2) * 100
        out_rate   = (label_df["outcomes_match_a"].sum()+label_df["outcomes_match_b"].sum()) / (len(label_df)*2) * 100
        theme_rate = (label_df["theme_match_a"].sum()+label_df["theme_match_b"].sum()) / (len(label_df)*2) * 100
        rate_df = pd.DataFrame({"Aspect": ["CoA","Outcomes","Theme"], "Match Rate %": [coa_rate, out_rate, theme_rate]})

        ac1, ac2 = st.columns(2)
        with ac1:
            fig4 = px.bar(agree_df, x="Aspect", y="Agreement %", text="Agreement %", range_y=[0,100],
                          title="Aspect label consistency (A vs B candidates)",
                          color="Aspect", color_discrete_map={"CoA":"#2a5f72","Outcomes":"#3d6b58","Theme":"#6b4a7a"})
            fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            navy_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True)
        with ac2:
            fig5 = px.bar(rate_df, x="Aspect", y="Match Rate %", text="Match Rate %", range_y=[0,100],
                          title="Aspect match rate (overall)",
                          color="Aspect", color_discrete_map={"CoA":"#2a5f72","Outcomes":"#3d6b58","Theme":"#6b4a7a"})
            fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            navy_fig(fig5)
            st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<div class='baseline-note'>Agreement = both candidates share the same aspect label. Match rate = frequency a candidate is labelled as matching the anchor for that aspect.</div>", unsafe_allow_html=True)
    else:
        st.info("Dev label file not found.")

    st.markdown("---")

    # ── Synthetic LLMs ──
    st.markdown("### Table 3.7 · Organiser vs. Additional Synthetic Data")
    cmp_df = pd.DataFrame([
        ["Organiser synthetic", 1900, 5691, 157.85, 158.0, 49.63],
        ["Additional synthetic", 1097, 3291, 187.50, 186.0, 49.68],
    ], columns=["Source","Rows","Unique texts","Mean words","Median words","text_a closer (%)"])
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)
    st.markdown("<div class='baseline-note'>Additional synthetic stories are ~30 words longer on average, generated by llama3.1:8b following organiser-style prompts. No text overlap between the two sources.</div>", unsafe_allow_html=True)

    synth_model_df = build_synth_model_stats()
    if not synth_model_df.empty:
        st.markdown("#### Organiser Synthetic: LLM Generation Models")
        fig6 = px.bar(synth_model_df, x="Stories Generated", y="Model", orientation="h",
                      title="Stories per generation model",
                      color="Stories Generated", color_continuous_scale=["#deedf2","#0f1e35"])
        fig6.update_layout(showlegend=False, coloraxis_showscale=False)
        navy_fig(fig6, height=260)
        st.plotly_chart(fig6, use_container_width=True)

# ======================== PAGE: Aspect Extraction Versions ========================
elif page == "Aspect Extraction Versions":
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.4</div>", unsafe_allow_html=True)
    st.markdown("## Narrative Aspect Extraction Pipeline")
    st.markdown("Three extraction versions were developed, each reflecting a different trade-off between **informativeness**, **structure**, and **compactness**. All are produced via zero-shot prompting of instruction-tuned Llama 3.1 8B (local inference with Ollama).")

    st.markdown("### Table 3.4 · Overview of Extraction Versions")

    v_data = [
        ("V1", "Verbose narrative prose",
         "Separate per-aspect prompts. Unrestricted prose output.",
         "Preserves rich narrative detail; best CoA discriminative signal (62.5%, p=0.0061).",
         "Verbose, occasionally overlapping; may infer outcomes not stated in story."),
        ("V2", "Structured role-label steps",
         "Single combined JSON prompt; numbered steps with role labels (protagonist, authority, ally).",
         "Makes event progression abstract & transferable; outcomes V2 is significant (p=0.0399).",
         "Can still produce semi-inferred resolutions; role labels add structure overhead."),
        ("V3", "Compact phrase-based JSON",
         "Highly constrained JSON prompt; semicolon-separated 3-5 phrase CoA; 2-sentence outcomes; 2-4 theme phrases.",
         "Cleanest output; easiest to inspect and append to model input.",
         "No V3 aspect reaches significance - compactness removes discriminative signal."),
    ]
    for code, name, desc, pro, con in v_data:
        st.markdown(f"""<div class="card" style="margin-bottom:0.5rem;">
          <span class="pill pill-gold">{code}</span>
          <strong style="font-size:1rem; font-family: 'EB Garamond', serif;">{name}</strong>
          <div style="font-size:0.83rem; color:#4a5d72; margin: 6px 0 4px;">{desc}</div>
          <div style="font-size:0.81rem;"><span style="color:#3d6b58;">✦ {pro}</span></div>
          <div style="font-size:0.81rem;"><span style="color:#c23b2a;">✦ {con}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Aspect length analysis (Table 3.9) ──
    st.markdown("### Table 3.9 · Mean Word Length of Extracted Aspects")
    len_df = pd.DataFrame([
        ["V1 - verbose prose",      65.03, 50.60, 55.59],
        ["V2 - structured role-label", 56.29, 24.89, 7.30],
        ["V3 - compact constrained",   17.51, 10.71, 7.39],
    ], columns=["Version", "CoA (words)", "Outcomes (words)", "Theme (words)"])
    st.dataframe(len_df, hide_index=True, use_container_width=True)

    len_melted = pd.DataFrame([
        {"Version": "V1", "Aspect": "CoA",      "Mean words": 65.03},
        {"Version": "V1", "Aspect": "Outcomes",  "Mean words": 50.60},
        {"Version": "V1", "Aspect": "Theme",     "Mean words": 55.59},
        {"Version": "V2", "Aspect": "CoA",       "Mean words": 56.29},
        {"Version": "V2", "Aspect": "Outcomes",  "Mean words": 24.89},
        {"Version": "V2", "Aspect": "Theme",     "Mean words": 7.30},
        {"Version": "V3", "Aspect": "CoA",       "Mean words": 17.51},
        {"Version": "V3", "Aspect": "Outcomes",  "Mean words": 10.71},
        {"Version": "V3", "Aspect": "Theme",     "Mean words": 7.39},
    ])
    fig = px.bar(len_melted, x="Version", y="Mean words", color="Aspect", barmode="group",
                 title="Mean word length per aspect and extraction version",
                 color_discrete_map={"CoA":"#2a5f72","Outcomes":"#3d6b58","Theme":"#6b4a7a"})
    navy_fig(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='baseline-note'>A shorter representation is easier to inspect but does not guarantee stronger similarity signal. V3 compactness comes at the cost of discriminative power.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Side-by-side comparison on a story ──
    st.markdown("### Compare Aspect Extractions for a Story")
    caches = load_all_aspect_caches()
    dev = load_dev_triples()

    stats_rows = []
    for ver, cache in caches.items():
        if not cache: continue
        stats_rows.append({
            "Version": f"V{ver}",
            "Avg CoA (words)": round(np.mean([word_count(v.get("coa","")) for v in cache.values()]),1),
            "Avg Outcomes (words)": round(np.mean([word_count(v.get("outcomes","")) for v in cache.values()]),1),
            "Avg Theme (words)": round(np.mean([word_count(v.get("theme","")) for v in cache.values()]),1),
        })
    if stats_rows:
        st.markdown("**Loaded cache statistics:**")
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True)

    unique_stories = {}
    for triple in dev:
        for field in ["anchor_text","text_a","text_b"]:
            txt = triple.get(field,"")
            if txt and txt not in unique_stories:
                unique_stories[txt] = triple.get(f"{field}_title", f"Story {len(unique_stories)+1}")

    if unique_stories:
        story_options = list(unique_stories.keys())
        selected = st.selectbox("Choose a story", story_options,
                                format_func=lambda x: unique_stories.get(x, x[:80]))
        if selected:
            norm_sel = norm_text(selected)
            st.markdown(f"<div class='story-block'>{selected}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            for label, key, p_class, c_class in [
                ("Course of Action", "coa", "pill-coa", "card-accent-coa"),
                ("Outcomes",         "outcomes", "pill-out", "card-accent-out"),
                ("Theme",            "theme", "pill-thm", "card-accent-thm"),
            ]:
                st.markdown(f"<span class='pill {p_class}'>{label}</span>", unsafe_allow_html=True)
                cc1, cc2, cc3 = st.columns(3)
                for ver, col in [(1,cc1),(2,cc2),(3,cc3)]:
                    entry = caches.get(ver, {}).get(norm_sel, {})
                    with col:
                        st.markdown(f"<div style='font-size:0.68rem; font-weight:600; color:#b8913a; margin-bottom:3px;'>V{ver}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card {c_class}'><small>{entry.get(key,'- not in cache -')}</small></div>", unsafe_allow_html=True)
    else:
        st.info("Dev triplet file not found - load data to compare extractions.")

# ======================== PAGE: Aspect Informativeness ========================
elif page == "Aspect Informativeness":
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.5.6 · Chapter 4 · Section 4.5</div>", unsafe_allow_html=True)
    st.markdown("## Aspect Informativeness Analysis")
    st.markdown("""
For each development triple, the anchor and both candidates are encoded with the same encoder.
Cosine similarity is computed for anchor–candidate pairs. The metric **%pos > neg** measures how
often the gold-closer candidate has higher similarity - values above 50% indicate a discriminative signal.
Statistical significance assessed via paired *t*-test (α = 0.05).
""")

    st.markdown("### Table 3.10 · Aspect Informativeness - RoBERTa-large, 200 dev triples")
    st.dataframe(DF_ROBERTA_200, hide_index=True, use_container_width=True)

    st.markdown("### Aspect Informativeness - BGE-M3, 200 dev triples")
    st.dataframe(DF_BGEM3_200, hide_index=True, use_container_width=True)

    st.markdown("*Notes: Sig. ✓ = p < 0.05 (paired t-test). Values above 57.5 exceed full-text baseline.*")

    st.markdown("---")

    # Visual: bar chart of % pos>neg by aspect and version, RoBERTa
    rob_fig_data = []
    for _, row in DF_ROBERTA_200.iterrows():
        rob_fig_data.append({"Version": row["Version"].split(" ")[0], "Aspect": row["Aspect"], "% pos>neg": row["% pos>neg"]})
    rob_df = pd.DataFrame(rob_fig_data)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(rob_df, x="Aspect", y="% pos>neg", color="Version", barmode="group",
                     title="RoBERTa-large: % pos>neg by aspect and version",
                     color_discrete_sequence=["#0f1e35","#2a5f72","#b8913a"])
        fig.add_hline(y=57.5, line_dash="dash", line_color="#c23b2a", annotation_text="Full-text baseline (57.5%)")
        fig.add_hline(y=50.0, line_dash="dot",  line_color="#8a9db0", annotation_text="Chance (50%)")
        navy_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    bge_fig_data = []
    for _, row in DF_BGEM3_200.iterrows():
        bge_fig_data.append({"Version": row["Version"].split(" ")[0], "Aspect": row["Aspect"], "% pos>neg": row["% pos>neg"]})
    bge_df = pd.DataFrame(bge_fig_data)
    with c2:
        fig2 = px.bar(bge_df, x="Aspect", y="% pos>neg", color="Version", barmode="group",
                      title="BGE-M3: % pos>neg by aspect and version",
                      color_discrete_sequence=["#0f1e35","#2a5f72","#b8913a"])
        fig2.add_hline(y=60.5, line_dash="dash", line_color="#c23b2a", annotation_text="Full-text baseline (60.5%)")
        fig2.add_hline(y=50.0, line_dash="dot",  line_color="#8a9db0", annotation_text="Chance (50%)")
        navy_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("### Table 4 (Thesis) · Aspect Informativeness - 166 Clean Triples with Gold Aspect Labels")
    st.dataframe(DF_166_GOLD, hide_index=True, use_container_width=True)
    st.markdown("*Significance: ** p < 0.01, * p < 0.05. Only aspects with gold matching labels included.*")

    st.markdown("---")

    st.markdown("### Complementarity Analysis - Aspect-to-Full-Text Correlations")
    st.markdown("Low correlation with full text means the aspect captures *different* information - ideal for complementarity. Low mutual correlation (CoA–Outcomes) also desirable.")
    st.dataframe(DF_CORRELATIONS, hide_index=True, use_container_width=True)

    corr_fig = px.bar(DF_CORRELATIONS, x="Pair", y="r", color="Model", barmode="group",
                      title="Pearson r between aspect and full-text cosine similarities",
                      text="r",
                      color_discrete_map={"RoBERTa":"#0f1e35","BGE-M3":"#2a5f72"})
    corr_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    corr_fig.add_hline(y=0, line_color="#8a9db0", line_width=1)
    navy_fig(corr_fig, height=360)
    st.plotly_chart(corr_fig, use_container_width=True)
    st.markdown("*Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant.*")

    st.markdown("---")
    st.markdown("### Implications for Downstream Modelling")
    st.markdown("""
<div class="result-highlight">
  Based on this analysis, the downstream aspect-based model variants use:
  <strong>CoA - Version 1</strong> (narrative prose, highest discriminative power: 62.5%, p=0.0061)
  and <strong>Outcomes & Theme - Version 2</strong> (structured role-label, significant outcomes signal: p=0.0399).
  No Version 3 aspect achieves statistical significance, confirming that compactness alone does not guarantee similarity signal.
</div>
""", unsafe_allow_html=True)

# ======================== PAGE: Experimental Results ========================
elif page == "Experimental Results":
    st.markdown("<div class='thesis-eyebrow'>Chapter 5 · All Experiments</div>", unsafe_allow_html=True)
    st.markdown("## Experimental Results")
    st.markdown("All models follow the same two-stage pipeline: Stage 1 contrastive pre-training on synthetic triplets → Stage 2 supervised ranking fine-tuning on 200 dev triples. Evaluation on revealed test sets only.")

    tab1, tab2, tab3, tab4 = st.tabs(["§5.2 Input Length Ablation", "§5.3 G2 Latent-Head Model", "§5.4 Explicit Aspect Variants", "§5.5 Ensemble"])

    with tab1:
        st.markdown("### RQ1 · Input Length Ablation (Full-Text Baseline)")
        st.markdown("Encoder: `sentence-transformers/all-roberta-large-v1`. No aspects, no latent heads. Only the tokenizer max length varies.")

        st.dataframe(DF_INPUT_LENGTH, hide_index=True, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=DF_INPUT_LENGTH["max_len"], y=DF_INPUT_LENGTH["Track A Acc."],
                                  name="Track A Accuracy", mode="lines+markers",
                                  line=dict(color="#0f1e35", width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=DF_INPUT_LENGTH["max_len"], y=DF_INPUT_LENGTH["Track B Acc."],
                                  name="Track B Accuracy", mode="lines+markers",
                                  line=dict(color="#c23b2a", width=2.5), marker=dict(size=8)))
        fig.update_layout(title="Input Length Ablation - Baseline",
                          xaxis_title="Max sequence length (tokens)", yaxis_title="Accuracy (%)",
                          yaxis=dict(range=[58, 73]))
        navy_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
<div class="finding-box">
<strong>Finding (RQ1):</strong> Track B is far more sensitive to truncation than Track A.
Track B accuracy rises from 62.25% (128 tokens) to 69.25% (512 tokens) - a +7pp gain.
Track A peaks at 128 tokens (68.25%) and does not benefit monotonically from longer inputs.
Embedding-based evaluation requires each story to be encoded independently;
missing final events or resolutions degrade the standalone representation used across all comparisons.
</div>
""", unsafe_allow_html=True)
        st.markdown("<div class='baseline-note'>Configuration chosen: 384 tokens for the baseline (near-complete coverage, lower memory); 512 tokens for the final G2 model (maximum context for representation quality).</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("### RQ2 · G2 Full-Text Latent-Head Model")
        st.markdown("""
**G2 architecture:** The same RoBERTa-large encoder as the baseline, plus two generic 256-dim linear projection heads.
During Stage 1, heads receive a triplet loss signal: $\\mathcal{L}_{G2} = 0.7\\mathcal{L}^g_{triplet} + 0.3\\mathcal{L}^g_{rank} + 0.3\\mathcal{L}_{heads}$.
For Track B, the final embedding is the concatenated L2-normalised vector: $e_{G2} = \\text{norm}([g; h_1; h_2])$.
        """)

        st.markdown("#### Effect of Additional Synthetic Data (Table 5.3)")
        st.dataframe(DF_SYNTHETIC_EFFECT, hide_index=True, use_container_width=True)

        synth_fig_data = []
        for _, row in DF_SYNTHETIC_EFFECT.iterrows():
            synth_fig_data.append({"Config": f"{row['Model']}\n({row['Synthetic data']})", "Model": row["Model"], "Synth": row["Synthetic data"],
                                    "Track A": row["Track A Acc."], "Track B": row["Track B Acc."]})
        sf_df = pd.DataFrame(synth_fig_data)
        fig2a = go.Figure()
        colors_map = {"Baseline": "#2a5f72", "G2": "#0f1e35"}
        markers_map = {"Organiser only": "circle", "Organiser + extra": "diamond"}
        for _, row in DF_SYNTHETIC_EFFECT.iterrows():
            lbl = f"{row['Model']} · {row['Synthetic data']}"
            fig2a.add_trace(go.Scatter(
                x=["Track A", "Track B"], y=[row["Track A Acc."], row["Track B Acc."]],
                name=lbl, mode="lines+markers",
                line=dict(color=colors_map.get(row["Model"],"#6b4a7a"),
                           dash="solid" if "only" in row["Synthetic data"] else "dash"),
                marker=dict(size=10)))
        fig2a.update_layout(title="Effect of Additional Synthetic Data on Baseline vs. G2", yaxis_title="Accuracy (%)", yaxis=dict(range=[58,74]))
        navy_fig(fig2a, height=360)
        st.plotly_chart(fig2a, use_container_width=True)

        st.markdown("""
<div class="finding-box">
<strong>Key observation:</strong> Extra synthetic data <em>harms</em> the baseline (Track B: 69.25%→61.00%)
but substantially <em>helps</em> G2 (Track B: 64.75%→71.75%). The latent heads regularise the model,
making it robust to longer and more diverse synthetic examples generated by a different LLM (llama3.1:8b).
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Final G2 Performance vs. Best Baseline (Table 5.4)")
        st.dataframe(DF_BEST_MODELS, hide_index=True, use_container_width=True)

        best_fig = go.Figure()
        for _, row in DF_BEST_MODELS.iterrows():
            if row["Track A Acc."] == "-":
                continue
            best_fig.add_trace(go.Bar(name=row["Model"],
                                       x=["Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"],
                                       y=[float(row["Track A Acc."]), float(row["Track A F1"]),
                                          float(row["Track B Acc."]), float(row["Track B F1"])]))
        best_fig.update_layout(barmode="group", title="Best Baseline vs. G2 - All Metrics",
                                yaxis_title="Score (%)", yaxis=dict(range=[60,76]),
                                colorway=["#2a5f72","#0f1e35","#b8913a"])
        navy_fig(best_fig, height=360)
        st.plotly_chart(best_fig, use_container_width=True)

        st.markdown("""
<div class="result-highlight">
  G2 achieves <strong>71.75% Track B accuracy</strong> (+2.5pp over best baseline) and
  <strong>69.00% Track A accuracy</strong> (+0.75pp). The largest gain is on Track B,
  confirming that generic latent heads learn task-relevant narrative subspaces especially
  beneficial for standalone story embeddings.
</div>
""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### RQ3 · Explicit Aspect-Based Model Variants (Table 5.5)")
        st.markdown("All variants use combined synthetic data and max_len=512 (except P* and P at 384 due to memory). Aspect choice: CoA from V1, Outcomes & Theme from V2.")
        st.dataframe(DF_ASPECTS_PERF, hide_index=True, use_container_width=True)

        asp_fig = px.scatter(DF_ASPECTS_PERF, x="Track A Acc.", y="Track B Acc.", text="Condition",
                              color="Condition", title="Aspect variant performance: Track A vs. Track B",
                              color_discrete_sequence=PALETTE_A, size=[10]*len(DF_ASPECTS_PERF))
        asp_fig.update_traces(textposition="top center", marker=dict(size=12))
        asp_fig.add_vline(x=69.0, line_dash="dash", line_color="#b8913a", annotation_text="G2 Track A")
        asp_fig.add_hline(y=71.75, line_dash="dash", line_color="#c23b2a", annotation_text="G2 Track B")
        asp_fig.update_layout(showlegend=False, xaxis=dict(range=[58,73]), yaxis=dict(range=[57,75]))
        navy_fig(asp_fig, height=420)
        st.plotly_chart(asp_fig, use_container_width=True)

        st.markdown("""
<div class="finding-box">
<strong>Finding (RQ3):</strong> No explicit aspect variant matches G2. The best aspect-aware model
(Condition N: full text + CoA + outcomes + latent heads) reaches 70.00% Track B - 1.75pp behind G2.
Condition E (aspects only, no full text) drops to 60.25% Track B. Including Theme (Condition P) actively
<em>hurts</em> performance (59.75% Track B), confirming that theme extraction is too noisy to be
useful as a direct similarity feature. Explicit aspects are valuable for interpretability and error
diagnosis but do not improve predictive accuracy over latent representations.
</div>
""", unsafe_allow_html=True)

    with tab4:
        st.markdown("### §5.5 · Embedding Ensemble for Track B")
        st.markdown("""
**Method:** Concatenate L2-normalised embeddings from two complementary sources with square-root scaling weights.
- **G2 embeddings** (task-tuned, 1536-dim concatenation: global + 2 × 256-dim heads)
- **Qwen3-Embedding-0.6B** with All-but-the-Top post-processing (highest zero-shot Track B baseline)
- **Weights:** (0.90 G2, 0.10 Qwen) - selected via grid search on dev set over {0.95/0.05, 0.90/0.10, 0.85/0.15, 0.80/0.20}
        """)
        st.dataframe(DF_ENSEMBLE, hide_index=True, use_container_width=True)

        ens_fig = px.bar(DF_ENSEMBLE, x="Model", y="Track B Acc. (%)", text="Track B Acc. (%)",
                          title="Track B accuracy: individual models and ensemble",
                          color="Model", color_discrete_sequence=["#2a5f72","#6b4a7a","#0f1e35"])
        ens_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        ens_fig.update_layout(showlegend=False, yaxis=dict(range=[60,75]))
        navy_fig(ens_fig, height=320)
        st.plotly_chart(ens_fig, use_container_width=True)

        st.markdown("""
<div class="finding-box">
<strong>Ensemble result:</strong> 72.00% Track B accuracy - +0.25pp over G2 alone, +6pp over Qwen zero-shot.
G2 tends to over-rely on lexical overlap; Qwen's zero-shot embeddings capture more abstract thematic similarity,
making the two sources genuinely complementary. No additional training required - pure post-processing.
</div>
""", unsafe_allow_html=True)

# ======================== PAGE: Multilingual ========================
elif page == "Multilingual Comparison":
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.3 · Chapter 5 · Section 5.6</div>", unsafe_allow_html=True)
    st.markdown("## Multilingual English–Romanian Comparison")
    st.markdown("""
**RQ5:** Does the narrative similarity pipeline remain effective when stories are translated to Romanian?

The English dataset was machine-translated to Romanian using **NLLB-200** (`facebook/nllb-200-distilled-600M`)
with sentence-level chunking (max 50 words/chunk) to reduce long-generation failures.
Translation cache: **11,775 unique stories**. Flagged rate: **0.49%** (51 possible truncations, 5 repetitions, 2 ellipsis artifacts).

Two multilingual encoders are evaluated with the **same G2-style architecture** (full text, max_len=512,
two generic latent heads) to isolate the effect of language change from encoder choice.
""")

    st.markdown("### Table 5.7 · English vs. Romanian Machine-Translation Results")
    st.dataframe(DF_MULTILINGUAL, hide_index=True, use_container_width=True)

    # Performance drop table
    drop_df = pd.DataFrame([
        ["multilingual-e5-base",                 -3.75, -3.78, -4.75, -4.79],
        ["paraphrase-multilingual-mpnet-base-v2", -3.00, -3.00, -4.00, -3.95],
    ], columns=["Model", "Track A Acc. drop", "Track A F1 drop", "Track B Acc. drop", "Track B F1 drop"])
    st.markdown("### Table 5.8 · Performance Drop (English → Romanian MT)")
    st.dataframe(drop_df, hide_index=True, use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        ml_fig = go.Figure()
        colors_ml = {"multilingual-e5-base": "#0f1e35", "paraphrase-multilingual-mpnet-base-v2": "#2a5f72"}
        for model, grp in DF_MULTILINGUAL.groupby("Model"):
            lbl_short = "E5" if "e5" in model else "MPNet"
            for _, row in grp.iterrows():
                ml_fig.add_trace(go.Bar(
                    name=f"{lbl_short} ({row['Language']})",
                    x=[f"Track A - {row['Language']}", f"Track B - {row['Language']}"],
                    y=[row["Track A Acc."], row["Track B Acc."]],
                    marker_color=colors_ml[model],
                    opacity=1.0 if "English" in row["Language"] else 0.55))
        ml_fig.update_layout(barmode="group", title="Accuracy by track and language", yaxis_title="Accuracy (%)", yaxis=dict(range=[54,74]))
        navy_fig(ml_fig, height=360)
        st.plotly_chart(ml_fig, use_container_width=True)

    with c2:
        drop_fig = go.Figure()
        for _, row in drop_df.iterrows():
            lbl = "E5" if "e5" in row["Model"] else "MPNet"
            drop_fig.add_trace(go.Bar(name=lbl,
                                       x=["Track A Acc.", "Track A F1", "Track B Acc.", "Track B F1"],
                                       y=[row["Track A Acc. drop"], row["Track A F1 drop"],
                                          row["Track B Acc. drop"], row["Track B F1 drop"]],
                                       text=[f"{v:.2f}" for v in [row["Track A Acc. drop"], row["Track A F1 drop"],
                                                                    row["Track B Acc. drop"], row["Track B F1 drop"]]]))
        drop_fig.update_traces(textposition="outside")
        drop_fig.update_layout(barmode="group", title="Performance drop: EN → RO MT",
                                yaxis_title="Δ accuracy (pp)", yaxis=dict(range=[-7,0]),
                                colorway=["#0f1e35","#2a5f72"])
        navy_fig(drop_fig, height=360)
        st.plotly_chart(drop_fig, use_container_width=True)

    st.markdown("""
<div class="finding-box">
<strong>Findings (RQ5):</strong>
Both multilingual encoders preserve a substantial part of the similarity signal after machine translation,
but not all of it. <strong>E5 is more robust</strong> (Track B drop: 4.75pp vs. MPNet's 4.00pp on absolute terms;
E5 starts higher).
Track B shows a larger drop than Track A because standalone story embeddings are more sensitive to
translation-induced compression or omission of narrative details.
This experiment is a <em>robustness study under machine translation</em>, not evaluation on a native Romanian benchmark.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Translation Pipeline Details")
    tr_col1, tr_col2 = st.columns([1.2, 1])
    with tr_col1:
        tr_stats = pd.DataFrame([
            ["Model", "facebook/nllb-200-distilled-600M"],
            ["Source → Target", "eng_Latn → ron_Latn"],
            ["Chunk strategy", "Sentence-level, max 50 words/chunk"],
            ["Batch size", "16"],
            ["Max input tokens", "256"],
            ["Max new tokens", "320"],
            ["Beam size", "2"],
            ["Repetition penalty", "1.12"],
            ["Translation cache entries", "11,775"],
            ["Coverage of main EDA texts", "100.00%"],
            ["Mean RO/EN word ratio", "0.940"],
            ["Flagged translations", "58 (0.49%)"],
        ], columns=["Parameter", "Value"])
        st.dataframe(tr_stats, hide_index=True, use_container_width=True)
    with tr_col2:
        st.markdown("""
<div class="card card-parchment">
<div class="thesis-eyebrow">Quality control</div>
Translations were automatically flagged for:
<ul style="margin: 6px 0 0 0; padding-left: 1.2rem; font-size:0.85rem; line-height:1.7;">
  <li>Very low target/source length ratio (possible truncation) - <strong>51 cases</strong></li>
  <li>Repeated n-gram patterns - <strong>5 cases</strong></li>
  <li>Ellipsis artifacts - <strong>2 cases</strong></li>
</ul>
<br>
<div style="font-size:0.82rem; color:#4a5d72;">
Median RO/EN word ratio: <strong>0.947</strong> - translations are slightly shorter but cover the same narrative content in most cases.
</div>
</div>
""", unsafe_allow_html=True)

# ======================== PAGE: Story & Aspect Explorer ========================
elif page == "Translation Explorer":
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.2 · English-Romanian Dataset</div>", unsafe_allow_html=True)
    st.markdown("## Translation Explorer")
    st.markdown("Browse development-set stories in English and their Romanian translations. Compare how stories are translated while maintaining narrative meaning.")
    # ── Load English data ──
    dev = load_dev_triples()

    # ── Load Romanian data and translation cache ──
    try:
        ro_path = resolve_data_path("narrative_nlp/dataset/romanian_narrative_similarity_dataset/dev_track_a_ro.jsonl", "dev_track_a_ro.jsonl")
        with open(ro_path, "r", encoding="utf-8") as f:
            dev_ro = [json.loads(line.strip()) for line in f if line.strip()]
        
        trans_cache_path = resolve_data_path("narrative_nlp/dataset/romanian_narrative_similarity_dataset/translation_cache_en_ro.json", "translation_cache_en_ro.json")
        with open(trans_cache_path, "r", encoding="utf-8") as f:
            trans_cache = json.load(f)
    except Exception as e:
        st.warning(f"Could not load Romanian data: {e}")
        dev_ro = []
        trans_cache = {}

    # ── Build story pairs (EN + RO) ──
    stories = {}
    for i, t_en in enumerate(dev):
        for field in ["anchor_text","text_a","text_b"]:
            text_en = t_en.get(field, "").strip()
            if not text_en or text_en in stories:
                continue
            
            # Look up Romanian translation
            text_ro = trans_cache.get(text_en, "")
            
            # If not in cache, try to match by index/field from dev_ro
            if not text_ro and i < len(dev_ro):
                t_ro = dev_ro[i]
                text_ro = t_ro.get(field, "").strip()
            
            title = t_en.get(f"{field}_title", "")
            
            stories[text_en] = {
                "title": title or "",
                "text_en": text_en,
                "text_ro": text_ro,
                "words_en": word_count(text_en),
                "words_ro": word_count(text_ro) if text_ro else 0,
                "has_translation": bool(text_ro),
            }

    story_list = list(stories.values())

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([2.5, 1, 1])
    with fc1:
        search = st.text_input("Search by title or English text", placeholder="keyword…")
    with fc2:
        filter_trans = st.selectbox("Translation", ["All", "With translation", "Missing translation"])
    with fc3:
        per_page = st.selectbox("Per page", [5, 10, 20], index=1)

    filtered = story_list
    if search:
        filtered = [s for s in filtered if search.lower() in s["title"].lower() or search.lower() in s["text_en"].lower()]
    if filter_trans == "With translation":
        filtered = [s for s in filtered if s["has_translation"]]
    elif filter_trans == "Missing translation":
        filtered = [s for s in filtered if not s["has_translation"]]

    st.markdown(f"<div class='section-label'>{len(filtered)} stories match</div>", unsafe_allow_html=True)

    total_pages = max(1, (len(filtered)+per_page-1)//per_page)
    page_num = st.number_input("Page", 1, total_pages, 1, step=1)
    page_stories = filtered[(page_num-1)*per_page: page_num*per_page]

    for s in page_stories:
        title_part = f"**{s['title']}** - " if s['title'] else ""
        trans_badge = "<span class='pill' style='background:#3d6b58; color:#faf7f2;'>RO ✓</span>" if s["has_translation"] else "<span class='pill' style='background:#c23b2a; color:#faf7f2;'>RO ✗</span>"
        
        with st.expander(f"{s['text_en'][:85]}…  ({s['words_en']} EN | {s['words_ro']} RO words)", expanded=False):
            st.markdown(f"{title_part}{trans_badge}", unsafe_allow_html=True)
            
            # ── Side-by-side comparison ──
            col_en, col_ro = st.columns(2, gap="large")
            
            with col_en:
                st.markdown("<div style='font-weight:600; color:#0f1e35; margin-bottom:0.5rem;'>EN English</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='story-block'>{s['text_en']}</div>", unsafe_allow_html=True)
            
            with col_ro:
                st.markdown("<div style='font-weight:600; color:#0f1e35; margin-bottom:0.5rem;'>🇷🇴 Română</div>", unsafe_allow_html=True)
                if s["has_translation"]:
                    st.markdown(f"<div class='story-block'>{s['text_ro']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='baseline-note'><em>Translation not available</em></div>", unsafe_allow_html=True)

    # ── Flagged Translations ──
    st.markdown("---")
    st.markdown("### Flagged Translations · Translation Quality Report")
    st.markdown("""
    Every NLLB-200 translation was automatically screened for three failure modes: very short/truncated
    output (target/source word-ratio &lt; 0.6), repeated n-gram patterns, and ellipsis artifacts left over
    from chunked generation. The table below lists every flagged case from the **11,775**-entry translation
    cache, alongside the full source/target texts for inspection.
    """, unsafe_allow_html=True)

    try:
        quality_path = resolve_data_path("narrative_nlp/dataset/romanian_narrative_similarity_dataset/translation_quality_report.json", "translation_quality_report.json")
        with open(quality_path, "r", encoding="utf-8") as f:
            quality_report = json.load(f)
    except Exception as e:
        st.warning(f"Could not load translation quality report: {e}")
        quality_report = {}

    # ── Load full translation cache + English source texts for full-text lookup ──
    try:
        trans_cache_path = resolve_data_path("narrative_nlp/dataset/romanian_narrative_similarity_dataset/translation_cache_en_ro.json", "translation_cache_en_ro.json")
        with open(trans_cache_path, "r", encoding="utf-8") as f:
            full_trans_cache = json.load(f)
    except Exception:
        full_trans_cache = {}

    @st.cache_data
    def build_flagged_lookup(_cache: dict):
        """Map normalised preview-prefix -> (full_en, full_ro) for fast matching."""
        lookup = []
        for en_text, ro_text in _cache.items():
            lookup.append((norm_text(en_text), en_text, ro_text))
        return lookup

    flagged_lookup = build_flagged_lookup(full_trans_cache) if full_trans_cache else []

    def find_full_translation(source_preview: str, target_preview: str):
        """Find the full EN/RO pair whose EN text starts with source_preview.
        Falls back to matching on target_preview prefix if EN match fails.
        Returns (full_en, full_ro) or (None, None) if not found."""
        if not flagged_lookup:
            return None, None
        norm_src = norm_text(source_preview)
        # Strip a trailing partial word (preview is often cut mid-word)
        norm_src_trimmed = norm_src.rsplit(" ", 1)[0] if " " in norm_src else norm_src

        for norm_en, full_en, full_ro in flagged_lookup:
            if norm_en.startswith(norm_src_trimmed):
                return full_en, full_ro

        # Fallback: match on the Romanian target preview prefix
        norm_tgt = norm_text(target_preview)
        norm_tgt_trimmed = norm_tgt.rsplit(" ", 1)[0] if " " in norm_tgt else norm_tgt
        for norm_en, full_en, full_ro in flagged_lookup:
            if norm_text(full_ro).startswith(norm_tgt_trimmed):
                return full_en, full_ro

        return None, None

    if quality_report:
        total = quality_report.get("total_translations", 0)
        flagged_count = quality_report.get("flagged_count", 0)
        flagged_rate = quality_report.get("flagged_rate", 0) * 100
        flag_counts = quality_report.get("flag_counts", {})
        length_ratio = quality_report.get("length_ratio", {})
        flagged_examples = quality_report.get("flagged_examples", [])

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.markdown(f"""<div class="metric-box">
        <div class="metric-val">{total:,}</div>
        <div class="metric-label">Total translations</div>
        </div>""", unsafe_allow_html=True)
        fc2.markdown(f"""<div class="metric-box">
        <div class="metric-val metric-val-accent">{flagged_count}</div>
        <div class="metric-label">Flagged cases</div>
        </div>""", unsafe_allow_html=True)
        fc3.markdown(f"""<div class="metric-box">
        <div class="metric-val">{flagged_rate:.2f}%</div>
        <div class="metric-label">Flagged rate</div>
        </div>""", unsafe_allow_html=True)
        fc4.markdown(f"""<div class="metric-box">
        <div class="metric-val">{length_ratio.get('median', 0):.3f}</div>
        <div class="metric-label">Median RO/EN<br>word ratio (all)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            flag_df = pd.DataFrame({
                "Flag type": ["Too short / truncated", "Repetition", "Ellipsis artifact"],
                "Count": [
                    flag_counts.get("too_short_or_truncated", 0),
                    flag_counts.get("repetition", 0),
                    flag_counts.get("ellipsis_artifact", 0),
                ],
            })
            fig_flags = px.bar(flag_df, x="Flag type", y="Count", text="Count",
                            title="Flagged cases by type",
                            color="Flag type",
                            color_discrete_map={
                                "Too short / truncated": "#c23b2a",
                                "Repetition": "#b8913a",
                                "Ellipsis artifact": "#6b4a7a",
                            })
            fig_flags.update_traces(textposition="outside")
            fig_flags.update_layout(showlegend=False)
            navy_fig(fig_flags, height=320)
            st.plotly_chart(fig_flags, use_container_width=True)

        with fcol2:
            ratios = [ex.get("length_ratio", 0) for ex in flagged_examples]
            if ratios:
                ratio_df = pd.DataFrame({"Length ratio (target/source words)": ratios})
                fig_ratio = px.histogram(ratio_df, x="Length ratio (target/source words)", nbins=20,
                                        title="Length-ratio distribution among flagged cases",
                                        color_discrete_sequence=["#2a5f72"])
                fig_ratio.add_vline(x=0.6, line_dash="dash", line_color="#c23b2a",
                                    annotation_text="too-short threshold (0.6)")
                navy_fig(fig_ratio, height=320)
                st.plotly_chart(fig_ratio, use_container_width=True)

        st.markdown(f"""<div class='baseline-note'>
        Across the full 11,775-entry cache, the length-ratio (target words ÷ source words) ranges from
        {length_ratio.get('min', 0):.3f} to {length_ratio.get('max', 0):.3f}, with a median of
        {length_ratio.get('median', 0):.3f} — translations are slightly more compact than their English
        source on average, consistent with Romanian's tendency toward shorter renderings of the same content.
        The {flagged_count} flagged cases ({flagged_rate:.2f}%) sit well outside this typical range.
        </div>""", unsafe_allow_html=True)

        if not full_trans_cache:
            st.markdown("<div class='baseline-note'>Note: translation cache not found - showing truncated previews from the quality report only.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Flagged Cases - Source / Target Comparison")

        flag_type_options = ["All", "Too short / truncated", "Repetition", "Ellipsis artifact"]
        flag_key_map = {
            "Too short / truncated": "too_short_or_truncated",
            "Repetition": "repetition",
            "Ellipsis artifact": "ellipsis_artifact",
        }

        sel_col1, sel_col2 = st.columns([1, 1])
        with sel_col1:
            selected_flag = st.selectbox("Filter by flag type", flag_type_options, key="flag_filter")
        with sel_col2:
            sort_order = st.selectbox("Sort by length ratio", ["Lowest first", "Highest first"], key="flag_sort")

        filtered_examples = flagged_examples
        if selected_flag != "All":
            key = flag_key_map[selected_flag]
            filtered_examples = [ex for ex in flagged_examples if key in ex.get("flags", [])]

        filtered_examples = sorted(
            filtered_examples,
            key=lambda ex: ex.get("length_ratio", 0),
            reverse=(sort_order == "Highest first"),
        )

        st.markdown(f"<div class='section-label'>{len(filtered_examples)} flagged case(s) match</div>", unsafe_allow_html=True)

        for i, ex in enumerate(filtered_examples, 1):
            flags = ex.get("flags", [])
            ratio = ex.get("length_ratio", 0)
            src_words = ex.get("source_words", 0)
            tgt_words = ex.get("target_words", 0)
            src_preview = ex.get("source_preview", "")
            tgt_preview = ex.get("target_preview", "")

            full_en, full_ro = find_full_translation(src_preview, tgt_preview)
            display_en = full_en if full_en else src_preview
            display_ro = full_ro if full_ro else tgt_preview
            is_truncated_display = full_en is None

            badge_map = {
                "too_short_or_truncated": "<span class='pill pill-miss'>Too short / truncated</span>",
                "repetition": "<span class='pill pill-gold'>Repetition</span>",
                "ellipsis_artifact": "<span class='pill pill-thm'>Ellipsis artifact</span>",
            }
            badges = " ".join(badge_map.get(f, f"<span class='pill'>{f}</span>") for f in flags)

            header = f"#{i:02d} - ratio {ratio:.3f} ({src_words} EN → {tgt_words} RO words)"
            with st.expander(header, expanded=False):
                st.markdown(badges, unsafe_allow_html=True)
                if is_truncated_display:
                    st.markdown("<div class='baseline-note'>Full text not found in translation cache - showing preview only.</div>", unsafe_allow_html=True)
                cmp_col1, cmp_col2 = st.columns(2, gap="large")
                with cmp_col1:
                    st.markdown("<div style='font-weight:600; color:#0f1e35; margin-bottom:0.5rem;'>EN Source</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='story-block'>{display_en}</div>", unsafe_allow_html=True)
                with cmp_col2:
                    st.markdown("<div style='font-weight:600; color:#0f1e35; margin-bottom:0.5rem;'>RO Target</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='story-block'>{display_ro}</div>", unsafe_allow_html=True)
    else:
        st.info("Translation quality report not found - load `translation_quality_report.json` to see flagged cases.")

# ======================== PAGE: Prediction Browser ========================
elif page == "Prediction Browser":
    st.markdown("<div class='thesis-eyebrow'>Chapter 5 · Section 5.4 · G2 Condition</div>", unsafe_allow_html=True)
    st.markdown("## Prediction Browser · G2 Test-Set Results")
    st.markdown(
        "Browse the Track A test triplets side by side with the G2 model's predictions. "
        "Each row shows the anchor, both candidates, the gold label, and whether G2 was correct. "
        "Version 3 aspect extractions are shown for quick interpretability inspection."
    )

    # ── File paths ──
    PRED_A_PATH  = DATA_PATHS.get("pred_track_a",
        resolve_data_path("narrative_nlp/G2_condition_predictions/Condition_G2_512_full_train_track_a.jsonl",
                          "Condition_G2_512_full_train_track_a.jsonl"))
    GOLD_A_PATH  = DATA_PATHS.get("test_a_labels",
        resolve_data_path("narrative_nlp/dataset/test_track_a_labels.jsonl",
                          "test_track_a_labels.jsonl"))
    TEST_A_PATH  = DATA_PATHS["test_a"]
    ASP_V3_PATH  = DATA_PATHS.get("aspects_v3")

    # ── Loaders ──
    @st.cache_data
    def load_pred_browser_data():
        """Load test triples, gold labels, G2 predictions, and V3 aspect cache.
        Returns (rows, aspects_v3) where rows is a list of dicts with all fields merged."""
        # Test triples
        if not TEST_A_PATH or not Path(TEST_A_PATH).exists():
            return [], {}
        with open(TEST_A_PATH, encoding="utf-8") as f:
            test_rows = [json.loads(l) for l in f if l.strip()]

        # Gold labels (keyed by anchor+text_a+text_b triple)
        gold_map = {}
        if Path(GOLD_A_PATH).exists():
            with open(GOLD_A_PATH, encoding="utf-8") as f:
                for l in f:
                    if not l.strip(): continue
                    obj = json.loads(l)
                    key = (norm_text(obj.get("anchor_text","")),
                           norm_text(obj.get("text_a","")),
                           norm_text(obj.get("text_b","")))
                    gold_map[key] = bool(obj.get("text_a_is_closer"))

        # G2 predictions (keyed the same way)
        pred_map = {}
        if Path(PRED_A_PATH).exists():
            with open(PRED_A_PATH, encoding="utf-8") as f:
                for l in f:
                    if not l.strip(): continue
                    obj = json.loads(l)
                    key = (norm_text(obj.get("anchor_text","")),
                           norm_text(obj.get("text_a","")),
                           norm_text(obj.get("text_b","")))
                    pred_map[key] = bool(obj.get("text_a_is_closer"))

        # V3 aspect cache
        asp_v3 = {}
        if ASP_V3_PATH and Path(ASP_V3_PATH).exists():
            with open(ASP_V3_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            asp_v3 = {norm_text(k): v for k, v in raw.items()}

        # Merge
        rows = []
        for obj in test_rows:
            an = obj.get("anchor_text","").strip()
            ta = obj.get("text_a","").strip()
            tb = obj.get("text_b","").strip()
            if not an: continue
            key = (norm_text(an), norm_text(ta), norm_text(tb))
            gold = gold_map.get(key)
            pred = pred_map.get(key)
            correct = (gold == pred) if (gold is not None and pred is not None) else None
            rows.append({
                "anchor": an,
                "text_a": ta,
                "text_b": tb,
                "gold_a_closer": gold,
                "pred_a_closer": pred,
                "correct": correct,
            })
        return rows, asp_v3

    all_rows, asp_v3 = load_pred_browser_data()

    def get_v3(text):
        """Return V3 aspect dict for a story text, or empty strings."""
        entry = asp_v3.get(norm_text(text), {})
        return {
            "coa":      entry.get("coa","") or "—",
            "outcomes": entry.get("outcomes","") or "—",
            "theme":    entry.get("theme","") or "—",
        }

    has_preds  = any(r["pred_a_closer"] is not None for r in all_rows)
    has_gold   = any(r["gold_a_closer"] is not None for r in all_rows)
    has_aspects = bool(asp_v3)

    # ── Status banners ──
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if not all_rows:
            st.warning(f"Test triplets not found at:\n`{TEST_A_PATH}`")
        else:
            st.markdown(f"""<div class="metric-box">
              <div class="metric-val" style="font-size:1.9rem;">{len(all_rows)}</div>
              <div class="metric-label">Track A test triplets</div>
            </div>""", unsafe_allow_html=True)
    with col_s2:
        if not has_preds:
            st.warning(f"G2 predictions not found at:\n`{PRED_A_PATH}`")
        else:
            n_correct = sum(1 for r in all_rows if r["correct"] is True)
            acc = n_correct / len(all_rows) * 100 if all_rows else 0
            st.markdown(f"""<div class="metric-box">
              <div class="metric-val" style="font-size:1.9rem;">{acc:.1f}%</div>
              <div class="metric-label">G2 Track A accuracy<br>({n_correct}/{len(all_rows)} correct)</div>
            </div>""", unsafe_allow_html=True)
    with col_s3:
        if not has_aspects:
            st.markdown(f"""<div class="card card-accent-gold" style="font-size:0.82rem; line-height:1.6;">
              <strong>V3 aspects not loaded.</strong><br>
              Expected: <code>aspects_cache_v3.json</code><br>
              Aspect columns will show "—".
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-box">
              <div class="metric-val" style="font-size:1.9rem;">{len(asp_v3):,}</div>
              <div class="metric-label">Stories in V3 aspect cache</div>
            </div>""", unsafe_allow_html=True)

    if not all_rows:
        st.stop()

    st.markdown("---")

    # ── Summary charts (only when predictions exist) ──
    if has_preds and has_gold:
        n_correct   = sum(1 for r in all_rows if r["correct"] is True)
        n_wrong     = sum(1 for r in all_rows if r["correct"] is False)
        n_unknown   = sum(1 for r in all_rows if r["correct"] is None)

        ch1, ch2 = st.columns(2)
        with ch1:
            breakdown_df = pd.DataFrame({
                "Outcome": ["Correct", "Wrong", "No label"],
                "Count":   [n_correct, n_wrong, n_unknown],
            })
            fig_bd = px.bar(breakdown_df, x="Outcome", y="Count", text="Count",
                            title="G2 prediction outcomes (Track A test set)",
                            color="Outcome",
                            color_discrete_map={"Correct":"#3d6b58","Wrong":"#c23b2a","No label":"#8a9db0"})
            fig_bd.update_traces(textposition="outside")
            fig_bd.update_layout(showlegend=False)
            navy_fig(fig_bd, height=300)
            st.plotly_chart(fig_bd, use_container_width=True)

        with ch2:
            # Gold label distribution vs. predicted
            gold_a = sum(1 for r in all_rows if r["gold_a_closer"] is True)
            gold_b = sum(1 for r in all_rows if r["gold_a_closer"] is False)
            pred_a = sum(1 for r in all_rows if r["pred_a_closer"] is True)
            pred_b = sum(1 for r in all_rows if r["pred_a_closer"] is False)
            dist_df = pd.DataFrame({
                "Source": ["Gold", "Gold", "G2 pred.", "G2 pred."],
                "Label":  ["text_a closer", "text_b closer", "text_a closer", "text_b closer"],
                "Count":  [gold_a, gold_b, pred_a, pred_b],
            })
            fig_dist = px.bar(dist_df, x="Source", y="Count", color="Label", barmode="group",
                              text="Count",
                              title="Gold vs. predicted label distribution",
                              color_discrete_map={"text_a closer":"#2a5f72","text_b closer":"#b8913a"})
            fig_dist.update_traces(textposition="outside")
            navy_fig(fig_dist, height=300)
            st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Filters ──
    st.markdown("#### Filter and search")
    fl1, fl2, fl3, fl4 = st.columns([2.5, 1.2, 1.2, 0.8])
    with fl1:
        search_q = st.text_input("Search anchor / candidate text", placeholder="keyword…", key="pb_search")
    with fl2:
        outcome_filter = st.selectbox(
            "Prediction outcome",
            ["All", "Correct ✓", "Wrong ✗", "No label"],
            key="pb_outcome"
        )
    with fl3:
        gold_filter = st.selectbox(
            "Gold label",
            ["All", "text_a is closer", "text_b is closer"],
            key="pb_gold"
        )
    with fl4:
        per_page = st.selectbox("Per page", [5, 10, 20], index=1, key="pb_per_page")

    # ── Apply filters ──
    filtered = all_rows
    if search_q:
        q = search_q.lower()
        filtered = [r for r in filtered if q in r["anchor"].lower()
                    or q in r["text_a"].lower() or q in r["text_b"].lower()]
    if outcome_filter == "Correct ✓":
        filtered = [r for r in filtered if r["correct"] is True]
    elif outcome_filter == "Wrong ✗":
        filtered = [r for r in filtered if r["correct"] is False]
    elif outcome_filter == "No label":
        filtered = [r for r in filtered if r["correct"] is None]
    if gold_filter == "text_a is closer":
        filtered = [r for r in filtered if r["gold_a_closer"] is True]
    elif gold_filter == "text_b is closer":
        filtered = [r for r in filtered if r["gold_a_closer"] is False]

    total_pages = max(1, (len(filtered) + per_page - 1) // per_page)
    st.markdown(f"<div class='section-label'>{len(filtered)} triplets match</div>", unsafe_allow_html=True)

    page_num = st.number_input("Page", 1, total_pages, 1, step=1, key="pb_page")
    page_rows = filtered[(page_num - 1) * per_page : page_num * per_page]

    st.markdown("---")

    # ── Row renderer ──
    def outcome_badge(row):
        if row["correct"] is True:
            return "<span class='pill pill-match'>✓ Correct</span>"
        if row["correct"] is False:
            return "<span class='pill pill-miss'>✗ Wrong</span>"
        return "<span class='pill pill-gold'>? No label</span>"

    def gold_badge(a_closer):
        if a_closer is True:
            return "<span class='pill pill-coa'>Gold: text A</span>"
        if a_closer is False:
            return "<span class='pill pill-out'>Gold: text B</span>"
        return "<span class='pill'>Gold: ?</span>"

    def pred_badge(a_closer):
        if a_closer is True:
            return "<span class='pill' style='background:#e8f0f8;color:#1a3a5c;border:1px solid #b0c4dc;'>Pred: text A</span>"
        if a_closer is False:
            return "<span class='pill' style='background:#f5ede0;color:#5c3a1a;border:1px solid #dcc4a0;'>Pred: text B</span>"
        return "<span class='pill'>Pred: ?</span>"

    def border_color(row):
        if row["correct"] is True:  return "#3d6b58"
        if row["correct"] is False: return "#c23b2a"
        return "#b8913a"

    for idx, row in enumerate(page_rows, start=(page_num - 1) * per_page + 1):
        bc = border_color(row)
        header_badges = (
            f"{outcome_badge(row)} "
            f"{gold_badge(row['gold_a_closer'])} "
            f"{pred_badge(row['pred_a_closer'])}"
        )
        preview = row["anchor"][:90] + ("…" if len(row["anchor"]) > 90 else "")
        with st.expander(f"#{idx:03d} · {preview}", expanded=False):
            st.markdown(header_badges, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Three-column story layout ──
            c_anchor, c_a, c_b = st.columns(3, gap="small")

            with c_anchor:
                st.markdown("<div class='section-label'>Anchor</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='story-block'>{row['anchor']}</div>", unsafe_allow_html=True)
                # V3 for anchor
                if has_aspects:
                    asp = get_v3(row["anchor"])
                    st.markdown("<div style='margin-top:0.6rem;'><span class='pill pill-coa'>CoA</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card card-accent-coa' style='font-size:0.8rem;'>{asp['coa']}</div>", unsafe_allow_html=True)
                    st.markdown("<span class='pill pill-out'>Outcomes</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card card-accent-out' style='font-size:0.8rem;'>{asp['outcomes']}</div>", unsafe_allow_html=True)
                    st.markdown("<span class='pill pill-thm'>Theme</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card card-accent-thm' style='font-size:0.8rem;'>{asp['theme']}</div>", unsafe_allow_html=True)

            # Determine which candidate is gold-correct and which G2 picked
            a_is_gold = row["gold_a_closer"]
            a_is_pred = row["pred_a_closer"]

            for col, text, is_gold, is_pred, label in [
                (c_a, row["text_a"], a_is_gold is True, a_is_pred is True, "Candidate A"),
                (c_b, row["text_b"], a_is_gold is False, a_is_pred is False, "Candidate B"),
            ]:
                with col:
                    # Build header indicators
                    indicators = []
                    if is_gold:
                        indicators.append("<span class='pill pill-match'>✓ Gold</span>")
                    if is_pred:
                        indicators.append("<span class='pill' style='background:#e8f0f8;color:#1a3a5c;border:1px solid #b0c4dc;'>G2 pick</span>")
                    # Story border reflects correctness of this candidate
                    if is_gold and is_pred:
                        left_col = "#3d6b58"  # correctly picked
                    elif is_pred and not is_gold:
                        left_col = "#c23b2a"  # G2 picked this but wrong
                    elif is_gold and not is_pred:
                        left_col = "#b8913a"  # gold but not picked
                    else:
                        left_col = "#8a9db0"  # neither gold nor picked

                    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)
                    if indicators:
                        st.markdown(" ".join(indicators), unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='story-block' style='border-left-color:{left_col};'>{text}</div>",
                        unsafe_allow_html=True
                    )
                    # V3 for candidate
                    if has_aspects:
                        asp = get_v3(text)
                        st.markdown("<div style='margin-top:0.6rem;'><span class='pill pill-coa'>CoA</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card card-accent-coa' style='font-size:0.8rem;'>{asp['coa']}</div>", unsafe_allow_html=True)
                        st.markdown("<span class='pill pill-out'>Outcomes</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card card-accent-out' style='font-size:0.8rem;'>{asp['outcomes']}</div>", unsafe_allow_html=True)
                        st.markdown("<span class='pill pill-thm'>Theme</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card card-accent-thm' style='font-size:0.8rem;'>{asp['theme']}</div>", unsafe_allow_html=True)
# ======================== PAGE: Live Aspect Extraction ========================
elif page == "Live Aspect Extraction":
    import urllib.request, urllib.error, concurrent.futures, time

    DEFAULT_HOST = os.getenv("OLLAMA_HOST","http://localhost:11434").rstrip("/")
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL","llama3.1:8b")
    ASPECT_KEYS = ("coa","outcomes","theme")

    # ── Prompts (same as thesis) ──
    V1_PROMPTS = {
        "coa": """You are a narrative analyst. Read the story summary below and write ONLY the sequence of plot events - what happens, in what order, and what causes what. Do NOT mention character names, specific locations, or themes. Do NOT write any introduction, heading, or label before your answer. Begin your response immediately with the first event. Write 2-4 sentences.\n\nStory:\n{story}\n\nResponse:""",
        "outcomes": """You are a narrative analyst. Read the story summary below and write ONLY the final outcome and resolution. What is the end state? What did the protagonist ultimately achieve, lose, or experience? Do NOT describe how they got there. Do NOT write any introduction, heading, or label. Begin your response immediately with the outcome. Write 1-2 sentences.\n\nStory:\n{story}\n\nResponse:""",
        "theme": """You are a narrative analyst. Read the story summary below and write ONLY the abstract themes and universal human experiences it explores. What fundamental aspects of human nature, society, or morality does it examine? Do NOT mention specific characters, places, or plot events. Do NOT write any introduction, heading, or label. Begin your response immediately with the theme. Write 1-3 sentences.\n\nStory:\n{story}\n\nResponse:""",
    }

    V2_SYS = ("You are a precise narrative analyst specialising in story structure. You follow instructions exactly. You always output valid JSON with no markdown fences, no extra keys, and no text outside the JSON object.")
    V2_PROMPT = """Analyse the story and extract three narrative aspects. Return ONLY valid JSON with keys: "coa", "outcomes", "theme". No extra text.\n\nCOA: 3-6 numbered steps, abstract action types, role labels.\nOUTCOMES: Exactly 2 sentences - final state + resolution label (conflict_resolved/unresolved/partial).\nTHEME: 2-4 short abstract phrases separated by semicolons.\n\nStory: {story}\n\nOutput:"""

    V3_SYS = ("You are a precise narrative analyst. Follow instructions exactly. Output valid JSON only with keys coa, outcomes, theme.")
    V3_PROMPT = """Analyse the story. Return ONLY valid JSON with keys "coa", "outcomes", "theme". No extra text.\n\nCOA: 3-5 short clauses separated by \" ; \". Use role labels (protagonist, antagonist, authority, ally). No character names.\nOUTCOMES: Exactly 2 cautious sentences about final state. Second sentence = one of: \"conflict resolved.\" / \"conflict unresolved.\" / \"conflict partially resolved.\"\nTHEME: 2-4 short lower-case phrases separated by semicolons.\n\nStory: {story}\n\nOutput:"""

    PREAMBLE_RE = re.compile(r"^(?:here (?:is|are)(?: the| a)?[^\n:]{0,80}?[\s:.-]*\n+|(?:certainly|sure|of course|absolutely)[!,.]?[^\n]*\n*|(?:course of action|outcomes?|abstract theme|response|answer)\s*[:]\s*\n*)", re.IGNORECASE)

    def fill(template, story): return template.replace("{story}", story.strip())
    def clean(text):
        if not text: return ""
        text = text.strip()
        for _ in range(4):
            c = PREAMBLE_RE.sub("", text).strip()
            if c == text: break
            text = c
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def call_ollama(host, model, prompt, max_tokens=500, json_mode=False, temperature=0.1):
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_tokens, "temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.1}}
        if json_mode: payload["format"] = "json"
        req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode()).get("response","").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {host}. Start with 'ollama serve'.") from e

    def parse_json(raw):
        text = re.sub(r"^```(?:json)?\s*","",raw.strip(),flags=re.IGNORECASE)
        text = re.sub(r"\s*```$","",text).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj,dict) and all(k in obj for k in ASPECT_KEYS): return obj
        except: pass
        recovered = {}
        for k in ASPECT_KEYS:
            m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            if m: recovered[k] = m.group(1).replace("\\n","\n").replace('\\"','"')
        if len(recovered)==3: return recovered
        raise ValueError(f"Could not parse JSON. Raw: {raw[:300]}")

    def extract_v1(story, host, model):
        return {k: clean(call_ollama(host, model, fill(V1_PROMPTS[k], story), 220, False, 0.2)) for k in ASPECT_KEYS}

    def extract_v2(story, host, model):
        raw = call_ollama(host, model, V2_SYS+"\n\n"+fill(V2_PROMPT,story), 500, True, 0.1)
        parsed = parse_json(raw)
        return {k: clean(parsed.get(k,"")) for k in ASPECT_KEYS}

    def postprocess_coa(text):
        text = clean(text)
        text = re.sub(r"\s*\n\s*"," ",text)
        text = re.sub(r"(?m)^\s*(\d+)[.)]\s*","",text)
        text = re.sub(r"(?i)\bstep\s+\d+\s*[:.-]\s*","",text)
        text = text.replace("\u2192","; ")
        text = re.sub(r"\s*(?:->)\s*","; ",text)
        text = re.sub(r"\s-\s","; ",text)
        text = re.sub(r"\s*;\s*","; ",text)
        return re.sub(r"\s{2,}"," ",text).strip(" ;")

    def extract_v3(story, host, model):
        raw = call_ollama(host, model, V3_SYS+"\n\n"+fill(V3_PROMPT,story), 500, True, 0.0)
        parsed = parse_json(raw)
        return {"coa": postprocess_coa(parsed.get("coa","")), "outcomes": clean(parsed.get("outcomes","")), "theme": clean(parsed.get("theme",""))}

    EXTRACTORS = {"v1": extract_v1, "v2": extract_v2, "v3": extract_v3}

    def ollama_connected(host):
        try:
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1): return True
        except: return False

    def get_executor():
        if "executor_live" not in st.session_state:
            st.session_state["executor_live"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return st.session_state["executor_live"]

    def job_running():
        job = st.session_state.get("live_job")
        return bool(job and not job["future"].done())

    def start_job(version, story, host, model):
        if job_running():
            st.session_state["live_error"] = "Already running."
            return
        st.session_state["live_result"] = None
        future = get_executor().submit(EXTRACTORS[version], story, host, model)
        st.session_state["live_job"] = {"future": future, "version": version, "model": model, "started_at": time.time()}
        st.session_state["live_error"] = None

    def update_job():
        job = st.session_state.get("live_job")
        if not job: return False
        future = job["future"]
        if not future.done(): return True
        try:
            result = future.result()
            st.session_state["live_error"] = None
            st.session_state["live_result"] = {"version": job["version"], "model": job["model"], "aspects": result}
        except Exception as e:
            st.session_state["live_error"] = str(e)
        st.session_state["live_job"] = None
        return False

    # Init state
    for k, v in [("live_result",None),("live_error",None),("live_job",None),("live_story",""),("live_mode","V1 - Verbose Prose")]:
        st.session_state.setdefault(k, v)

    update_job()

    # ── Page ──
    st.markdown("<div class='thesis-eyebrow'>Chapter 3 · Section 3.4 · RQ6</div>", unsafe_allow_html=True)
    st.markdown("## Live Aspect Extraction")
    st.markdown("Extract **course of action**, **outcomes**, and **abstract theme** from any story summary using a local LLM. Demonstrates the three extraction strategies developed in the thesis (§3.4.1).")

    EXAMPLE = ("A young inventor discovers a hidden city beneath the desert after decoding a map left by her missing father. She enters the city with a reluctant guide, uncovers a machine that controls the region's water supply, and is forced to choose between restoring water to nearby villages or preserving the city's ancient secrecy. After sabotaging the machine's lock system, she escapes as water returns to the surface, but the city is exposed to the outside world.")

    MODE_MAP = {"V1 - Verbose Prose": "v1", "V2 - Role-Label Steps": "v2", "V3 - Compact Phrases": "v3"}
    MODE_DESC = {
        "V1 - Verbose Prose": "Separate per-aspect prompts. Rich narrative prose. Best CoA discriminative signal (62.5%, p=0.0061).",
        "V2 - Role-Label Steps": "Single combined JSON prompt. Numbered steps with role labels. Best outcomes signal (p=0.0399).",
        "V3 - Compact Phrases": "Highly constrained JSON. Semicolon-separated clauses. Cleanest output - no aspect reaches significance.",
    }

    inp_col, cfg_col = st.columns([0.62, 0.38], gap="large")

    with inp_col:
        with st.container(border=True):
            st.markdown("#### Story summary")
            bc1, bc2 = st.columns(2)
            if bc1.button("Load example story"):
                st.session_state["live_story"] = EXAMPLE
            if bc2.button("Clear"):
                st.session_state["live_story"] = ""
                st.session_state["live_result"] = None
                st.session_state["live_error"] = None

            story = st.text_area("Input story", key="live_story", height=280, label_visibility="collapsed",
                                  placeholder="Paste a story summary, chapter outline, or synopsis here…")
            st.caption(f"{len(story)} characters")

    with cfg_col:
        with st.container(border=True):
            st.markdown("#### Extraction settings")
            mode = st.radio("Extraction version", list(MODE_MAP.keys()), key="live_mode")
            st.markdown(f"<div class='card card-accent-gold' style='font-size:0.83rem; margin-top:0;'>{MODE_DESC[mode]}</div>", unsafe_allow_html=True)
            requested = None
            if st.button("Extract narrative aspects", type="primary", use_container_width=True,
                         disabled=job_running() or not story.strip()):
                requested = MODE_MAP[mode]
            st.markdown("---")
            st.markdown("#### Model settings")
            host = st.text_input("Ollama host", DEFAULT_HOST)
            model = st.text_input("Model", DEFAULT_MODEL)

            connected = ollama_connected(host.strip())
            status_color = "#3d6b58" if connected else "#c23b2a"
            status_text = "● Connected" if connected else "● Not connected"
            st.markdown(f"<div style='font-size:0.82rem; font-weight:600; color:{status_color};'>{status_text}</div>", unsafe_allow_html=True)
            if not connected:
                with st.expander("Setup"):
                    st.code(f"ollama pull {model or DEFAULT_MODEL}", language="bash")
                    st.code("ollama serve", language="bash")

    if requested:
        if not story.strip():
            st.warning("Paste a story summary first.")
        else:
            start_job(requested, story.strip(), host.strip() or DEFAULT_HOST, model.strip() or DEFAULT_MODEL)

    was = job_running()
    still = update_job()
    if was and not still: 
        st.rerun()

    with st.container(border=True):
        st.markdown("#### Extraction output")
        if still:
            job = st.session_state["live_job"]
            elapsed = int(time.time()-job["started_at"])
            st.info(f"Extracting with {job['version'].upper()}… {elapsed}s elapsed. Running Llama 3.1 8B locally.")
            time.sleep(0.8)
            st.rerun()
        if st.session_state.get("live_error"):
            st.error(f"Extraction failed: {st.session_state['live_error']}")
        result = st.session_state.get("live_result")
        if result:
            st.success(f"✓ Extraction complete - {result['version'].upper()} | model: {result['model']}")
            aspects = result["aspects"]
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("<span class='pill pill-coa'>Course of Action</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='card card-accent-coa'>{aspects.get('coa','-')}</div>", unsafe_allow_html=True)
            with r2:
                st.markdown("<span class='pill pill-out'>Outcomes</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='card card-accent-out'>{aspects.get('outcomes','-')}</div>", unsafe_allow_html=True)
            with r3:
                st.markdown("<span class='pill pill-thm'>Abstract Theme</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='card card-accent-thm'>{aspects.get('theme','-')}</div>", unsafe_allow_html=True)
            st.download_button("Export JSON", data=json.dumps(result, indent=2, ensure_ascii=False),
                               file_name=f"aspects_{result['version']}.json", mime="application/json")
            with st.expander("Raw JSON"):
                st.json(result)
        elif not still and not st.session_state.get("live_error"):
            st.markdown("<div class='baseline-note'>Paste a story and click <em>Extract narrative aspects</em> to see results.</div>", unsafe_allow_html=True)