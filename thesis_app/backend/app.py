"""
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
    page_title="Narrative Similarity · Thesis Demo",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== DESIGN TOKENS (LIGHT GRAY THEME) ========================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@300;400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');

:root {
    --bg:        #f5f7fa;
    --surface:   #ffffff;
    --border:    #e0e4e8;
    --accent:    #c8a96e;
    --accent2:   #6e9ec8;
    --accent3:   #c86e9e;
    --text:      #1a2a3a;
    --muted:     #5a6a7a;
    --success:   #2e7d64;
    --danger:    #b73c2c;
    --coa:       #3a6ea5;
    --out:       #b88b4a;
    --theme:     #8a5ea8;
}

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    color: var(--text);
}
.card-light {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    color: #333333;
}
.card-light-blue {
    background: #e6f2ff;
    border: 1px solid #b3d9ff;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    color: #003d99;
}
.card-accent-coa  { border-left: 3px solid var(--coa);   }
.card-accent-out  { border-left: 3px solid var(--out);   }
.card-accent-thm  { border-left: 3px solid var(--theme); }
.card-accent-gold { border-left: 3px solid var(--accent);}

.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 4px;
}
.pill-coa   { background: #e0ecf9; color: var(--coa);   border: 1px solid var(--coa);   }
.pill-out   { background: #f9ede0; color: var(--out);   border: 1px solid var(--out);   }
.pill-thm   { background: #f0e6f4; color: var(--theme); border: 1px solid var(--theme); }
.pill-match { background: #dff0e8; color: var(--success);border: 1px solid var(--success);}
.pill-miss  { background: #fbe9e7; color: var(--danger); border: 1px solid var(--danger); }

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

[data-testid="stDataFrame"] { background: var(--surface) !important; }
.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8em; }

hr { border-color: var(--border) !important; }

.story-block {
    background: #e6f2ff;
    border: 1px solid #b3d9ff;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88em;
    line-height: 1.7;
    color: #003d99;
}

.badge-correct { color: var(--success); font-weight: 600; }
.badge-wrong   { color: var(--danger);  font-weight: 600; }

.baseline-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: var(--muted);
    margin-top: 4px;
}

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ======================== PATH RESOLUTION ========================
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent  # typical repo structure

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
    "dev":         resolve_data_path("narrative_nlp/dataset/dev_track_a.jsonl", "dev_track_a.jsonl"),
    "dev_labels":  resolve_data_path("narrative_nlp/dataset/dev_track_a_labels.jsonl", "dev_track_a_labels.jsonl"),
    "aspects_v1":  resolve_data_path("narrative_nlp/dataset/aspects_cache_v1.json", "aspects_cache_v1.json"),
    "aspects_v2":  resolve_data_path("narrative_nlp/dataset/aspects_cache_v2.json", "aspects_cache_v2.json"),
    "aspects_v3":  resolve_data_path("narrative_nlp/dataset/aspects_cache_v3.json", "aspects_cache_v3.json"),
    "test_a":      resolve_data_path("narrative_nlp/dataset/test_track_a.jsonl", "test_track_a.jsonl"),
    "test_b":      resolve_data_path("narrative_nlp/dataset/test_track_b.jsonl", "test_track_b.jsonl"),
    "synth":       resolve_data_path("narrative_nlp/dataset/synthetic_data_for_classification.jsonl", "synthetic_data_for_classification.jsonl"),
    "synth_new":   resolve_data_path("narrative_nlp/dataset/synthetic_data_new.jsonl", "synthetic_data_new.jsonl"),
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
def load_dev_triples():
    return load_jsonl_file(DATA_PATHS["dev"])

@st.cache_data
def load_test_a_rows():
    return load_jsonl_file(DATA_PATHS["test_a"])

@st.cache_data
def load_test_b_rows():
    return load_jsonl_file(DATA_PATHS["test_b"])

@st.cache_data
def load_synth_rows():
    return load_jsonl_file(DATA_PATHS["synth"])

@st.cache_data
def load_synth_new_rows():
    return load_jsonl_file(DATA_PATHS["synth_new"])

@st.cache_data
def load_dev_labels():
    return load_jsonl_file(DATA_PATHS["dev_labels"])

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
    return {
        1: load_aspects_cache(1),
        2: load_aspects_cache(2),
        3: load_aspects_cache(3),
    }

# ======================== HELPER DATAFRAMES ========================
@st.cache_data
def build_dataset_summary():
    dev = load_dev_triples()
    test_a = load_test_a_rows()
    test_b = load_test_b_rows()
    synth = load_synth_rows()
    synth_new = load_synth_new_rows()
    def split_summary(name, rows, kind="triples", labeled=False):
        if kind == "triples":
            texts = []
            for r in rows:
                for field in ["anchor_text", "text_a", "text_b"]:
                    if field in r:
                        texts.append(r[field])
            unique_texts = len(set(texts))
            avg_words = np.mean([word_count(t) for t in texts]) if texts else 0
            out = {
                "Split": name,
                "Rows": len(rows),
                "Unique texts": unique_texts,
                "Avg words / text": round(float(avg_words), 1),
                "Labeled": "Yes" if labeled else "No",
            }
            if labeled and rows:
                pos = sum(bool(r.get("text_a_is_closer")) for r in rows)
                neg = len(rows) - pos
                out["Positive"] = pos
                out["Negative"] = neg
            return out
        texts = [r["text"] for r in rows if "text" in r]
        avg_words = np.mean([word_count(t) for t in texts]) if texts else 0
        return {
            "Split": name,
            "Rows": len(rows),
            "Unique texts": len(set(texts)),
            "Avg words / text": round(float(avg_words), 1),
            "Labeled": "No",
        }
    rows = [
        split_summary("Dev Track A", dev, labeled=True),
        split_summary("Test Track A", test_a, labeled=False),
        split_summary("Test Track B", test_b, kind="single_text"),
        split_summary("Synthetic v1", synth, labeled=True),
        split_summary("Synthetic v2", synth_new, labeled=True),
    ]
    return pd.DataFrame(rows)

@st.cache_data
def build_dev_text_length_df():
    dev = load_dev_triples()
    records = []
    for i, row in enumerate(dev):
        for field, label in [("anchor_text", "Anchor"), ("text_a", "Text A"), ("text_b", "Text B")]:
            text = row.get(field, "")
            records.append({
                "triple_id": i,
                "field": label,
                "words": word_count(text),
                "chars": char_count(text),
            })
    return pd.DataFrame(records)

@st.cache_data
def build_unique_story_df():
    dev = load_dev_triples()
    aspects_v3 = load_aspects_cache(3)
    unique_texts = []
    seen = set()
    for row in dev:
        for field in ["anchor_text", "text_a", "text_b"]:
            txt = row.get(field, "")
            if txt and txt not in seen:
                seen.add(txt)
                entry = aspects_v3.get(norm_text(txt), {})
                unique_texts.append({
                    "text": txt,
                    "words": word_count(txt),
                    "chars": char_count(txt),
                    "coa_words": word_count(entry.get("coa", "")),
                    "outcomes_words": word_count(entry.get("outcomes", "")),
                    "theme_words": word_count(entry.get("theme", "")),
                    "resolution_status": entry.get("resolution_status", "unknown") or "unknown",
                    "theme": entry.get("theme", ""),
                })
    return pd.DataFrame(unique_texts)

@st.cache_data
def build_theme_frequency_df(top_n: int = 15):
    df = build_unique_story_df()
    counter = Counter()
    for theme in df["theme"].fillna(""):
        parts = [p.strip().lower() for p in re.split(r"[;,]", theme) if p.strip()]
        for part in parts:
            counter[part] += 1
    top = counter.most_common(top_n)
    return pd.DataFrame(top, columns=["theme_phrase", "count"])

@st.cache_data
def build_resolution_df():
    df = build_unique_story_df()
    counts = df["resolution_status"].value_counts().reset_index()
    counts.columns = ["resolution_status", "count"]
    return counts

@st.cache_data
def build_aspect_length_df():
    aspects = load_aspects_cache(3)
    records = []
    for value in aspects.values():
        records.extend([
            {"aspect": "CoA", "words": word_count(value.get("coa", ""))},
            {"aspect": "Outcomes", "words": word_count(value.get("outcomes", ""))},
            {"aspect": "Theme", "words": word_count(value.get("theme", ""))},
        ])
    return pd.DataFrame(records)

@st.cache_data
def build_synth_new_metadata():
    rows = load_synth_new_rows()
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    gen_df = pd.DataFrame(Counter(r.get("generation_type", "unknown") for r in rows).most_common(),
                          columns=["generation_type", "count"])
    genre_df = pd.DataFrame(Counter(r.get("seed_genre", "unknown") for r in rows).most_common(12),
                            columns=["seed_genre", "count"])
    return gen_df, genre_df

@st.cache_data
def build_dev_label_analysis():
    labels = load_dev_labels()
    if not labels:
        return pd.DataFrame()
    
    records = []
    for row in labels:
        coa_match_a = row.get("course_of_actions", [False, False])[0]
        coa_match_b = row.get("course_of_actions", [False, False])[1]
        out_match_a = row.get("outcomes", [False, False])[0]
        out_match_b = row.get("outcomes", [False, False])[1]
        theme_match_a = row.get("abstract_theme", [False, False])[0]
        theme_match_b = row.get("abstract_theme", [False, False])[1]
        human_a = row.get("human_labels", [False, False])[0]
        human_b = row.get("human_labels", [False, False])[1]
        
        records.append({
            "coa_match_a": coa_match_a,
            "coa_match_b": coa_match_b,
            "outcomes_match_a": out_match_a,
            "outcomes_match_b": out_match_b,
            "theme_match_a": theme_match_a,
            "theme_match_b": theme_match_b,
            "human_label_a": human_a,
            "human_label_b": human_b,
        })
    return pd.DataFrame(records)

@st.cache_data
def build_synth_model_stats():
    synth = load_synth_rows()
    if not synth:
        return pd.DataFrame()
    model_counts = Counter(r.get("model_name", "unknown") for r in synth)
    df = pd.DataFrame(model_counts.most_common(), columns=["Model", "Stories Generated"])
    return df

# ======================== THESIS TABLES (precomputed) ========================
# Table 4.2: RoBERTa-large, 200 dev triples
DF_ROBERTA_200 = pd.DataFrame([
    ["V1 (narrative prose)", "Full text", 57.5, 0.0271, 0.0069, "yes"],
    ["V1 (narrative prose)", "CoA", 62.5, 0.0270, 0.0061, "yes"],
    ["V1 (narrative prose)", "Outcomes", 51.5, 0.0137, 0.0805, "no"],
    ["V1 (narrative prose)", "Theme", 51.0, 0.0013, 0.8888, "no"],
    ["V2 (role-label steps)", "Full text", 57.5, 0.0271, 0.0069, "yes"],
    ["V2 (role-label steps)", "CoA", 55.0, 0.0178, 0.0197, "yes"],
    ["V2 (role-label steps)", "Outcomes", 57.0, 0.0174, 0.0399, "yes"],
    ["V2 (role-label steps)", "Theme", 53.5, 0.0140, 0.1962, "no"],
    ["V3 (compact phrases)", "Full text", 57.5, 0.0271, 0.0069, "yes"],
    ["V3 (compact phrases)", "CoA", 53.5, 0.0114, 0.2439, "no"],
    ["V3 (compact phrases)", "Outcomes", 54.5, 0.0174, 0.1749, "no"],
    ["V3 (compact phrases)", "Theme", 56.5, 0.0146, 0.2168, "no"],
], columns=["Version", "Aspect", "% pos>neg", "Mean diff.", "p-value", "Sig."])

# Table 4.3: BGE-M3, 200 dev triples
DF_BGEM3_200 = pd.DataFrame([
    ["V1 (narrative prose)", "Full text", 60.5, 0.0092, 0.0020, "yes"],
    ["V1 (narrative prose)", "CoA", 59.5, 0.0067, 0.0364, "yes"],
    ["V1 (narrative prose)", "Outcomes", 57.0, 0.0071, 0.0125, "yes"],
    ["V1 (narrative prose)", "Theme", 53.5, 0.0020, 0.3993, "no"],
    ["V2 (role-label steps)", "Full text", 60.5, 0.0092, 0.0020, "yes"],
    ["V2 (role-label steps)", "CoA", 63.0, 0.0110, 0.0007, "yes"],
    ["V2 (role-label steps)", "Outcomes", 54.0, 0.0076, 0.0191, "yes"],
    ["V2 (role-label steps)", "Theme", 52.5, 0.0084, 0.0548, "no"],
    ["V3 (compact phrases)", "Full text", 60.5, 0.0092, 0.0020, "yes"],
    ["V3 (compact phrases)", "CoA", 57.5, 0.0068, 0.0273, "yes"],
    ["V3 (compact phrases)", "Outcomes", 57.0, 0.0054, 0.2797, "no"],
    ["V3 (compact phrases)", "Theme", 55.5, 0.0094, 0.0248, "yes"],
], columns=["Version", "Aspect", "% pos>neg", "Mean diff.", "p-value", "Sig."])

# Table 4.4: 166 clean triples with gold aspect labels
DF_166_GOLD = pd.DataFrame([
    ["RoBERTa", "V1", "CoA", 63.3, 0.0308, 0.221, "[+0.0096, +0.0523]", 0.0049, "**"],
    ["RoBERTa", "V2", "Outcomes", 57.8, 0.0194, 0.160, "[+0.0012, +0.0382]", 0.0408, "*"],
    ["BGE-M3",  "V2", "CoA", 64.5, 0.0107, 0.236, "[+0.0039, +0.0176]", 0.0028, "**"],
    ["BGE-M3",  "V1", "Outcomes", 58.4, 0.0075, 0.186, "[+0.0014, +0.0135]", 0.0177, "*"],
    ["BGE-M3",  "V3", "Theme", 56.6, 0.0117, 0.199, "[+0.0026, +0.0205]", 0.0111, "*"],
], columns=["Model", "Ver.", "Aspect", "%pos>neg", "Diff.", "Cohen's d", "95% CI", "p", "Sig."])

# Table 4.5: Complementarity correlations
DF_CORRELATIONS = pd.DataFrame([
    ["RoBERTa", "V1 (narrative prose)", "CoA–Full", 0.2719, 0.0004, "***"],
    ["RoBERTa", "V1 (narrative prose)", "Outcomes–Full", 0.3943, 0.0000, "***"],
    ["RoBERTa", "V1 (narrative prose)", "CoA–Outcomes", 0.2356, 0.0022, "**"],
    ["BGE-M3",  "V2 (role-label steps)", "CoA–Full", 0.2057, 0.0078, "**"],
    ["BGE-M3",  "V2 (role-label steps)", "Outcomes–Full", 0.0486, 0.5338, "n.s."],
    ["BGE-M3",  "V2 (role-label steps)", "CoA–Outcomes", 0.2120, 0.0061, "**"],
    ["BGE-M3",  "V3 (compact phrases)", "Theme–Full", 0.0267, 0.7328, "n.s."],
], columns=["Model", "Version", "Pair", "r", "p-value", "Sig."])

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("""
<div style='padding: 0.5rem 0 1.2rem 0;'>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.65rem;
              text-transform: uppercase; letter-spacing: 0.12em; color: #5a6a7a;
              margin-bottom: 4px;'>Master Thesis · 2026</div>
  <div style='font-family: Playfair Display, serif; font-size: 1.3rem;
              font-weight: 700; line-height: 1.2; color: #1a2a3a;'>
    Narrative<br>Similarity
  </div>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem;
              color: #5a6a7a; margin-top: 8px;'>
    Alexandru Ioan Cuza University<br>Iași · Computational Linguistics MSc
  </div>
</div>
<hr style='margin: 0.5rem 0 1rem 0;'>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🗂 Dataset",
         "🔬 Compare Extraction Versions",
         "⚡ Real-time Aspect Extraction",
         "📈 Aspect Informativeness",
         "🔍 Aspect Explorer",
         "📊 Aspect-Aware Model Variants"],
        label_visibility="collapsed",
    )
    page = page.split(" ", 1)[1]

# ======================== PAGE: Dataset ========================
if page == "Dataset":
    st.markdown("## 🗂 Dataset")
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    dev = load_dev_triples()
    test_a = load_test_a_rows()
    test_b = load_test_b_rows()
    synth = load_synth_rows()

    if not dev:
        st.warning("`dev_track_a.jsonl` not found.")
        st.stop()

    # ----- Helper functions -----
    def compute_dataset_stats(name, rows, is_ranking=False):
        if is_ranking:
            texts = []
            for r in rows:
                texts.extend([r.get("anchor_text", ""), r.get("text_a", ""), r.get("text_b", "")])
            texts = [t for t in texts if t]
            labels = Counter(bool(r.get("text_a_is_closer")) for r in rows)
        else:
            texts = [r.get("text", "") for r in rows if r.get("text", "")]
            labels = None

        unique_texts = len(set(texts))
        word_lengths = [word_count(t) for t in texts]
        char_lengths = [char_count(t) for t in texts]

        stats = {
            "Dataset": name,
            "Rows": len(rows),
            "Unique Texts": unique_texts,
            "Mean words": round(np.mean(word_lengths), 1) if word_lengths else 0,
            "Std words": round(float(np.std(word_lengths)), 1) if len(word_lengths) > 1 else 0,
            "Min words": int(np.min(word_lengths)) if word_lengths else 0,
            "Max words": int(np.max(word_lengths)) if word_lengths else 0,
        }
        if labels:
            stats["Pos : Neg"] = f"{labels[True]} : {labels[False]}"
        return stats, word_lengths, char_lengths

    dev_stats, dev_words, _ = compute_dataset_stats("Development (Track A)", dev, is_ranking=True)
    test_a_stats, test_a_words, _ = compute_dataset_stats("Test (Track A)", test_a, is_ranking=True)
    test_b_stats, test_b_words, _ = compute_dataset_stats("Test (Track B)", test_b, is_ranking=False)
    synth_stats, synth_words, _ = compute_dataset_stats("Synthetic (Classification)", synth, is_ranking=True)

    # ----- Quick overview metrics -----
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"<div class='metric-box'><div class='metric-val'>{len(dev)}</div><div class='metric-label'>Dev triples</div></div>", unsafe_allow_html=True)
    with col_m2:
        n_unique = len(set([t for row in dev for t in [row.get("anchor_text",""), row.get("text_a",""), row.get("text_b","")] if t]))
        st.markdown(f"<div class='metric-box'><div class='metric-val'>{n_unique}</div><div class='metric-label'>Unique dev stories</div></div>", unsafe_allow_html=True)
    with col_m3:
        avg_words_dev = np.mean([word_count(t) for row in dev for t in [row.get("anchor_text",""), row.get("text_a",""), row.get("text_b","")] if t])
        st.markdown(f"<div class='metric-box'><div class='metric-val'>{avg_words_dev:.0f}</div><div class='metric-label'>Avg words / story</div></div>", unsafe_allow_html=True)
    with col_m4:
        pos_neg = f"{sum(row.get('text_a_is_closer', False) for row in dev)} : {sum(not row.get('text_a_is_closer', True) for row in dev)}"
        st.markdown(f"<div class='metric-box'><div class='metric-val'>{pos_neg}</div><div class='metric-label'>Pos : Neg (dev)</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ----- Corpus statistics table -----
    st.markdown("### Corpus Statistics")
    summary_table = pd.DataFrame([dev_stats, test_a_stats, test_b_stats, synth_stats])
    st.dataframe(summary_table, width="stretch", hide_index=True, use_container_width=True)

    st.markdown("---")

    # ----- Text length analysis -----
    st.markdown("### Text Length Distribution")
    col_dist1, col_dist2 = st.columns(2)
    
    data_for_box = []
    for words, name in [(dev_words, "Dev"), (test_a_words, "Test A"), (test_b_words, "Test B"), (synth_words, "Synth")]:
        for w in words:
            data_for_box.append({"Dataset": name, "Words": w})
    box_df = pd.DataFrame(data_for_box)
    
    with col_dist1:
        fig = px.box(box_df, x="Dataset", y="Words", category_orders={"Dataset": ["Dev", "Test A", "Test B", "Synth"]})
        fig.update_layout(title=dict(text="Word‑count distribution by dataset", font=dict(family="Playfair Display", size=14, color="#1a2a3a")), 
                         xaxis_title="", yaxis_title="Word count",
                         paper_bgcolor="#f5f7fa", plot_bgcolor="#f5f7fa", font=dict(family="JetBrains Mono", color="#1a2a3a"), 
                         margin=dict(l=10, r=10, t=50, b=30), height=350, showlegend=False)
        fig.update_xaxes(gridcolor="#e0e4e8")
        fig.update_yaxes(gridcolor="#e0e4e8")
        st.plotly_chart(fig, width="stretch", use_container_width=True)
    
    with col_dist2:
        fig = px.histogram(box_df, x="Words", color="Dataset", nbins=30, barmode="overlay", 
                          category_orders={"Dataset": ["Dev", "Test A", "Test B", "Synth"]},     color_discrete_map={
        "Dev": None,          # keep default blue
        "Test A": None,       # keep default orange
        "Test B": None,  # green (replaces pink)
        "Synth": "#2ca02c"         # keep default red
    })
        fig.update_traces(opacity=0.65)
        fig.update_layout(title=dict(text="Word‑length frequency distribution", font=dict(family="Playfair Display", size=14, color="#1a2a3a")), 
                         xaxis_title="Word count", yaxis_title="Frequency",
                         paper_bgcolor="#f5f7fa", plot_bgcolor="#f5f7fa", font=dict(family="JetBrains Mono", color="#1a2a3a"), 
                         margin=dict(l=10, r=10, t=50, b=30), height=350, showlegend=True)
        fig.update_xaxes(gridcolor="#e0e4e8")
        fig.update_yaxes(gridcolor="#e0e4e8")
        fig.update_layout(legend=dict(x=0.65, y=0.95))
        st.plotly_chart(fig, width="stretch", use_container_width=True)

    st.markdown("---")

    # ----- Dataset composition (textual description) -----
    st.markdown("### Dataset Composition")
    with st.container():
        st.markdown(
            f"""
            <div class='card' style='background: #ffffff;'>
            <ul style='margin-bottom: 0;'>
            <li><strong>Development Track A</strong> – {len(dev)} ranking triples, perfectly balanced labels. Each triple contains an anchor narrative and two candidates with binary similarity judgments. Contains {n_unique} unique stories across anchor/candidate slots.</li>
            <li><strong>Test Track A</strong> – {len(test_a)} unlabeled instances, same ranking triple format for zero‑shot and few‑shot evaluation.</li>
            <li><strong>Test Track B</strong> – {len(test_b)} single‑text format instances, enabling cross‑distribution robustness assessment.</li>
            <li><strong>Synthetic Corpus</strong> – {len(synth)} artificially generated ranking pairs for data augmentation, created with multiple LLM backends.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ----- Development set annotation analysis -----
    st.markdown("### Development Set Annotation Analysis")
    label_df = build_dev_label_analysis()
    if not label_df.empty:
        coa_agree = (label_df["coa_match_a"] == label_df["coa_match_b"]).sum() / len(label_df) * 100
        out_agree = (label_df["outcomes_match_a"] == label_df["outcomes_match_b"]).sum() / len(label_df) * 100
        theme_agree = (label_df["theme_match_a"] == label_df["theme_match_b"]).sum() / len(label_df) * 100
        
        col_ann1, col_ann2, col_ann3 = st.columns(3)
        for col, val, label in [
            (col_ann1, f"{coa_agree:.1f}%", "CoA agreement across candidates"),
            (col_ann2, f"{out_agree:.1f}%", "Outcomes agreement across candidates"),
            (col_ann3, f"{theme_agree:.1f}%", "Theme agreement across candidates"),
        ]:
            col.markdown(f"<div class='metric-box'><div class='metric-val' style='font-size:2rem'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            agree_df = pd.DataFrame({
                "Aspect": ["CoA", "Outcomes", "Theme"],
                "Agreement %": [coa_agree, out_agree, theme_agree]
            })
            fig = px.bar(agree_df, x="Aspect", y="Agreement %", text="Agreement %", range_y=[0, 100])
            fig.update_traces(marker_color="#c8a96e", textposition="outside", texttemplate="%{text:.1f}%")
            fig.update_layout(title=dict(text="Aspect matching consistency (A vs B)", font=dict(family="Playfair Display", size=14, color="#1a2a3a")), 
                             xaxis_title="", yaxis_title="Agreement (%)",
                             paper_bgcolor="#f5f7fa", plot_bgcolor="#f5f7fa", font=dict(family="JetBrains Mono", color="#1a2a3a"), 
                             margin=dict(l=10, r=10, t=50, b=30), height=330, showlegend=False)
            fig.update_xaxes(gridcolor="#e0e4e8")
            fig.update_yaxes(gridcolor="#e0e4e8")
            st.plotly_chart(fig, width="stretch", use_container_width=True)
        
        with col_chart2:
            coa_match_rate = (label_df["coa_match_a"].sum() + label_df["coa_match_b"].sum()) / (len(label_df) * 2) * 100
            out_match_rate = (label_df["outcomes_match_a"].sum() + label_df["outcomes_match_b"].sum()) / (len(label_df) * 2) * 100
            theme_match_rate = (label_df["theme_match_a"].sum() + label_df["theme_match_b"].sum()) / (len(label_df) * 2) * 100
            match_df = pd.DataFrame({
                "Aspect": ["CoA", "Outcomes", "Theme"],
                "Match Rate %": [coa_match_rate, out_match_rate, theme_match_rate]
            })
            fig = px.bar(match_df, x="Aspect", y="Match Rate %", text="Match Rate %", range_y=[0, 100])
            fig.update_traces(marker_color="#6e9ec8", textposition="outside", texttemplate="%{text:.1f}%")
            fig.update_layout(title=dict(text="Aspect matching frequency (overall)", font=dict(family="Playfair Display", size=14, color="#1a2a3a")), 
                             xaxis_title="", yaxis_title="Match Rate (%)",
                             paper_bgcolor="#f5f7fa", plot_bgcolor="#f5f7fa", font=dict(family="JetBrains Mono", color="#1a2a3a"), 
                             margin=dict(l=10, r=10, t=50, b=30), height=330, showlegend=False)
            fig.update_xaxes(gridcolor="#e0e4e8")
            fig.update_yaxes(gridcolor="#e0e4e8")
            st.plotly_chart(fig, width="stretch", use_container_width=True)
        
        st.markdown("<div class='baseline-note'>Agreement measures how often both candidates share the same aspect label (both match or both do not match the anchor). Match rate is the frequency with which a candidate is labelled as matching the anchor for that aspect.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ----- Synthetic data generation models -----
    st.markdown("### Synthetic Data: LLM Generation Models")
    synth_model_df = build_synth_model_stats()
    if not synth_model_df.empty:
        col_mod1, col_mod2 = st.columns([1, 1.5])
        with col_mod1:
            st.markdown("#### Generation Models Summary")
            st.dataframe(synth_model_df, width="stretch", hide_index=True, use_container_width=True)
            total_synth = synth_model_df["Stories Generated"].sum()
            st.markdown(f"<div class='baseline-note'><b>Total synthetic stories:</b> {total_synth} &nbsp;|&nbsp; <b>Models used:</b> {len(synth_model_df)}</div>", unsafe_allow_html=True)
        with col_mod2:
            st.markdown("#### Notes on Synthetic Data")
            st.markdown("""
            <div class='card' style='background: #ffffff;'>
            <ul>
            <li>Synthetic pairs are generated using a contrastive prompting strategy.</li>
            <li>Positive pairs are summaries of the same story (Wikidata‑linked); negative pairs are from different stories.</li>
            <li>Multiple LLMs (including Llama, Mistral, GPT variants) contribute to increase diversity.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No synthetic model generation data available.")

# ======================== PAGE: Ablation Results ========================
elif page == "Aspect-Aware Model Variants":
    st.markdown("## 📊 Experimental Results")
    st.markdown("<div class='section-title'>Aspect‑Aware Residual Fusion Model & Integration Strategies</div>", unsafe_allow_html=True)

    # ----- 5. Aspect-Aware Residual Fusion Model -----
    st.markdown("### Aspect‑Aware Residual Fusion Model")
    st.markdown("""
    **Fusion strategies** – Attention‑based, Gated, and Concatenation – were evaluated across single‑aspect,
    two‑aspect, and three‑aspect combinations. All models used a maximum sequence length of 384 tokens
    and the two‑stage contrastive training protocol.
    """)

    # Table 5.1: Residual fusion ablation (placeholder values '—')
    fusion_data = [
        # Single aspect
        ("action (CoA) only", "attention", 384, 69.00, 64.25),
        # ("action (CoA) only", "gated", 384, "—", "—"),
        # ("action (CoA) only", "concat", 384, "—", "—"),
        ("outcome only", "attention", 384, 66.50, 66.49),
        # ("outcome only", "gated", 384, "—", "—"),
        # ("outcome only", "concat", 384, "—", "—"),
        # Two aspects
        ("action + outcome", "attention", 384, 69.50, 68.50),
        # ("action + outcome", "gated", 384, "—", "—"),
        # ("action + outcome", "concat", 384, "—", "—"),
        # Three aspects
        ("action + outcome + theme", "attention", 384, 68, 65.50),
        ("action + outcome + theme", "gated", 384, 65.50, 63),
        ("action + outcome + theme", "concat", 384, 66.75, 65),
    ]
    df_fusion = pd.DataFrame(fusion_data, columns=["Aspects Used", "Fusion Type", "Max Len", "Acc. A (%)", "Acc. B (%)"])
    st.dataframe(df_fusion, width="stretch", hide_index=True)
    st.markdown("*Table: Performance of aspect‑aware residual fusion models (results pending).*")

    st.markdown("---")

    # ----- 6. Aspect Integration as Heads, Inputs, and Multi-View Conditions -----
    st.markdown("### Aspect Integration as Heads, Inputs, and Multi‑View Conditions")
    st.markdown("""
    We compared three integration strategies: **aspect heads** (additional classification heads on top of the full‑text encoder),
    **appended input** (aspect text concatenated to the raw summary), and **multi‑view** (separate forward passes for each aspect
    with soft‑label fusion). All models were trained with Stage 1 on synthetic triplets (full text only) and Stage 2 fine‑tuning
    on the development set using the configured aspect‑aware behaviour. Max sequence length = 384.
    """)
    # Table 6.2: Stage 2 only aspect‑aware behaviour (from thesis)
    stage2_data = [
        ("J", "Full‑text encoder with CoA head only", 68.50, 68.43, 65.50, 65.46),
        ("K", "Full‑text encoder with outcomes head only", 68.75, 68.74, 65.50, 65.46),
        ("L", "Full‑text encoder with theme head only", 65.25, 65.24, 66.50, 66.49),
        ("M", "Full‑text encoder with CoA and outcomes heads", 67.00, 66.97, 68.75, 68.72),
        ("H", "Full text + appended CoA and outcomes", 69.25, 69.21, 69.50, 69.49),
        ("I", "Full text + appended CoA, outcomes, and theme", 67.75, 67.67, 68.75, 68.74),
        ("N", "CoA+Outcomes heads + appended CoA and outcomes", 70.00, 69.98, 65.00, 64.94),
        ("O", "Multi‑view (soft labels in Stage 2)", 60.50, 60.48, 59.50, 59.49),
        ("P", "Multi‑view (separate full/CoA/outcomes/theme forwards)", 62.75, 62.75, 63.75, 63.67),
    ]
    df_stage2 = pd.DataFrame(stage2_data, columns=["Cond.", "Description", "Acc A (%)", "Macro-F1 A (%)", "Acc B (%)", "Macro-F1 B (%)"])
    df_stage2 = df_stage2.fillna("—")
    st.dataframe(df_stage2, width="stretch", hide_index=True)
    st.markdown("*Table: Conditions with Stage 1 full‑text only and Stage 2 aspect‑aware behaviour.*")

# ======================== PAGE: Aspect Extraction (V1/V2/V3) ========================
elif page == "Compare Extraction Versions":
    st.markdown("## 🔬 Compare Extraction Versions")
    st.markdown("<div class='section-title'>Three prompt strategies compared</div>", unsafe_allow_html=True)

    descriptions = {
        1: "Version 1 – Narrative Prose  \nSeparate per-aspect LLM calls, unrestricted prose. Retains rich detail but can be verbose. Best discriminative signal for CoA.",
        2: "Version 2 – Role-Label Steps  \nSingle combined JSON prompt; numbered steps with role labels (protagonist, authority). Better structural abstraction.",
        3: "Version 3 – Compact Phrases  \nHighly constrained prompt returning semicolon-separated phrases. Cleanest output, but loses some similarity signal."
    }
    for ver, desc in descriptions.items():
        st.markdown(
            f"<div class='card' style='background: #ffffff;'>{desc}</div>",
            unsafe_allow_html=True,
        )

    caches = load_all_aspect_caches()
    dev = load_dev_triples()
    stats_rows = []
    for ver, cache in caches.items():
        if not cache:
            continue
        n_stories = len(cache)
        avg_coa_len = np.mean([word_count(v.get("coa","")) for v in cache.values()])
        avg_out_len = np.mean([word_count(v.get("outcomes","")) for v in cache.values()])
        avg_thm_len = np.mean([word_count(v.get("theme","")) for v in cache.values()])
        stats_rows.append({
            "Version": f"V{ver}",
            "Avg CoA words": round(avg_coa_len,1),
            "Avg Outcomes words": round(avg_out_len,1),
            "Avg Theme words": round(avg_thm_len,1),
        })
    st.markdown("#### Cache Statistics")
    st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### Compare Extractions for a Specific Story")
    unique_stories = {}
    for triple in dev:
        for field in ["anchor_text", "text_a", "text_b"]:
            txt = triple.get(field, "")
            if txt and txt not in unique_stories:
                unique_stories[txt] = triple.get(f"{field}_title", f"Story {len(unique_stories)+1}")
    story_options = list(unique_stories.keys())
    selected_text = st.selectbox("Choose a story summary", story_options, format_func=lambda x: unique_stories.get(x, x[:80]))
    if selected_text:
        norm_sel = norm_text(selected_text)
        header_cols = st.columns(3)
        for ver, col in zip([1, 2, 3], header_cols):
            with col:
                st.markdown(f"**Version {ver}**")

        aspect_rows = [
            ("CoA", "coa", "card-accent-coa", "pill-coa"),
            ("Outcomes", "outcomes", "card-accent-out", "pill-out"),
            ("Theme", "theme", "card-accent-thm", "pill-thm"),
        ]
        for label, key, card_class, pill_class in aspect_rows:
            cols = st.columns(3)
            for ver, col in zip([1, 2, 3], cols):
                entry = caches.get(ver, {}).get(norm_sel, {})
                with col:
                    st.markdown(f"<span class='pill {pill_class}'>{label}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card {card_class}'><small>{entry.get(key, '—')}</small></div>",
                        unsafe_allow_html=True,
                    )

# ======================== PAGE: Real-time Aspect Extraction ========================
elif page == "Real-time Aspect Extraction":
    import urllib.request
    import urllib.error
    import concurrent.futures
    import time
    
    # ========== EXTRACT ASPECTS APP INTEGRATION ==========
    
    DEFAULT_HOST_RT = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    DEFAULT_MODEL_RT = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ASPECT_KEYS_RT = ("coa", "outcomes", "theme")
    MODE_TO_VERSION_RT = {
        "Detailed": "v1",
        "Step-by-step": "v2",
        "Compact": "v3",
    }
    MODE_DESCRIPTIONS_RT = {
        "Detailed": "Rich narrative analysis with prose explanations.",
        "Step-by-step": "Shows the extraction process in clear stages.",
        "Compact": "Returns concise structured results.",
    }
    VERSION_TO_MODE_RT = {version: mode for mode, version in MODE_TO_VERSION_RT.items()}
    
    EXAMPLE_STORY_RT = (
        "A young inventor discovers a hidden city beneath the desert after decoding "
        "a map left by her missing father. She enters the city with a reluctant guide, "
        "uncovers a machine that controls the region's water supply, and is forced to "
        "choose between restoring water to nearby villages or preserving the city's "
        "ancient secrecy. After sabotaging the machine's lock system, she escapes as "
        "water returns to the surface, but the city is exposed to the outside world."
    )
    
    # Prompts for V1, V2, V3
    V1_PROMPTS_RT = {
        "coa": """You are a narrative analyst. Read the story summary below and write ONLY the sequence of plot events - what happens, in what order, and what causes what. Do NOT mention character names, specific locations, or themes. Do NOT write any introduction, heading, or label before your answer. Begin your response immediately with the first event. Write 2-4 sentences.

Story:
{story}

Response:""",
        "outcomes": """You are a narrative analyst. Read the story summary below and write ONLY the final outcome and resolution. What is the end state? What did the protagonist ultimately achieve, lose, or experience? Do NOT describe how they got there. Do NOT write any introduction, heading, or label. Begin your response immediately with the outcome. Write 1-2 sentences.

Story:
{story}

Response:""",
        "theme": """You are a narrative analyst. Read the story summary below and write ONLY the abstract themes and universal human experiences it explores. What fundamental aspects of human nature, society, or morality does it examine? Do NOT mention specific characters, places, or plot events. Do NOT write any introduction, heading, or label. Begin your response immediately with the theme. Write 1-3 sentences.

Story:
{story}

Response:""",
    }
    
    V2_SYSTEM_MSG_RT = (
        "You are a precise narrative analyst specialising in story structure. "
        "You follow instructions exactly. "
        "You always output valid JSON with no markdown fences, no extra keys, "
        "and no text outside the JSON object."
    )
    
    V2_COMBINED_PROMPT_RT = """Analyse the story summary and extract three narrative aspects.
Return ONLY valid JSON with keys: "coa", "outcomes", "theme".
No extra text.

CORE PRINCIPLE
Each aspect must capture DIFFERENT information:
- COA = process (what happens)
- OUTCOMES = final state (what is true at the end)
- THEME = abstract meaning (what it represents)

Avoid overlap between them.

COA (Course of Action)
Describe the FULL causal sequence of events.

REQUIREMENTS:
- 3-6 numbered steps (1. 2. 3. ...)
- MUST include the FINAL transition into resolution
- Each step = one causal event
- Use abstract action types (escape, betrayal, investigation, confrontation, sacrifice)

USE:
- role labels (protagonist, antagonist, authority, ally)
- generic locations (city, prison, battlefield)

FORBIDDEN:
- character names
- specific places
- themes or emotions
- vague verbs ("deals with", "goes through")

OUTCOMES (STRICT FORMAT)
Describe ONLY the final stable state.

Write EXACTLY 2 sentences:

Sentence 1:
- protagonist final status (success / failure / survival / transformation)

Sentence 2:
- type of resolution:
  - conflict_resolved / unresolved / partial
  - + nature of change (personal / relational / systemic)

FORBIDDEN:
- "having..." clauses
- process descriptions
- vague words like "things improve"

THEME (NORMALIZED)
Write 2-4 SHORT phrases (not full sentences).

Each phrase must be:
- abstract
- generalizable across stories

FORMAT:
"theme1; theme2; theme3"

EXAMPLE

Story:
"A young man injures his brother, is placed under supervision, falsely accused, and later proven innocent."

Output:
{
  "coa": "1. Protagonist commits violence and is processed by authority.\\n2. Authority imposes supervision and assigns a helper.\\n3. Community falsely accuses protagonist, escalating conflict.\\n4. Evidence emerges that clears protagonist and resolves accusations.",
  "outcomes": "Protagonist is exonerated and transitions to a stable path of self-improvement. The conflict is resolved with personal transformation and social reintegration.",
  "theme": "redemption; social stigma; justice vs prejudice"
}

NOW ANALYSE

Story: {story}

Output:"""
    
    V3_SYSTEM_MSG_RT = (
        "You are a precise narrative analyst specialising in story structure. "
        "You follow instructions exactly. "
        "You always output valid JSON with no markdown fences, no extra keys, "
        "and no text outside the JSON object. "
        "You never invent events, outcomes, or successful resolutions that are not clearly supported by the story."
    )
    
    V3_COMBINED_PROMPT_RT = """Analyse the story summary and extract three narrative aspects.
Return ONLY valid JSON with keys: "coa", "outcomes", "theme".
No extra text.

CORE PRINCIPLE
Each aspect must capture DIFFERENT information:
- COA = process (what happens)
- OUTCOMES = final state (what is true at the end)
- THEME = abstract meaning (what it represents)

Avoid overlap between them.

COA (Course of Action)
Describe the FULL causal sequence of events.

REQUIREMENTS:
- 3-5 short clauses separated by " ; "
- MUST include the FINAL transition into resolution
- Each clause = one causal event
- Keep clauses short and structural, not long plot-summary prose
- Do not use numbering, bullet points, or arrows
- Use abstract action types (escape, betrayal, investigation, confrontation, sacrifice)

USE:
- role labels (protagonist, antagonist, authority, ally)
- generic locations (city, prison, battlefield)

FORBIDDEN:
- character names
- specific places
- themes or emotions
- vague verbs ("deals with", "goes through")

OUTCOMES (STRICT FORMAT)
Describe ONLY the final stable state explicitly supported by the summary.

Write EXACTLY 2 sentences:

Sentence 1:
- state the clearest end-state for the protagonist or central conflict
- use cautious wording if the ending is unclear

Sentence 2:
- state exactly one resolution label:
  - conflict resolved
  - conflict unresolved
  - conflict partially resolved

CRITICAL:
- Do NOT infer success, justice served, systemic change, broader implications, or future consequences unless clearly stated.
- If the ending is unclear, say so explicitly.
- Prefer literal wording over interpretation.

FORBIDDEN:
- "having..." clauses
- process descriptions
- vague words like "things improve"
- phrases like "broader threats remain", "personal transformation", "systemic change", "justice served", "new hope" unless explicitly stated

THEME (NORMALIZED)
Write 2-4 SHORT phrases (not full sentences).

Each phrase must be:
- abstract
- generalizable across stories
- 1-5 words long
- lower-case concept phrase

FORMAT:
"theme1; theme2; theme3"

EXAMPLE

Story:
"A young man injures his brother, is placed under supervision, falsely accused, and later proven innocent."

Output:
{
  "coa": "protagonist commits violence; authority imposes supervision; community escalates accusations; evidence clears protagonist",
  "outcomes": "Protagonist is exonerated and allowed to move forward. conflict resolved.",
  "theme": "redemption; social stigma; justice vs prejudice"
}

NOW ANALYSE

Story: {story}

Output:"""
    
    V3_REPAIR_PROMPT_RT = """You are repairing a noisy narrative-aspect extraction.
Return ONLY valid JSON with keys: "coa", "outcomes", "theme".

Requirements:
- remove character names and specific places
- keep coa as 3-5 short structural clauses separated by " ; "
- remove numbering, arrows, and bullet points from coa
- keep outcomes as exactly 2 short cautious sentences about final state only
- make the second outcomes sentence exactly one of: "conflict resolved." / "conflict unresolved." / "conflict partially resolved."
- keep theme as 2-4 short lower-case phrases separated by semicolons
- do not invent success, justice served, systemic change, broader implications, or details not clearly present in the story

Story:
{story}

Current extraction:
{bad_json}

Return repaired JSON only."""
    
    PREAMBLE_RE_RT = re.compile(
        r"^(?:"
        r"here (?:is|are)(?: the| a)?[^\n:]{0,80}?[\s:.-]*\n+"
        r"|"
        r"(?:certainly|sure|of course|absolutely)[!,.]?[^\n]*\n*"
        r"|"
        r"(?:course of action|outcomes?|abstract theme|response|answer)\s*[:]\s*\n*"
        r")",
        re.IGNORECASE,
    )
    
    # Helper functions
    def fill_story_rt(template, story_text):
        return template.replace("{story}", story_text.strip())
    
    def call_ollama_rt(host, model, prompt, max_tokens=500, json_mode=False, temperature=0.1):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        if json_mode:
            payload["format"] = "json"
        
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {host}. Start Ollama with 'ollama serve' and pull the model with 'ollama pull {model}'."
            ) from exc
    
    def clean_response_rt(text):
        if not text:
            return ""
        text = text.strip()
        for _ in range(4):
            cleaned = PREAMBLE_RE_RT.sub("", text).strip()
            if cleaned == text:
                break
            text = cleaned
        text = re.sub(r"^\s*\n+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def parse_json_response_rt(raw):
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and all(key in obj for key in ASPECT_KEYS_RT):
                return obj
        except json.JSONDecodeError:
            pass
        
        recovered = {}
        for key in ASPECT_KEYS_RT:
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                recovered[key] = (
                    match.group(1)
                    .replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
        
        if len(recovered) == 3:
            return recovered
        
        raise ValueError(f"Could not parse JSON response. Raw response: {raw[:500]}")
    
    def theme_parts_rt(text):
        return [
            part.strip(" -.;,").lower()
            for part in re.split(r"[;\n,]", text)
            if part.strip(" -.;,")
        ]
    
    def postprocess_coa_rt(text):
        text = clean_response_rt(text)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"(?m)^\s*(\d+)[.)]\s*", "", text)
        text = re.sub(r"(?i)\bstep\s+\d+\s*[:.-]\s*", "", text)
        text = text.replace("\u2192", "; ")
        text = re.sub(r"\s*(?:->)\s*", "; ", text)
        text = re.sub(r"\s-\s", "; ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"(?i)\b(protagonist|antagonist|authority|ally)\s*:\s*", r"\1 ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip(" ;")
    
    def postprocess_outcomes_rt(text):
        text = clean_response_rt(text)
        text = re.sub(r"\bresolved with partial resolution\b", "partially resolved", text, flags=re.IGNORECASE)
        text = re.sub(r"\bconflict is resolved with unresolved\b", "conflict remains unresolved", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprotagonist succeeds in\b", "protagonist attempts to", text, flags=re.IGNORECASE)
        text = re.sub(r"\bultimately (solves|resolves)\b", "ultimately addresses", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfully resolves\b", "largely resolves", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\b(personal transformation|systemic change|justice served|new hope|broader (threats|issues|relationships) remain)\b",
            "", text, flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", text).strip()
    
    def postprocess_theme_rt(text):
        phrases = []
        seen = set()
        for part in theme_parts_rt(clean_response_rt(text)):
            part = re.sub(r"\b(the|a|an|story explores|themes? of)\b", "", part, flags=re.IGNORECASE)
            part = re.sub(r"\b(moral of the story|central theme|main theme)\b", "", part, flags=re.IGNORECASE)
            part = re.sub(r"\s+", " ", part).strip(" -.;,")
            if part and len(part.split()) <= 5 and part not in seen:
                seen.add(part)
                phrases.append(part)
        return "; ".join(phrases[:4])
    
    def extract_v1_rt(story_text, host, model):
        aspects = {}
        for key, prompt_template in V1_PROMPTS_RT.items():
            prompt = fill_story_rt(prompt_template, story_text)
            aspects[key] = clean_response_rt(
                call_ollama_rt(
                    host=host,
                    model=model,
                    prompt=prompt,
                    max_tokens=220,
                    json_mode=False,
                    temperature=0.2,
                )
            )
        return aspects
    
    def extract_v2_rt(story_text, host, model):
        prompt = V2_SYSTEM_MSG_RT + "\n\n" + fill_story_rt(V2_COMBINED_PROMPT_RT, story_text)
        raw = call_ollama_rt(
            host=host,
            model=model,
            prompt=prompt,
            max_tokens=500,
            json_mode=True,
            temperature=0.1,
        )
        parsed = parse_json_response_rt(raw)
        return {key: clean_response_rt(parsed.get(key, "")) for key in ASPECT_KEYS_RT}
    
    def extract_v3_rt(story_text, host, model):
        prompt = V3_SYSTEM_MSG_RT + "\n\n" + fill_story_rt(V3_COMBINED_PROMPT_RT, story_text)
        raw = call_ollama_rt(
            host=host,
            model=model,
            prompt=prompt,
            max_tokens=500,
            json_mode=True,
            temperature=0.0,
        )
        parsed = parse_json_response_rt(raw)
        aspects = {
            "coa": postprocess_coa_rt(parsed.get("coa", "")),
            "outcomes": postprocess_outcomes_rt(parsed.get("outcomes", "")),
            "theme": postprocess_theme_rt(parsed.get("theme", "")),
        }
        return aspects
    
    EXTRACTORS_RT = {
        "v1": extract_v1_rt,
        "v2": extract_v2_rt,
        "v3": extract_v3_rt,
    }
    
    def run_extraction_rt(version, story_text, host, model):
        return EXTRACTORS_RT[version](story_text=story_text, host=host, model=model)
    
    def get_executor_rt():
        if "executor_rt" not in st.session_state:
            st.session_state["executor_rt"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return st.session_state["executor_rt"]
    
    def active_job_running_rt():
        job = st.session_state.get("rt_active_job")
        return bool(job and not job["future"].done())
    
    def start_extraction_job_rt(version, story_text, host, model):
        if active_job_running_rt():
            running_version = st.session_state["rt_active_job"]["version"]
            st.session_state["rt_last_error"] = f"Already running {running_version}."
            return
        
        st.session_state["rt_last_result"] = None
        future = get_executor_rt().submit(run_extraction_rt, version, story_text, host, model)
        st.session_state["rt_active_job"] = {
            "future": future,
            "version": version,
            "model": model,
            "started_at": time.time(),
        }
        st.session_state["rt_last_error"] = None
    
    def update_extraction_job_rt():
        job = st.session_state.get("rt_active_job")
        if not job:
            return False
        
        future = job["future"]
        if not future.done():
            return True
        
        try:
            result = future.result()
        except Exception as exc:
            st.session_state["rt_last_error"] = str(exc)
        else:
            st.session_state["rt_last_error"] = None
            st.session_state["rt_last_result"] = {
                "version": job["version"],
                "model": job["model"],
                "aspects": result,
            }
        
        st.session_state["rt_active_job"] = None
        return False
    
    def ollama_connected_rt(host):
        try:
            request = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=1):
                return True
        except Exception:
            return False
    
    def use_example_story_rt():
        st.session_state["rt_story_input"] = EXAMPLE_STORY_RT
    
    def clear_story_input_rt():
        st.session_state["rt_story_input"] = ""
        st.session_state["rt_last_result"] = None
        st.session_state["rt_last_error"] = None
    
    def open_prompts_rt(version):
        st.session_state["rt_prompt_version"] = version
        st.session_state["rt_prompt_open"] = True
    
    def close_prompts_rt():
        st.session_state["rt_prompt_open"] = False
    
    def prompts_for_version_rt(version, story_text):
        story_for_preview = story_text.strip() or "{story}"
        
        if version == "v1":
            return [
                ("CoA", fill_story_rt(V1_PROMPTS_RT["coa"], story_for_preview)),
                ("Outcomes", fill_story_rt(V1_PROMPTS_RT["outcomes"], story_for_preview)),
                ("Theme", fill_story_rt(V1_PROMPTS_RT["theme"], story_for_preview)),
            ]
        
        if version == "v2":
            return [
                ("System", V2_SYSTEM_MSG_RT),
                ("Combined", fill_story_rt(V2_COMBINED_PROMPT_RT, story_for_preview)),
            ]
        
        if version == "v3":
            repair_preview = fill_story_rt(V3_REPAIR_PROMPT_RT, story_for_preview).replace(
                "{bad_json}", "{bad_extraction_json}"
            )
            return [
                ("System", V3_SYSTEM_MSG_RT),
                ("Combined", fill_story_rt(V3_COMBINED_PROMPT_RT, story_for_preview)),
                ("Repair", repair_preview),
            ]
        
        raise ValueError(f"Unknown prompt version: {version}")
    
    def render_prompt_preview_rt(version, story_text):
        prompts = prompts_for_version_rt(version, story_text)
        header_col, close_col = st.columns([0.94, 0.06])
        mode = VERSION_TO_MODE_RT.get(version, version)
        header_col.subheader(f"Prompt template: {mode}")
        close_col.button(
            "X",
            key=f"close_prompts_rt_{version}",
            help="Close prompts",
            on_click=close_prompts_rt,
            use_container_width=True,
        )
        
        tabs = st.tabs([name for name, _ in prompts])
        for tab, (_, prompt_text) in zip(tabs, prompts):
            with tab:
                st.code(prompt_text, language="text")
    
    # Initialize session state
    st.session_state.setdefault("rt_prompt_open", False)
    st.session_state.setdefault("rt_prompt_version", None)
    st.session_state.setdefault("rt_last_result", None)
    st.session_state.setdefault("rt_last_error", None)
    st.session_state.setdefault("rt_active_job", None)
    st.session_state.setdefault("rt_story_input", "")
    st.session_state.setdefault("rt_selected_mode", "Detailed")
    
    update_extraction_job_rt()
    
    # Title
    st.markdown("## ⚡ Real-time Aspect Extraction")
    st.markdown(
        "<div class='secondary-copy'>Extract course of action, outcomes, and abstract theme from story summaries using a local LLM.</div>",
        unsafe_allow_html=True,
    )
    
    # Main layout
    input_col, config_col = st.columns([0.64, 0.36], gap="large")
    
    with input_col:
        with st.container(border=True):
            st.markdown("### Story summary")
            st.caption("Paste a synopsis, chapter outline, or scene.")
            
            example_col, clear_col = st.columns([0.5, 0.5])
            example_col.button(
                "Use example story",
                on_click=use_example_story_rt,
                use_container_width=True,
            )
            clear_col.button(
                "Clear input",
                on_click=clear_story_input_rt,
                use_container_width=True,
            )
            
            story_rt = st.text_area(
                "Input story",
                key="rt_story_input",
                height=300,
                label_visibility="collapsed",
                placeholder=(
                    "Paste a story summary, synopsis, or chapter outline here.\n"
                    "Example: A young inventor discovers a hidden city beneath the desert..."
                ),
            )
            st.caption(f"{len(story_rt)} characters | Recommended length: 200-2,000 words")
    
    with config_col:
        with st.container(border=True):
            st.markdown("### Extraction settings")
            selected_mode_rt = st.radio(
                "Extraction style",
                list(MODE_TO_VERSION_RT.keys()),
                key="rt_selected_mode",
                horizontal=True,
            )
            st.markdown(
                f"<div class='mode-help'>{MODE_DESCRIPTIONS_RT[selected_mode_rt]}</div>",
                unsafe_allow_html=True,
            )
            
            requested_version_rt = None
            if st.button(
                "Extract narrative aspects",
                type="primary",
                use_container_width=True,
                disabled=active_job_running_rt() or not story_rt.strip(),
            ):
                requested_version_rt = MODE_TO_VERSION_RT[selected_mode_rt]
            
            st.button(
                "View prompt template",
                use_container_width=True,
                on_click=open_prompts_rt,
                args=(MODE_TO_VERSION_RT[selected_mode_rt],),
            )
            
            st.markdown("---")
            st.markdown("### Model settings")
            host_rt = st.text_input("Host", DEFAULT_HOST_RT, key="rt_host")
            model_rt = st.text_input("Model", DEFAULT_MODEL_RT, key="rt_model")
            st.session_state["current_host_rt"] = host_rt.strip() or DEFAULT_HOST_RT
            
            connected_rt = ollama_connected_rt(st.session_state["current_host_rt"])
            status_color = "#16a34a" if connected_rt else "#dc2626"
            status_text = "Connected" if connected_rt else "Not connected"
            st.markdown(
                "<div class='settings-label'>Connection status</div>"
                f"<div style='font-weight:600;color:{status_color};'>{status_text}</div>",
                unsafe_allow_html=True,
            )
            
            if not connected_rt:
                with st.expander("Setup commands"):
                    st.code(f"ollama pull {model_rt}", language="bash")
                    st.code("ollama serve", language="bash")
    
    if requested_version_rt:
        if not story_rt.strip():
            st.warning("Paste a story summary first.")
        else:
            start_extraction_job_rt(
                requested_version_rt,
                story_rt.strip(),
                st.session_state["current_host_rt"],
                model_rt.strip() or DEFAULT_MODEL_RT,
            )
    
    # Results section
    was_running = active_job_running_rt()
    job_running = update_extraction_job_rt()
    if was_running and not job_running:
        st.rerun()
    
    with st.container(border=True):
        st.markdown("### Results")
        
        if job_running:
            job = st.session_state["rt_active_job"]
            elapsed = int(time.time() - job["started_at"])
            mode = VERSION_TO_MODE_RT.get(job["version"], job["version"])
            st.info(
                f"Extracting narrative aspects with {mode} mode... {elapsed}s\n\n"
                "This may take a few seconds depending on the model."
            )
        
        last_error = st.session_state.get("rt_last_error")
        if last_error:
            configured_host = st.session_state.get("current_host_rt", DEFAULT_HOST_RT)
            st.error(
                "Could not complete the extraction.\n\n"
                f"{last_error}\n\n"
                f"Check that Ollama is running at {configured_host}."
            )
        
        last_result = st.session_state.get("rt_last_result")
        if last_result:
            mode = VERSION_TO_MODE_RT.get(last_result["version"], last_result["version"])
            st.success(f"Extraction complete. Mode: {mode}.")
            aspects = last_result["aspects"]
            
            out1, out2, out3 = st.columns(3)
            with out1:
                st.markdown("<span class='pill pill-coa'>**Course of Action**</span>", unsafe_allow_html=True)
                st.write(aspects.get("coa", ""))
            with out2:
                st.markdown("<span class='pill pill-out'>**Outcomes**</span>", unsafe_allow_html=True)
                st.write(aspects.get("outcomes", ""))
            with out3:
                st.markdown("<span class='pill pill-thm'>**Abstract Theme**</span>", unsafe_allow_html=True)
                st.write(aspects.get("theme", ""))
            
            action_col1, action_col2 = st.columns([0.24, 0.76])
            with action_col1:
                st.download_button(
                    "Export JSON",
                    data=json.dumps(last_result, indent=2, ensure_ascii=False),
                    file_name=f"aspects_{mode.lower().replace('-', '_')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            
            with st.expander("Raw JSON"):
                st.json(last_result)
        
        if not job_running and not last_error and not last_result:
            st.info(
                "No extraction yet.\n\n"
                "Paste a story summary and click Extract narrative aspects to see results here."
            )
    
    # Prompt preview
    prompt_version = st.session_state.get("rt_prompt_version")
    if st.session_state.get("rt_prompt_open") and prompt_version:
        st.divider()
        render_prompt_preview_rt(prompt_version, story_rt)

# ======================== PAGE: Aspect Informativeness ========================
elif page == "Aspect Informativeness":
    st.markdown("## 📈 Aspect Informativeness")
    st.markdown("<div class='section-title'>Statistical evaluation of extracted aspects (V1, V2, V3)</div>", unsafe_allow_html=True)

    st.markdown("#### Aspect informativeness (RoBERTa‑large, 200 development triples)")
    st.dataframe(DF_ROBERTA_200, width="stretch", hide_index=True)
    st.markdown("*Notes: % pos>neg is the percentage of triples where the positive candidate’s similarity exceeds the negative's. Mean diff. is the average (pos-neg) cosine similarity. Statistical significance (Sig. yes) is defined as p < 0.05 (paired t‑test).*")

    st.markdown("#### Aspect informativeness (BGE‑M3, 200 development triples)")
    st.dataframe(DF_BGEM3_200, width="stretch", hide_index=True)

    st.markdown("#### Aspect informativeness on the 166 clean triples (gold aspect labels)")
    st.dataframe(DF_166_GOLD, width="stretch", hide_index=True)
    st.markdown("*Significance: ** p<0.01, * p<0.05*")

    st.markdown("#### Aspect informativeness (Correlations)")
    st.dataframe(DF_CORRELATIONS, width="stretch", hide_index=True)
    st.markdown("*Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05*")

    st.markdown("---")
    st.markdown("#### Methodological Summary")
    st.markdown("""
    - **Course of action (CoA)**: best discriminative power with RoBERTa + V1 (62.5% pos>neg).  
    - **Outcomes**: most consistent with V1 across both models.  
    - **Theme**: only significant with BGE‑M3 + V3, but weak (55.5%).  
    - **Complementarity**: CoA and Outcomes capture partially overlapping but not identical information (r ≈ 0.21–0.24).  
    - **Gold aspect labels** have low inter‑annotator agreement (α ≈ 0.05–0.11); extracted aspects sometimes outperform them.
    """)

# ======================== PAGE: Aspect Explorer ========================
elif page == "Aspect Explorer":
    st.markdown("## 🔍 Aspect Explorer")
    st.markdown("<div class='section-title'>Browse extracted aspects (Version 3 – compact phrases)</div>", unsafe_allow_html=True)
    aspects = load_aspects_cache(3)
    dev = load_dev_triples()
    if not aspects:
        st.warning("Aspect cache V3 not found.")
        st.stop()
    stories = {}
    for t in dev:
        for field in ["anchor_text", "text_a", "text_b"]:
            text = t.get(field, "")
            if text and text not in stories:
                title = t.get(f"{field}_title", "")
                norm = norm_text(text)
                entry = aspects.get(norm, {})
                stories[text] = {
                    "title": title or "",
                    "text": text,
                    "coa": entry.get("coa", ""),
                    "outcomes": entry.get("outcomes", ""),
                    "theme": entry.get("theme", ""),
                    "has_asp": bool(entry.get("coa") or entry.get("theme"))
                }
    story_list = list(stories.values())
    search = st.text_input("Search stories", placeholder="keyword in title or text…")
    filtered = [s for s in story_list if search.lower() in s["title"].lower() or search.lower() in s["text"].lower()] if search else story_list
    st.markdown(f"<div class='section-title'>{len(filtered)} stories match</div>", unsafe_allow_html=True)
    per_page = st.selectbox("Per page", [10,20,50], index=0)
    total_pages = max(1, (len(filtered)+per_page-1)//per_page)
    page_num = st.number_input("Page", 1, total_pages, 1, step=1)
    page_stories = filtered[(page_num-1)*per_page : page_num*per_page]
    for s in page_stories:
        title_part = f"**{s['title']}** — " if s['title'] else ""
        with st.expander(f"{title_part}{s['text'][:80]}…", expanded=False):
            st.markdown(f"<div class='story-block'>{s['text']}</div>", unsafe_allow_html=True)
            if s["has_asp"]:
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown("<span class='pill pill-coa'>CoA</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card' style='border-left: 3px solid #3a6ea5;'><small>{s['coa'] or '—'}</small></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<span class='pill pill-out'>Outcomes</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card' style='border-left: 3px solid #b88b4a;'><small>{s['outcomes'] or '—'}</small></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<span class='pill pill-thm'>Theme</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card' style='border-left: 3px solid #8a5ea8;'><small>{s['theme'] or '—'}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill pill-miss'>No aspects in cache</span>", unsafe_allow_html=True)