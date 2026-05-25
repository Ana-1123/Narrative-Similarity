"""
Run:
    pip install streamlit plotly pandas sentence-transformers scipy
    streamlit run app.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Narrative Similarity · Thesis Demo",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== DESIGN TOKENS ========================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=JetBrains+Mono:wght@300;400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #14171f;
    --border:    #1e2330;
    --accent:    #c8a96e;
    --accent2:   #6e9ec8;
    --accent3:   #c86e9e;
    --text:      #e8e4dc;
    --muted:     #6b7280;
    --success:   #6ec88a;
    --danger:    #c86e6e;
    --coa:       #6e9ec8;
    --out:       #c8a96e;
    --theme:     #a86ec8;
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
.pill-coa   { background: #1a2a3a; color: var(--coa);   border: 1px solid var(--coa);   }
.pill-out   { background: #3a2a1a; color: var(--out);   border: 1px solid var(--out);   }
.pill-thm   { background: #2a1a3a; color: var(--theme); border: 1px solid var(--theme); }
.pill-match { background: #1a3a2a; color: var(--success);border: 1px solid var(--success);}
.pill-miss  { background: #3a1a1a; color: var(--danger); border: 1px solid var(--danger); }

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
              text-transform: uppercase; letter-spacing: 0.12em; color: #6b7280;
              margin-bottom: 4px;'>Master Thesis · 2026</div>
  <div style='font-family: Playfair Display, serif; font-size: 1.3rem;
              font-weight: 700; line-height: 1.2; color: #e8e4dc;'>
    Aspect-Aware<br>Narrative<br>Similarity
  </div>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem;
              color: #6b7280; margin-top: 8px;'>
    Alexandru Ioan Cuza University<br>Iași · Computational Linguistics MSc
  </div>
</div>
<hr style='margin: 0.5rem 0 1rem 0;'>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        # ["📖 Live Demo",
        ["🗂 Dataset",
         "📊 Ablation Results",
         "🔬 Aspect Extraction (V1/V2/V3)",
         "📈 Aspect Informativeness",
         "🔍 Aspect Explorer"],
        #  "❌ Error Analysis"],
        label_visibility="collapsed",
    )
    page = page.split(" ", 1)[1]

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("""
# <div style='font-family: JetBrains Mono, monospace; font-size: 0.68rem; color: #6b7280; line-height: 1.8;'>
#   <b style='color:#c8a96e;'>Model:</b> RoBERTa-large<br>
# </div>
# """, unsafe_allow_html=True)

# ======================== PAGE: Live Demo ========================
if page == "Live Demo":
    st.markdown("## 📖 Live Demonstration")
    st.markdown("<div class='section-title'>Aspect extraction · Similarity comparison</div>", unsafe_allow_html=True)

    triples = load_dev_triples()
    aspects = load_aspects_cache(3)  # use V3 for clean display

    if not triples:
        st.warning("Dev triples not found.")
        st.stop()

    # Story selector
    col_l, col_r = st.columns([2, 1])
    with col_l:
        mode = st.radio("Input mode", ["Pick from dataset", "Type my own"], horizontal=True)

    if mode == "Pick from dataset" and triples:
        with col_r:
            idx = st.number_input("Triple index (0–199)", 0, len(triples)-1, value=0, step=1)
        triple = triples[int(idx)]
        anchor_text = triple["anchor_text"]
        coa_gold    = triple.get("course_of_actions", [None, None])
        out_gold    = triple.get("outcomes",          [None, None])
        thm_gold    = triple.get("abstract_theme",    [None, None])
        label_gold  = triple.get("text_a_is_closer")
    else:
        anchor_text = st.text_area("Paste a story summary:", height=140)
        coa_gold = out_gold = thm_gold = [None, None]
        label_gold = None
        triple = None

    if not anchor_text.strip():
        st.info("Select a triple index or paste a story summary to begin.")
        st.stop()

    st.markdown("#### Anchor Story")
    st.markdown(f"<div class='story-block'>{anchor_text}</div>", unsafe_allow_html=True)

    norm_key = norm_text(anchor_text)
    entry = aspects.get(norm_key, {})
    coa      = entry.get("coa",      "")
    outcomes = entry.get("outcomes", "")
    theme    = entry.get("theme",    "")
    has_asp = bool(coa or outcomes or theme)

    st.markdown("---")
    st.markdown("#### Extracted Aspects")
    if not has_asp:
        st.warning("No pre-extracted aspects found for this story.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<span class='pill pill-coa'>CoA</span> Course of Action", unsafe_allow_html=True)
            st.markdown(f"<div class='card card-accent-coa' style='min-height:120px'><small>{coa or '—'}</small></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<span class='pill pill-out'>OUT</span> Outcomes", unsafe_allow_html=True)
            st.markdown(f"<div class='card card-accent-out' style='min-height:120px'><small>{outcomes or '—'}</small></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<span class='pill pill-thm'>THM</span> Abstract Theme", unsafe_allow_html=True)
            st.markdown(f"<div class='card card-accent-thm' style='min-height:120px'><small>{theme or '—'}</small></div>", unsafe_allow_html=True)

    if triple is not None:
        st.markdown("---")
        st.markdown("#### Similarity Comparison")
        text_a = triple["text_a"]
        text_b = triple["text_b"]

        def match_badge(match_list, idx):
            if match_list[0] is None:
                return ""
            val = match_list[idx]
            return f"<span class='pill pill-match'>✓ Match</span>" if val else f"<span class='pill pill-miss'>✗ No match</span>"

        for story, label, story_idx in [(text_a, "Text A", 0), (text_b, "Text B", 1)]:
            is_closer = (label == "Text A" and label_gold) or (label == "Text B" and not label_gold)
            gold_str = "<span class='badge-correct'>← Gold: closer</span>" if is_closer else "<span class='badge-wrong'>Gold: not closer</span>" if label_gold is not None else ""
            st.markdown(f"<div class='section-title'>{label} &nbsp;{gold_str}</div>", unsafe_allow_html=True)
            norm_s = norm_text(story)
            s_entry = aspects.get(norm_s, {})
            col_txt, col_asp = st.columns([3, 2])
            with col_txt:
                st.markdown(f"<div class='story-block'>{story[:400]}{'…' if len(story)>400 else ''}</div>", unsafe_allow_html=True)
            with col_asp:
                coa_badge = match_badge(coa_gold, story_idx)
                out_badge = match_badge(out_gold, story_idx)
                thm_badge = match_badge(thm_gold, story_idx)
                st.markdown(
                    f"<div class='card'>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-coa'>CoA</span>{coa_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em;margin-bottom:10px'>{s_entry.get('coa','—')[:180]}</div>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-out'>OUT</span>{out_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em;margin-bottom:10px'>{s_entry.get('outcomes','—')[:150]}</div>"
                    f"<div style='margin-bottom:6px'><span class='pill pill-thm'>THM</span>{thm_badge}</div>"
                    f"<div style='color:#ffffff;font-size:0.8em'>{s_entry.get('theme','—')[:120]}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        if coa_gold[0] is not None:
            st.markdown("---")
            st.markdown("<div class='section-title'>Gold aspect label pattern</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for c, (asp_name, asp_labels) in zip(cols, [("Course of Action", coa_gold), ("Outcomes", out_gold), ("Abstract Theme", thm_gold)]):
                pattern = {(True, True): "Both match anchor", (True, False): "Only A matches", (False, True): "Only B matches", (False, False): "Neither matches"}.get(tuple(asp_labels), "Unknown")
                c.markdown(f"<div class='metric-box'><div class='metric-label'>{asp_name}</div><div style='font-family: JetBrains Mono; font-size:0.85rem; color: #c8a96e; margin-top:6px'>{pattern}</div></div>", unsafe_allow_html=True)

# ======================== PAGE: Dataset ========================
# ======================== PAGE: Dataset (IMPROVED LAYOUT) ========================
elif page == "Dataset":
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
    test_a_stats, test_a_words, _ = compute_dataset_stats("Test (Track A)", test_a, is_ranking=False)
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
    
    # Prepare combined data for plots
    data_for_box = []
    for words, name in [(dev_words, "Dev"), (test_a_words, "Test A"), (test_b_words, "Test B"), (synth_words, "Synth")]:
        for w in words:
            data_for_box.append({"Dataset": name, "Words": w})
    box_df = pd.DataFrame(data_for_box)
    
    with col_dist1:
        fig = px.box(box_df, x="Dataset", y="Words", category_orders={"Dataset": ["Dev", "Test A", "Test B", "Synth"]})
        fig.update_layout(title=dict(text="Word‑count distribution by dataset", font=dict(family="Playfair Display", size=14, color="#e8e4dc")), 
                         xaxis_title="", yaxis_title="Word count",
                         paper_bgcolor="#14171f", plot_bgcolor="#14171f", font=dict(family="JetBrains Mono", color="#e8e4dc"), 
                         margin=dict(l=10, r=10, t=50, b=30), height=350, showlegend=False)
        fig.update_xaxes(gridcolor="#1e2330")
        fig.update_yaxes(gridcolor="#1e2330")
        st.plotly_chart(fig, width="stretch", use_container_width=True)
    
    with col_dist2:
        fig = px.histogram(box_df, x="Words", color="Dataset", nbins=30, barmode="overlay", 
                          category_orders={"Dataset": ["Dev", "Test A", "Test B", "Synth"]})
        fig.update_traces(opacity=0.65)
        fig.update_layout(title=dict(text="Word‑length frequency distribution", font=dict(family="Playfair Display", size=14, color="#e8e4dc")), 
                         xaxis_title="Word count", yaxis_title="Frequency",
                         paper_bgcolor="#14171f", plot_bgcolor="#14171f", font=dict(family="JetBrains Mono", color="#e8e4dc"), 
                         margin=dict(l=10, r=10, t=50, b=30), height=350, showlegend=True)
        fig.update_xaxes(gridcolor="#1e2330")
        fig.update_yaxes(gridcolor="#1e2330")
        fig.update_layout(legend=dict(x=0.65, y=0.95))
        st.plotly_chart(fig, width="stretch", use_container_width=True)

    st.markdown("---")

    # ----- Dataset composition (textual description) -----
    st.markdown("### Dataset Composition")
    with st.container():
        st.markdown(
            f"""
            <div class='card' style='background: #0d0f14;'>
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
        # Compute aspect agreement across A vs B candidates
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
        
        # Two charts side by side: agreement % and match rate %
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            agree_df = pd.DataFrame({
                "Aspect": ["CoA", "Outcomes", "Theme"],
                "Agreement %": [coa_agree, out_agree, theme_agree]
            })
            fig = px.bar(agree_df, x="Aspect", y="Agreement %", text="Agreement %", range_y=[0, 100])
            fig.update_traces(marker_color="#c8a96e", textposition="outside", texttemplate="%{text:.1f}%")
            fig.update_layout(title=dict(text="Aspect matching consistency (A vs B)", font=dict(family="Playfair Display", size=14, color="#e8e4dc")), 
                             xaxis_title="", yaxis_title="Agreement (%)",
                             paper_bgcolor="#14171f", plot_bgcolor="#14171f", font=dict(family="JetBrains Mono", color="#e8e4dc"), 
                             margin=dict(l=10, r=10, t=50, b=30), height=330, showlegend=False)
            fig.update_xaxes(gridcolor="#1e2330")
            fig.update_yaxes(gridcolor="#1e2330")
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
            fig.update_layout(title=dict(text="Aspect matching frequency (overall)", font=dict(family="Playfair Display", size=14, color="#e8e4dc")), 
                             xaxis_title="", yaxis_title="Match Rate (%)",
                             paper_bgcolor="#14171f", plot_bgcolor="#14171f", font=dict(family="JetBrains Mono", color="#e8e4dc"), 
                             margin=dict(l=10, r=10, t=50, b=30), height=330, showlegend=False)
            fig.update_xaxes(gridcolor="#1e2330")
            fig.update_yaxes(gridcolor="#1e2330")
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
            <div class='card' style='background: #0d0f14;'>
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
elif page == "Ablation Results":
    st.markdown("## 📊 Ablation Results")
    st.markdown("<div class='section-title'>All experimental conditions</div>", unsafe_allow_html=True)

    RESULTS = [
        {"Cond": "A",  "Name": "Baseline",          "Input":   "Full text",      "MaxLen": 128, "Heads": "—", "Track A %": 70.75, "Track B %": 65.75, "Status": "✓ Complete", "Note": "Competition-equivalent"},
        {"Cond": "B",  "Name": "CoA only",           "Input":   "CoA text",       "MaxLen": 128, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "C",  "Name": "Outcomes only",      "Input":   "Outcomes text",  "MaxLen": 128, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "D",  "Name": "Theme only",         "Input":   "Theme text",     "MaxLen": 128, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "E",  "Name": "Concat aspects",     "Input":   "CoA+Out+Theme",  "MaxLen": 128, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "F",  "Name": "Extended ctx",       "Input":   "Full text",      "MaxLen": 512, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "G",  "Name": "Aspect heads",       "Input":   "Full text",      "MaxLen": 128, "Heads": "✓", "Track A %": 68.75, "Track B %": 66.00, "Status": "✓ Complete", "Note": "Old architecture"},
        {"Cond": "G+", "Name": "Aspect heads (fixed)","Input":  "Full text",      "MaxLen": 128, "Heads": "✓", "Track A %": 64.50, "Track B %": 59.75, "Status": "✓ Complete", "Note": "ModernBERT"},
        {"Cond": "H",  "Name": "Aspect heads+",      "Input":   "CoA text",       "MaxLen": 128, "Heads": "✓", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
        {"Cond": "J",  "Name": "CoA + Theme",        "Input":   "CoA+Theme",      "MaxLen": 128, "Heads": "—", "Track A %": None,  "Track B %": None,  "Status": "⏳ Pending",  "Note": ""},
    ]
    NARRATIVE_TEAM = {"Track A %": 64.25, "Track B %": 69.25}
    PLOT_BG = "#14171f"
    PLOT_GRID = "#1e2330"
    FONT_COLOR = "#e8e4dc"

    df = pd.DataFrame(RESULTS)
    best_a = max((r["Track A %"] for r in RESULTS if r["Track A %"]), default=0)
    best_b = max((r["Track B %"] for r in RESULTS if r["Track B %"]), default=0)
    complete = sum(1 for r in RESULTS if r["Track A %"] is not None)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, label in [
        (m1, f"{best_a:.2f}%", "Best Track A"),
        (m2, f"{best_b:.2f}%", "Best Track B"),
        (m3, f"{NARRATIVE_TEAM['Track A %']:.2f}%", "NTeam Track A"),
        (m4, f"{NARRATIVE_TEAM['Track B %']:.2f}%", "NTeam Track B"),
        (m5, f"{complete}/{len(RESULTS)}", "Conditions done"),
    ]:
        col.markdown(f"<div class='metric-box'><div class='metric-val'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2 = st.tabs(["Track A — Classification", "Track B — Embedding Ranking"])
    for tab, track_col, nt_val in [(tab1, "Track A %", NARRATIVE_TEAM["Track A %"]), (tab2, "Track B %", NARRATIVE_TEAM["Track B %"])]:
        with tab:
            df_t = df[df[track_col].notna()].copy()
            df_t = df_t.sort_values(track_col, ascending=True)
            colors = ["#c8a96e" if v == df_t[track_col].max() else "#6e9ec8" for v in df_t[track_col]]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_t[track_col], y=df_t["Cond"] + " · " + df_t["Name"], orientation="h", marker_color=colors, text=[f"{v:.2f}%" for v in df_t[track_col]], textposition="outside", textfont=dict(family="JetBrains Mono", size=11, color=FONT_COLOR)))
            fig.add_vline(x=nt_val, line_dash="dash", line_color="#6e6e6e", line_width=1.5)
            fig.add_annotation(x=nt_val, y=-0.8, text=f"Narrative Team {nt_val}%", showarrow=False, font=dict(family="JetBrains Mono", size=10, color="#6b7280"), xanchor="left")
            fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(family="JetBrains Mono", color=FONT_COLOR), margin=dict(l=10, r=80, t=30, b=40), xaxis=dict(gridcolor=PLOT_GRID, range=[55, 78], ticksuffix="%"), yaxis=dict(gridcolor=PLOT_GRID), height=380, showlegend=False)
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### All Conditions")
    display_df = df[["Cond", "Name", "Input", "MaxLen", "Heads", "Track A %", "Track B %", "Status", "Note"]].copy()
    display_df["Track A %"] = display_df["Track A %"].apply(lambda v: f"{v:.2f}%" if v is not None else "—")
    display_df["Track B %"] = display_df["Track B %"].apply(lambda v: f"{v:.2f}%" if v is not None else "—")
    st.dataframe(display_df, width="stretch", hide_index=True)

# ======================== PAGE: Aspect Extraction (V1/V2/V3) ========================
elif page == "Aspect Extraction (V1/V2/V3)":
    st.markdown("## 🔬 Aspect Extraction Methods")
    st.markdown("<div class='section-title'>Three prompt strategies compared</div>", unsafe_allow_html=True)

    descriptions = {
        1: "Version 1 – Narrative Prose  \nSeparate per-aspect LLM calls, unrestricted prose. Retains rich detail but can be verbose. Best discriminative signal for CoA.",
        2: "Version 2 – Role-Label Steps  \nSingle combined JSON prompt; numbered steps with role labels (protagonist, authority). Better structural abstraction.",
        3: "Version 3 – Compact Phrases  \nHighly constrained prompt returning semicolon-separated phrases. Cleanest output, but loses some similarity signal."
    }
    for ver, desc in descriptions.items():
        st.markdown(
            f"<div class='card-light' style='white-space: pre-wrap;'>{desc}</div>",
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
                        f"<div class='card-light {card_class}'><small>{entry.get(key, '—')}</small></div>",
                        unsafe_allow_html=True,
                    )

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
                    st.markdown(f"<div class='card-light-blue card-accent-coa'><small>{s['coa'] or '—'}</small></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<span class='pill pill-out'>Outcomes</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-light-blue card-accent-out'><small>{s['outcomes'] or '—'}</small></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<span class='pill pill-thm'>Theme</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-light-blue card-accent-thm'><small>{s['theme'] or '—'}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill pill-miss'>No aspects in cache</span>", unsafe_allow_html=True)

# ======================== PAGE: Error Analysis ========================
elif page == "Error Analysis":
    st.markdown("## ❌ Error Analysis")
    st.markdown("<div class='section-title'>Analyse model mistakes by aspect pattern</div>", unsafe_allow_html=True)
    triples = load_dev_triples()
    if not triples:
        st.warning("dev_track_a.jsonl not found.")
        st.stop()
    if triples[0].get("course_of_actions") is None:
        st.warning("Gold aspect labels not present in dev_track_a.jsonl. This page requires the post-release version with course_of_actions, outcomes, abstract_theme columns.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pred_filter = st.selectbox("Prediction", ["All", "Correct", "Incorrect"])
    with col2:
        coa_filter = st.selectbox("CoA pattern", ["Any", "Both match", "Only A", "Only B", "Neither"])
    with col3:
        out_filter = st.selectbox("Outcomes pattern", ["Any", "Both match", "Only A", "Only B", "Neither"])
    with col4:
        thm_filter = st.selectbox("Theme pattern", ["Any", "Both match", "Only A", "Only B", "Neither"])

    def asp_pattern(labels):
        if labels[0] is None:
            return "Unknown"
        return {(True, True): "Both match", (True, False): "Only A", (False, True): "Only B", (False, False): "Neither"}.get(tuple(labels), "Unknown")

    def simulate_prediction(triple):
        coa = triple.get("course_of_actions", [False, False])
        out = triple.get("outcomes", [False, False])
        thm = triple.get("abstract_theme", [False, False])
        if any(x is None for x in coa + out + thm):
            return None
        score_a = 0.35 * coa[0] + 0.40 * out[0] + 0.25 * thm[0]
        score_b = 0.35 * coa[1] + 0.40 * out[1] + 0.25 * thm[1]
        return score_a >= score_b

    rows = []
    for t in triples:
        pred = simulate_prediction(t)
        gold = t.get("text_a_is_closer")
        correct = (pred == gold) if pred is not None else None
        rows.append({
            "triple": t,
            "pred": pred,
            "gold": gold,
            "correct": correct,
            "coa_pat": asp_pattern(t.get("course_of_actions", [None, None])),
            "out_pat": asp_pattern(t.get("outcomes", [None, None])),
            "thm_pat": asp_pattern(t.get("abstract_theme", [None, None])),
        })

    def match_filter(row_val, filt):
        return filt == "Any" or row_val == filt

    filtered_rows = [r for r in rows if (pred_filter == "All" or (pred_filter == "Correct" and r["correct"] is True) or (pred_filter == "Incorrect" and r["correct"] is False)) and match_filter(r["coa_pat"], coa_filter) and match_filter(r["out_pat"], out_filter) and match_filter(r["thm_pat"], thm_filter)]

    total_valid = sum(1 for r in rows if r["correct"] is not None)
    n_correct = sum(1 for r in rows if r["correct"] is True)
    n_wrong = sum(1 for r in rows if r["correct"] is False)

    col_a, col_b, col_c, col_d = st.columns(4)
    for c, val, lbl in [(col_a, str(total_valid), "Total triples"), (col_b, f"{n_correct}", "Correct"), (col_c, f"{n_wrong}", "Incorrect"), (col_d, f"{len(filtered_rows)}", "Filtered")]:
        c.markdown(f"<div class='metric-box'><div class='metric-val'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Error distribution by aspect pattern")
    chart_cols = st.columns(3)
    for col, (asp_name, pat_key, color) in zip(chart_cols, [("CoA", "coa_pat", "#6e9ec8"), ("Outcomes", "out_pat", "#c8a96e"), ("Theme", "thm_pat", "#a86ec8")]):
        pat_counts = Counter(r[pat_key] for r in rows if r["correct"] is False)
        if pat_counts:
            labels = list(pat_counts.keys())
            vals = [pat_counts[l] for l in labels]
            fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=color, text=vals, textposition="outside", textfont=dict(family="JetBrains Mono", size=11, color="#e8e4dc")))
            fig.update_layout(title=dict(text=f"{asp_name} — errors by pattern", font=dict(family="Playfair Display", size=13, color="#e8e4dc")), paper_bgcolor="#14171f", plot_bgcolor="#14171f", font=dict(family="JetBrains Mono", color="#e8e4dc"), yaxis=dict(gridcolor="#1e2330"), margin=dict(t=40, b=20, l=10, r=10), height=250, showlegend=False)
            col.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown(f"<div class='section-title'>{len(filtered_rows)} triples matching filters</div>", unsafe_allow_html=True)
    show_n = min(len(filtered_rows), 20)
    for r in filtered_rows[:show_n]:
        t = r["triple"]
        correct = r["correct"]
        verdict = "<span class='badge-correct'>✓ Correct</span>" if correct else "<span class='badge-wrong'>✗ Incorrect</span>"
        with st.expander(f"{verdict.replace('<span ', '').split('>')[1].split('<')[0]}  ·  CoA: {r['coa_pat']} · Out: {r['out_pat']} · Thm: {r['thm_pat']}", expanded=False):
            st.markdown(verdict, unsafe_allow_html=True)
            col_anc, col_ab = st.columns([1, 2])
            with col_anc:
                st.markdown("**Anchor**")
                st.markdown(f"<div class='story-block'>{t['anchor_text'][:300]}…</div>", unsafe_allow_html=True)
            with col_ab:
                for label, text_field, idx in [("Text A", "text_a", 0), ("Text B", "text_b", 1)]:
                    gold_mark = " ← **Gold closer**" if ((idx == 0 and t["text_a_is_closer"]) or (idx == 1 and not t["text_a_is_closer"])) else ""
                    st.markdown(f"**{label}**{gold_mark}")
                    st.markdown(f"<div class='story-block'>{t.get(text_field,'')[:200]}…</div>", unsafe_allow_html=True)
            asp_html = "<div class='card' style='margin-top:8px'><table style='width:100%;font-family:JetBrains Mono;font-size:0.78em;border-collapse:collapse'>"
            asp_html += "<tr><th style='text-align:left;padding:3px 8px;color:#6b7280'>Aspect</th><th style='padding:3px 8px;color:#6b7280'>vs A</th><th style='padding:3px 8px;color:#6b7280'>vs B</th></tr>"
            for asp_name, field in [("CoA", "course_of_actions"), ("Outcomes", "outcomes"), ("Abstract Theme", "abstract_theme")]:
                lbl = t.get(field, [None, None])
                def cell(v):
                    if v is None: return "—"
                    return "✓" if v else "✗"
                asp_html += f"<tr><td style='padding:3px 8px'>{asp_name}</td><td style='padding:3px 8px;text-align:center;color:{'#6ec88a' if lbl[0] else '#c86e6e'}'>{cell(lbl[0])}</td><td style='padding:3px 8px;text-align:center;color:{'#6ec88a' if lbl[1] else '#c86e6e'}'>{cell(lbl[1])}</td></tr>"
            asp_html += "</table></div>"
            st.markdown(asp_html, unsafe_allow_html=True)

    if len(filtered_rows) > show_n:
        st.markdown(f"<div class='baseline-note'>Showing first {show_n} of {len(filtered_rows)} matching triples.</div>", unsafe_allow_html=True)