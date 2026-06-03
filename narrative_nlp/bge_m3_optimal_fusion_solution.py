"""
BGE-M3 Optimal Fusion Solution - Kaggle Edition
Conditions A-H + Track B concatenated embeddings
Best versions: CoA V2 (role-label steps), Outcomes V1 (narrative prose)
"""

%env REPRODUCIBLE_SEED=42
%env STRICT_DETERMINISM=1
%env CUBLAS_WORKSPACE_CONFIG=:4096:8
%env DL_NUM_WORKERS=0

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

_TORCH_WAS_PRELOADED = "torch" in sys.modules
_CUBLAS_ENV_WAS_PRESET = "CUBLAS_WORKSPACE_CONFIG" in os.environ

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ===================== REPRODUCIBILITY / SEEDING =====================
SEED = int(os.environ.get("REPRODUCIBLE_SEED", 42))
STRICT_DETERMINISM = os.environ.get("STRICT_DETERMINISM", "1").strip().lower() not in {"0", "false", "no"}
STRICT_DETERMINISM_ENABLED = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _disable_strict_determinism(reason):
    global STRICT_DETERMINISM, STRICT_DETERMINISM_ENABLED
    STRICT_DETERMINISM = False
    STRICT_DETERMINISM_ENABLED = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    print("Warning: strict deterministic algorithms disabled.")
    print("         Reason:", reason)

def configure_determinism():
    global STRICT_DETERMINISM_ENABLED
    if not STRICT_DETERMINISM:
        print("Torch deterministic algorithms: DISABLED (STRICT_DETERMINISM=0)")
        return

    if torch.cuda.is_available() and _TORCH_WAS_PRELOADED and not _CUBLAS_ENV_WAS_PRESET:
        _disable_strict_determinism(
            "torch was already imported before CUBLAS_WORKSPACE_CONFIG was set."
        )
        return

    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    valid_cfg = {":4096:8", ":16:8"}
    if torch.cuda.is_available() and cublas_cfg not in valid_cfg:
        _disable_strict_determinism(
            f"unsupported CUBLAS_WORKSPACE_CONFIG='{cublas_cfg}'. Use :4096:8 or :16:8."
        )
        return

    try:
        torch.use_deterministic_algorithms(True)
        STRICT_DETERMINISM_ENABLED = True
        print("Torch deterministic algorithms: ENABLED")
    except Exception as e:
        _disable_strict_determinism(str(e))

set_seed(SEED)
configure_determinism()

DL_NUM_WORKERS = int(os.environ.get("DL_NUM_WORKERS", 0))
dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)

def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)

# ===================== CONFIGURATION =====================
CONDITION = "A"  # Change to A, B, C, D, E, F, G, H

SYNTH_PATH          = "/kaggle/input/synthetic_data_for_classification.jsonl"
DEV_TRACK_A_PATH    = "/kaggle/input/dev_track_a.jsonl"
TEST_TRACK_A_PATH   = "/kaggle/input/test_track_a.jsonl"
TEST_TRACK_B_PATH   = "/kaggle/input/test_track_b.jsonl"
TEST_TRACK_A_LABELS = "/kaggle/input/test_track_a_labels.jsonl"
TEST_TRACK_B_LABELS = "/kaggle/input/test_track_b_labels.jsonl"

ASPECTS_CACHE_V1    = "/kaggle/input/aspects_cache_v1.json"
ASPECTS_CACHE_V2    = "/kaggle/input/aspects_cache_v2.json"
ASPECTS_CACHE_V3    = "/kaggle/input/aspects_cache_v3.json"

OUT_DIR = "/kaggle/working/"

MODEL_NAME = "BAAI/bge-m3"

STAGE1_EPOCHS = 1
STAGE2_EPOCHS = 2
STAGE1_LR     = 1e-5
STAGE2_LR     = 6e-6
STAGE1_BS     = 2
STAGE2_BS     = 4
PRED_BS       = 16
ENC_BS        = 8
MARGIN        = 0.3

USE_SYNTHETIC_STAGE1 = True
RUN_CV = False
CV_SPLITS = 5
RUN_FINAL_TEST = True

LABEL_SMOOTH_POS = 0.90
LABEL_SMOOTH_NEG = 0.10

HIDDEN_DIM = 1024
PROJ_DIM = 256

# Condition configurations for BGE-M3 (CoA V2, Outcomes V1)
_CONDITION_CONFIGS = {
    "A": ("Baseline", "full_text", 256, [], False),
    "B": ("CoA V2 only", "coa_v2", 256, [], False),
    "C": ("Outcomes V1 only", "outcomes_v1", 256, [], False),
    "D": ("CoA V2 + Outcomes V1 concat", "concat_v2_v1", 256, [], False),
    "E": ("Full text + CoA V2 head", "full_text", 256, ["coa_v2"], False),
    "F": ("Full text + Outcomes V1 head", "full_text", 256, ["outcomes_v1"], False),
    "G": ("Full text + CoA V2 + Outcomes V1 heads", "full_text", 256, ["coa_v2", "outcomes_v1"], False),
    "H": ("Full text + CoA V2/Out V1 appended", "full_text_plus", 384, [], False),
}

_CONDITION_NOTES = {
    "A": "BGE-M3 baseline: full text only",
    "B": "CoA V2 (role-label steps) as input",
    "C": "Outcomes V1 (narrative prose) as input",
    "D": "Concatenated aspects: CoA V2 + Outcomes V1",
    "E": "Aspect head encoding for CoA V2",
    "F": "Aspect head encoding for Outcomes V1",
    "G": "Multi-aspect heads: CoA V2 + Outcomes V1",
    "H": "Hybrid appended aspects: full + CoA V2 + Outcomes V1",
}

assert CONDITION in _CONDITION_CONFIGS
_cname, INPUT_MODE, MAX_LEN, ASPECTS, USE_SOFT_LABELS = _CONDITION_CONFIGS[CONDITION]
_cnote = _CONDITION_NOTES.get(CONDITION, "")
USE_ASPECT_HEADS = bool(ASPECTS)
ACTIVE_ASPECTS = list(ASPECTS)

if RUN_FINAL_TEST and not RUN_CV:
    suffix = "full_train"
elif RUN_CV:
    suffix = f"cv_{CV_SPLITS}folds"
else:
    suffix = "results"

RUN_PREFIX = f"BGE-M3_Condition_{CONDITION}_{MAX_LEN}_{suffix}"

OUT_TRACK_A = os.path.join(OUT_DIR, f"{RUN_PREFIX}_track_a.jsonl")
OUT_TRACK_B = os.path.join(OUT_DIR, f"{RUN_PREFIX}_track_b.npy")
METRICS_JSON = os.path.join(OUT_DIR, f"{RUN_PREFIX}_eval_metrics.json")
CV_RESULTS_JSON = os.path.join(OUT_DIR, f"{RUN_PREFIX}_cv_results.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ===================== ASPECT CACHE LOADING =====================
def _norm_key(text):
    return " ".join(str(text).split())

def _read_json_or_jsonl(path):
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except Exception:
            pass
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_aspects_cache(path):
    raw = _read_json_or_jsonl(path)
    if raw is None:
        return {}
    cache = {}
    def add_record(story_text, payload):
        if not story_text:
            return
        cache[_norm_key(story_text)] = {
            "coa": payload.get("coa") or "",
            "outcomes": payload.get("outcomes") or "",
            "theme": payload.get("theme") or "",
            "title": payload.get("title", ""),
        }
    if isinstance(raw, list):
        for obj in raw:
            if isinstance(obj, dict):
                add_record(obj.get("text") or obj.get("raw_text") or obj.get("story"), obj)
    elif isinstance(raw, dict):
        if "records" in raw and isinstance(raw["records"], list):
            for obj in raw["records"]:
                if isinstance(obj, dict):
                    add_record(obj.get("text") or obj.get("raw_text") or obj.get("story"), obj)
        else:
            for k, v in raw.items():
                if isinstance(v, dict):
                    add_record(k, v)
    return cache

ASPECTS_CACHE_V1_DATA = load_aspects_cache(ASPECTS_CACHE_V1)
ASPECTS_CACHE_V2_DATA = load_aspects_cache(ASPECTS_CACHE_V2)
ASPECTS_CACHE_V3_DATA = load_aspects_cache(ASPECTS_CACHE_V3)

def _nonempty_parts(parts):
    return [p for p in parts if str(p).strip()]

def get_story_input(text, mode):
    """Get input text based on condition mode."""
    if mode == "full_text":
        return text

    entry_v1 = ASPECTS_CACHE_V1_DATA.get(_norm_key(text), {})
    entry_v2 = ASPECTS_CACHE_V2_DATA.get(_norm_key(text), {})

    coa_v2 = entry_v2.get("coa", text) or text
    outcomes_v1 = entry_v1.get("outcomes", text) or text

    if mode == "coa_v2":
        return coa_v2
    if mode == "outcomes_v1":
        return outcomes_v1
    if mode == "concat_v2_v1":
        parts = _nonempty_parts([coa_v2, outcomes_v1])
        return " [SEP] ".join(parts) if parts else text
    if mode == "full_text_plus":
        parts = _nonempty_parts([text, coa_v2, outcomes_v1])
        return " [SEP] ".join(parts)
    return text

def get_story_views(text):
    """Get all views for aspect encoding."""
    entry_v1 = ASPECTS_CACHE_V1_DATA.get(_norm_key(text), {})
    entry_v2 = ASPECTS_CACHE_V2_DATA.get(_norm_key(text), {})
    return {
        "full_text": text,
        "coa_v2": entry_v2.get("coa", text) or text,
        "outcomes_v1": entry_v1.get("outcomes", text) or text,
    }

def batch_to_views(text_list):
    views = [get_story_views(t) for t in text_list]
    return {
        "full_text": [v["full_text"] for v in views],
        "coa_v2": [v["coa_v2"] for v in views],
        "outcomes_v1": [v["outcomes_v1"] for v in views],
    }

# ===================== MODEL =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE == "cuda":
    encoder.gradient_checkpointing_enable()
    print("Gradient checkpointing: ENABLED")
else:
    print("Gradient checkpointing: DISABLED (CPU)")

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

def mean_pool(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_emb * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2, dim=1)

class BGEM3Model(nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim, aspect_keys):
        super().__init__()
        self.encoder = encoder
        self.aspect_keys = list(aspect_keys) if aspect_keys else []
        for key in self.aspect_keys:
            if key == "coa_v2":
                self.head_coa_v2 = nn.Linear(hidden_dim, proj_dim)
            elif key == "outcomes_v1":
                self.head_outcomes_v1 = nn.Linear(hidden_dim, proj_dim)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out, attention_mask)

    def forward(self, input_ids, attention_mask):
        emb = self.encode(input_ids, attention_mask)
        if not self.aspect_keys:
            return emb
        out = {"global": emb}
        for key in self.aspect_keys:
            if key == "coa_v2":
                out["coa_v2"] = F.normalize(self.head_coa_v2(emb), p=2, dim=1)
            elif key == "outcomes_v1":
                out["outcomes_v1"] = F.normalize(self.head_outcomes_v1(emb), p=2, dim=1)
        return out

def build_model():
    return BGEM3Model(encoder, HIDDEN_DIM, PROJ_DIM, ACTIVE_ASPECTS).to(DEVICE)

model = build_model()

# ===================== DATASETS =====================
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

class SyntheticTripletDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                anchor = obj.get("anchor_text", "")
                ta = obj.get("text_a", "")
                tb = obj.get("text_b", "")
                closer = obj.get("text_a_is_closer")
                if not anchor or not ta or not tb:
                    continue
                if len(anchor) < 20 or len(ta) < 20 or len(tb) < 20:
                    continue
                pos, neg = (ta, tb) if closer else (tb, ta)
                self.data.append((get_story_input(anchor, INPUT_MODE),
                                  get_story_input(pos, INPUT_MODE),
                                  get_story_input(neg, INPUT_MODE)))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

class TrackARowsDataset(Dataset):
    def __init__(self, rows):
        self.rows_eval = list(rows)
        self.data = []
        for obj in rows:
            self.data.append((get_story_input(obj["anchor_text"], INPUT_MODE),
                              get_story_input(obj["text_a"], INPUT_MODE),
                              get_story_input(obj["text_b"], INPUT_MODE),
                              bool(obj["text_a_is_closer"])))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

class TrackATestDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append((get_story_input(obj["anchor_text"], INPUT_MODE),
                                  get_story_input(obj["text_a"], INPUT_MODE),
                                  get_story_input(obj["text_b"], INPUT_MODE)))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ===================== LOSSES & HELPERS =====================
def _get_global_emb(output):
    return output["global"] if isinstance(output, dict) else output

def _triplet_loss(a, p, n, margin=MARGIN):
    return F.triplet_margin_loss(a, p, n, margin=margin)

def _rank_loss(a, p, n):
    return F.relu(F.cosine_similarity(a, n).mean() - F.cosine_similarity(a, p).mean() + 0.2)

def _aspect_triplet_loss(out_a, out_p, out_n):
    if not ACTIVE_ASPECTS:
        return torch.tensor(0.0, device=DEVICE)
    return sum(_triplet_loss(out_a[h], out_p[h], out_n[h]) for h in ACTIVE_ASPECTS) / float(len(ACTIVE_ASPECTS))

# ===================== TRAINING & EVALUATION =====================
def train_stage1(model, dataset, epochs=STAGE1_EPOCHS):
    if len(dataset) == 0:
        return model
    loader = DataLoader(dataset, batch_size=STAGE1_BS, shuffle=True,
                        num_workers=DL_NUM_WORKERS, worker_init_fn=_worker_init_fn, generator=dl_generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE1_LR)
    model.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"Stage1 Epoch {ep+1}")
        running = 0.0
        for anchors, positives, negatives in loop:
            texts = list(anchors) + list(positives) + list(negatives)
            enc = tokenize(texts)
            out = model(enc["input_ids"], enc["attention_mask"])
            B = len(anchors)
            if USE_ASPECT_HEADS:
                ga, gp, gn = out["global"][:B], out["global"][B:2*B], out["global"][2*B:]
                a_heads = {k: out[k][:B] for k in ACTIVE_ASPECTS}
                p_heads = {k: out[k][B:2*B] for k in ACTIVE_ASPECTS}
                n_heads = {k: out[k][2*B:] for k in ACTIVE_ASPECTS}
                loss = (0.7 * _triplet_loss(ga, gp, gn) +
                        0.3 * _rank_loss(ga, gp, gn) +
                        0.3 * _aspect_triplet_loss(a_heads, p_heads, n_heads))
            else:
                a, p, n = out[:B], out[B:2*B], out[2*B:]
                loss = 0.7 * _triplet_loss(a, p, n) + 0.3 * _rank_loss(a, p, n)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            loop.set_postfix(loss=f"{running / (loop.n + 1):.4f}")
    model.eval()
    return model

def _macro_f1_from_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(( y_pred &  y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    fp = int(( y_pred & ~y_true).sum())
    fn = int((~y_pred &  y_true).sum())
    f1_pos = 2 * tp / (2 * tp + fp + fn + 1e-9)
    f1_neg = 2 * tn / (2 * tn + fn + fp + 1e-9)
    return (f1_pos + f1_neg) / 2.0, tp, tn, fp, fn

def evaluate_trackA_rows(model, rows):
    ds = TrackARowsDataset(rows)
    loader = DataLoader(ds, batch_size=PRED_BS, shuffle=False,
                        num_workers=DL_NUM_WORKERS, worker_init_fn=_worker_init_fn)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for anchors, ta_list, tb_list, labels in loader:
            texts = list(anchors) + list(ta_list) + list(tb_list)
            enc = tokenize(texts)
            out = model(enc["input_ids"], enc["attention_mask"])
            B = len(anchors)
            g = _get_global_emb(out)
            a, ta, tb = g[:B], g[B:2*B], g[2*B:]
            pred = (F.cosine_similarity(a, ta) > F.cosine_similarity(a, tb)).cpu().tolist()
            y_pred.extend([bool(x) for x in pred])
            y_true.extend([bool(x) for x in labels])
    acc = accuracy_score(y_true, y_pred)
    macro_f1, tp, tn, fp, fn = _macro_f1_from_binary(y_true, y_pred)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def train_stage2(model, dataset, epochs=STAGE2_EPOCHS):
    loader = DataLoader(dataset, batch_size=STAGE2_BS, shuffle=True,
                        num_workers=DL_NUM_WORKERS, worker_init_fn=_worker_init_fn, generator=dl_generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"Stage2 Epoch {ep+1}")
        running = 0.0
        for anchors, ta_list, tb_list, closer_list in loop:
            y = torch.tensor(closer_list, dtype=torch.float32, device=DEVICE)
            if USE_SOFT_LABELS:
                y = y * LABEL_SMOOTH_POS + (1.0 - y) * LABEL_SMOOTH_NEG
            texts = list(anchors) + list(ta_list) + list(tb_list)
            enc = tokenize(texts)
            out = model(enc["input_ids"], enc["attention_mask"])
            B = len(anchors)
            g = _get_global_emb(out)
            a, ta, tb = g[:B], g[B:2*B], g[2*B:]
            score = F.cosine_similarity(a, ta) - F.cosine_similarity(a, tb)
            loss = criterion(score, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            loop.set_postfix(loss=f"{running / (loop.n + 1):.4f}")

        epoch_eval = evaluate_trackA_rows(model, dataset.rows_eval)
        print(
            f"Stage2 Epoch {ep+1} summary | "
            f"loss={running / max(len(loader), 1):.4f} | "
            f"acc={epoch_eval['accuracy']*100:.2f}% | "
            f"macro_f1={epoch_eval['macro_f1']*100:.2f}%"
        )
    model.eval()
    return model

def evaluate_trackB_style_rows(model, rows):
    unique_texts, seen = [], set()
    for obj in rows:
        for key in ["anchor_text", "text_a", "text_b"]:
            txt = obj[key].strip()
            if txt not in seen:
                seen.add(txt)
                unique_texts.append(txt)
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(unique_texts), ENC_BS):
            batch_raw = unique_texts[i:i+ENC_BS]
            batch_inp = [get_story_input(t, INPUT_MODE) for t in batch_raw]
            enc = tokenize(batch_inp)
            out = model(enc["input_ids"], enc["attention_mask"])
            emb = torch.cat([out["global"]] + [out[k] for k in ACTIVE_ASPECTS], dim=1) if USE_ASPECT_HEADS else _get_global_emb(out)
            emb = F.normalize(emb, p=2, dim=1) if USE_ASPECT_HEADS else emb
            embs.append(emb.cpu().numpy().astype(np.float32))
    X = np.vstack(embs)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    lookup = {t: X[i] for i, t in enumerate(unique_texts)}
    y_true, y_pred = [], []
    for obj in rows:
        a = lookup[obj["anchor_text"].strip()]
        ta = lookup[obj["text_a"].strip()]
        tb = lookup[obj["text_b"].strip()]
        y_pred.append(float(np.dot(a, ta)) > float(np.dot(a, tb)))
        y_true.append(bool(obj["text_a_is_closer"]))
    acc = accuracy_score(y_true, y_pred)
    macro_f1, tp, tn, fp, fn = _macro_f1_from_binary(y_true, y_pred)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def run_cross_validation(initial_model_state, dev_path, n_splits=5):
    rows = load_jsonl(dev_path)
    labels = [int(bool(r["text_a_is_closer"])) for r in rows]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(rows, labels), start=1):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        fold_model = build_model()
        fold_model.load_state_dict(initial_model_state)
        fold_model = train_stage2(fold_model, TrackARowsDataset(train_rows), epochs=STAGE2_EPOCHS)
        res_a = evaluate_trackA_rows(fold_model, val_rows)
        res_b = evaluate_trackB_style_rows(fold_model, val_rows)
        fold_results.append({"fold": fold_id, "track_a_accuracy": res_a["accuracy"], "track_a_macro_f1": res_a["macro_f1"],
                             "track_b_style_accuracy": res_b["accuracy"], "track_b_style_macro_f1": res_b["macro_f1"],
                             "n_train": len(train_rows), "n_val": len(val_rows)})
    summary = {"condition": CONDITION, "name": _cname, "condition_note": _cnote, "n_splits": n_splits,
               "track_a_cv_mean": float(np.mean([r["track_a_accuracy"] for r in fold_results])),
               "track_a_cv_std": float(np.std([r["track_a_accuracy"] for r in fold_results])),
               "track_b_style_cv_mean": float(np.mean([r["track_b_style_accuracy"] for r in fold_results])),
               "track_b_style_cv_std": float(np.std([r["track_b_style_accuracy"] for r in fold_results])),
               "folds": fold_results}
    with open(CV_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def write_trackA_predictions(model, test_path, out_path):
    ds = TrackATestDataset(test_path)
    loader = DataLoader(ds, batch_size=PRED_BS, shuffle=False,
                        num_workers=DL_NUM_WORKERS, worker_init_fn=_worker_init_fn)
    preds = []
    model.eval()
    with torch.no_grad():
        for anchors, ta_list, tb_list in loader:
            texts = list(anchors) + list(ta_list) + list(tb_list)
            enc = tokenize(texts)
            out = model(enc["input_ids"], enc["attention_mask"])
            B = len(anchors)
            g = _get_global_emb(out)
            a, ta, tb = g[:B], g[B:2*B], g[2*B:]
            batch_preds = (F.cosine_similarity(a, ta) > F.cosine_similarity(a, tb)).cpu().tolist()
            preds.extend(batch_preds)
    with open(test_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line, p in zip(fin, preds):
            obj = json.loads(line)
            obj["text_a_is_closer"] = bool(p)
            fout.write(json.dumps(obj) + "\n")

def build_trackB_embeddings(model, in_path, out_npy):
    """Track B: concatenated embeddings [e_full, e_coa_v2, e_out_v1]"""
    texts = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), ENC_BS):
            batch_raw = texts[i:i+ENC_BS]
            batch_inp = [get_story_input(t, INPUT_MODE) for t in batch_raw]
            enc = tokenize(batch_inp)
            out = model(enc["input_ids"], enc["attention_mask"])
            if USE_ASPECT_HEADS:
                emb = torch.cat([out["global"]] + [out[k] for k in ACTIVE_ASPECTS], dim=1)
            else:
                emb = _get_global_emb(out)
            emb = F.normalize(emb, p=2, dim=1)
            embs.append(emb.cpu().numpy().astype(np.float32))
    X = np.vstack(embs).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    np.save(out_npy, X)

def evaluate_trackA(predictions_path, gold_labels_path):
    pred = pd.read_json(predictions_path, lines=True)
    gold = pd.read_json(gold_labels_path, lines=True)
    y_pred = pred["text_a_is_closer"].astype(bool).tolist()
    y_true = gold["text_a_is_closer"].astype(bool).tolist()
    acc = accuracy_score(y_true, y_pred)
    macro_f1, tp, tn, fp, fn = _macro_f1_from_binary(y_true, y_pred)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def evaluate_trackB(track_b_input_path, embeddings_npy, gold_labels_path):
    pred = pd.read_json(track_b_input_path, lines=True)
    embs = np.load(embeddings_npy, allow_pickle=False)
    normed = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12, None)
    lookup = dict(zip(pred["text"], normed))
    df = pd.read_json(gold_labels_path, lines=True)
    for col in ["anchor_text", "text_a", "text_b"]:
        df[col] = df[col].astype(str).str.strip()
    df["anchor_embedding"] = df["anchor_text"].map(lookup)
    df["a_embedding"] = df["text_a"].map(lookup)
    df["b_embedding"] = df["text_b"].map(lookup)
    missing = int((df["anchor_embedding"].isna() | df["a_embedding"].isna() | df["b_embedding"].isna()).sum())
    if missing:
        df = df[~(df["anchor_embedding"].isna() | df["a_embedding"].isna() | df["b_embedding"].isna())].copy()
    y_true = df["text_a_is_closer"].astype(bool).tolist()
    y_pred = [float(r["anchor_embedding"].dot(r["a_embedding"])) > float(r["anchor_embedding"].dot(r["b_embedding"])) for _, r in df.iterrows()]
    acc = accuracy_score(y_true, y_pred)
    macro_f1, tp, tn, fp, fn = _macro_f1_from_binary(y_true, y_pred)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "missing": missing, "embedding_dim": int(normed.shape[1])}

def print_summary(res_a, res_b, cv_summary=None):
    payload = {"condition": CONDITION, "name": _cname, "condition_note": _cnote,
               "input_mode": INPUT_MODE, "max_len": MAX_LEN, "use_aspect_heads": USE_ASPECT_HEADS,
               "active_aspects": ACTIVE_ASPECTS, "use_soft_labels": USE_SOFT_LABELS,
               "track_a_test": res_a, "track_b_test": res_b, "cv": cv_summary}
    print("\n" + "="*60)
    print(f"  BGE-M3 FINAL SUMMARY — Condition {CONDITION}: {_cname}")
    print("="*60)
    print(f"  Track A accuracy : {res_a['accuracy']*100:.2f}%")
    print(f"  Track A macro-F1 : {res_a['macro_f1']*100:.2f}%")
    print(f"  Track B accuracy : {res_b['accuracy']*100:.2f}%")
    print(f"  Track B macro-F1 : {res_b['macro_f1']*100:.2f}%")
    print("="*60)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# ===================== RUN =====================
print(f"Starting BGE-M3 run for condition {CONDITION}: {_cname}")
print(f"Condition note: {_cnote}")

if USE_SYNTHETIC_STAGE1:
    if DEVICE != "cpu":
        synth_ds = SyntheticTripletDataset(SYNTH_PATH)
        model = train_stage1(model, synth_ds, epochs=STAGE1_EPOCHS)

initial_model_state = deepcopy(model.state_dict())

cv_summary = run_cross_validation(initial_model_state, DEV_TRACK_A_PATH, n_splits=CV_SPLITS) if RUN_CV else None

if RUN_FINAL_TEST:
    model.load_state_dict(initial_model_state)
    full_dev_rows = load_jsonl(DEV_TRACK_A_PATH)
    model = train_stage2(model, TrackARowsDataset(full_dev_rows), epochs=STAGE2_EPOCHS)
    write_trackA_predictions(model, TEST_TRACK_A_PATH, OUT_TRACK_A)
    build_trackB_embeddings(model, TEST_TRACK_B_PATH, OUT_TRACK_B)
    res_a = evaluate_trackA(OUT_TRACK_A, TEST_TRACK_A_LABELS)
    res_b = evaluate_trackB(TEST_TRACK_B_PATH, OUT_TRACK_B, TEST_TRACK_B_LABELS)
    print_summary(res_a, res_b, cv_summary=cv_summary)
