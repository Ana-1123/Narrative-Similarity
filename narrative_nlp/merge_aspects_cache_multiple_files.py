"""
Builds an optimal merged aspects cache **only for the stories that appear in the specified files**.
The input JSONL files may contain:
  - triples with anchor_text, text_a, text_b, OR
  - single text field.

Stories may appear multiple times; we take the unique set of story texts.

Merging strategy per aspect is based on the empirical informativeness analysis
(200 dev triples, RoBERTa-large cosine similarity):

  CoA      → V1 (aspects_cache_v1.json) wins:
               % pos>neg = 62.5%, mean_diff = 0.027, p = 0.006 ✓
               V2 gives 55.0% / p = 0.020 (significant but weaker)
               V3 gives 53.5% / p = 0.244 (not significant)

  Outcomes → V2 (aspects_cache_v2.json) wins:
               % pos>neg = 57.0%, p = 0.040 ✓
               V1 gives 51.5% / p = 0.081 (not significant)
               V3 gives 54.5% / p = 0.175 (not significant)

  Theme    → No version is statistically significant (best p = 0.197)
               Use V2 as fallback (least noisy)

Output: merged_aspects_cache.json (only for stories from the input files)
  Format: {story_text: {"coa": "...", "outcomes": "...", "theme": "..."}}
"""

import json
from pathlib import Path

DATA_DIR   = "./dataset/"

V1_PATH     = DATA_DIR + "aspects_cache_v1.json"
V2_PATH     = DATA_DIR + "aspects_cache_v2.json"
V3_PATH     = DATA_DIR + "aspects_cache_v3.json"
# List of JSONL files containing stories (dev + two others)
STORY_FILES = [
    DATA_DIR + "dev_track_a.jsonl",
    DATA_DIR + "test_track_a.jsonl",
    DATA_DIR + "test_track_b.jsonl"
]
OUT_PATH    = "./dataset/merged_aspects_cache_d_t_t.json"

# ── Which version to use for each aspect ─────────────────────────────────────
ASPECT_SOURCE = {
    "coa":      "v1",   # 62.5% discrimination, p=0.006
    "outcomes": "v2",   # 57.0% discrimination, p=0.040 — only significant version
    "theme":    "v2",   # No version significant; V2 least noisy (p=0.196 vs 0.889)
}


def _norm(text: str) -> str:
    """Normalise text: collapse whitespace, strip."""
    return " ".join(str(text).split())


def load_cache(path: str, label: str) -> dict:
    """Load a cache JSON and normalise its keys."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {label} not found at {path}")
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    cache = {_norm(k): v for k, v in raw.items()}
    print(f"  {label}: {len(cache)} entries loaded from {path}")
    return cache


def load_stories_from_files(file_paths: list) -> set:
    """
    Extract unique story texts from JSONL files.
    Supports two formats:
      1. {"text": "story content"}
      2. {"anchor_text": "...", "text_a": "...", "text_b": "..."}
    """
    stories = set()
    for file_path in file_paths:
        p = Path(file_path)
        if not p.exists():
            print(f"  WARNING: file not found: {file_path}")
            continue
        with open(p, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  Warning: skipping invalid JSON line {line_num} in {file_path}")
                    continue
                
                # Auto-detect format
                if "text" in record and isinstance(record["text"], str):
                    # Single text field (e.g., the new file)
                    stories.add(_norm(record["text"]))
                else:
                    # Triple fields format (original dev/train/val)
                    for field in ("anchor_text", "text_a", "text_b"):
                        text = record.get(field)
                        if text and isinstance(text, str):
                            stories.add(_norm(text))
        print(f"  Loaded from {file_path}: current total unique stories = {len(stories)}")
    return stories


def get_field(entry: dict, field: str) -> str:
    """
    Retrieve a field from a cache entry, handling variant key names.
    Returns empty string if not found.
    """
    if field == "coa":
        return (entry.get("coa") or
                entry.get("course_of_action") or "").strip()
    if field == "outcomes":
        return (entry.get("outcomes") or
                entry.get("outcome") or "").strip()
    if field == "theme":
        return (entry.get("theme") or
                entry.get("abstract_theme") or "").strip()
    return ""


def main():
    print("Loading caches …")
    caches = {
        "v1": load_cache(V1_PATH, "V1 (aspects_cache_v1)"),
        "v2": load_cache(V2_PATH, "V2 (aspects_cache_v2)"),
        "v3": load_cache(V3_PATH, "V3 (aspects_cache_v3)"),
    }

    # Load stories from all input files (dev + two others, possibly with single "text" field)
    all_stories = load_stories_from_files(STORY_FILES)
    if not all_stories:
        print("No stories found in any input file. Exiting.")
        return

    # Build merged cache only for these stories
    merged = {}
    stats = {
        "coa_source": {}, "outcomes_source": {}, "theme_source": {},
        "coa_missing": 0, "outcomes_missing": 0, "theme_missing": 0,
        "stories_not_in_any_cache": 0
    }

    for story_key in all_stories:
        entry = {}
        story_found = False
        for asp, src in ASPECT_SOURCE.items():
            # Try preferred source first, then fall back through remaining versions
            fallback_order = [src] + [v for v in ("v1", "v2", "v3") if v != src]
            chosen_src = None
            value = ""
            for v in fallback_order:
                if story_key in caches[v]:
                    candidate = get_field(caches[v][story_key], asp)
                    if candidate:
                        value = candidate
                        chosen_src = v
                        story_found = True
                        break
            entry[asp] = value
            if chosen_src:
                stats[f"{asp}_source"][chosen_src] = \
                    stats[f"{asp}_source"].get(chosen_src, 0) + 1
            else:
                stats[f"{asp}_missing"] += 1

        if not story_found:
            stats["stories_not_in_any_cache"] += 1

        merged[story_key] = entry

    # Save merged cache
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nMerged cache written: {OUT_PATH}")
    print(f"Total entries (all input stories): {len(merged)}")

    # Coverage report
    print("\n── Coverage report (all input stories) ──────────────────────────────")
    print(f"{'Aspect':<12} {'V1 used':>10} {'V2 used':>10} {'V3 used':>10} "
          f"{'Missing':>10}")
    for asp in ("coa", "outcomes", "theme"):
        src_counts = stats[f"{asp}_source"]
        missing    = stats[f"{asp}_missing"]
        print(f"  {asp:<10} "
              f"{src_counts.get('v1',0):>10} "
              f"{src_counts.get('v2',0):>10} "
              f"{src_counts.get('v3',0):>10} "
              f"{missing:>10}")

    print(f"\n  Stories not found in any cache: {stats['stories_not_in_any_cache']} / {len(all_stories)}")

    print("\nMerge strategy:")
    for asp, src in ASPECT_SOURCE.items():
        src_label = {"v1": "V1", "v2": "V2", "v3": "V3"}[src]
        print(f"  {asp:<10} → primary: {src_label}")

    # Spot-check up to 3 entries
    print("\n── Spot-check (first 3 input stories) ────────────────────────────────")
    for i, (k, v) in enumerate(list(merged.items())[:3]):
        print(f"\n  Story: {k[:80]}…")
        print(f"  CoA    : {v['coa'][:120]}")
        print(f"  Outcomes: {v['outcomes'][:100]}")
        print(f"  Theme  : {v['theme'][:80]}")
    print()


if __name__ == "__main__":
    main()