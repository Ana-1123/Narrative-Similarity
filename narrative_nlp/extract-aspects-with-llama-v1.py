"""
Aspect extraction

Extracts three narrative aspect descriptions per story:
  • Course of Action (CoA)  — what happens, in order
  • Outcomes               — the final resolution
  • Abstract Theme         — universal motifs and patterns

Output: aspects_cache_llm_extracted.json
  {
    "<story_text>": {
        "title":    "...",
        "coa":      "...",
        "outcomes": "...",
        "theme":    "..."
    },
    ...
  }
"""

import os, sys, json, time, random, re
from pathlib import Path
from tqdm import tqdm

DATA_FILES = {
    "dev_track_a":  "dev_track_a.jsonl",    # 200 triples
    "test_track_a": "test_track_a.jsonl",   # 400 triples
    "test_track_b": "test_track_b.jsonl",   # 849 individual story texts
}

CACHE_PATH  = "aspects_cache_v2.json"
OLLAMA_MODEL  = "llama3.1:8b"
OLLAMA_URL    = "http://localhost:11434/api/generate"

# Prompts

COA_PROMPT = """\
You are a narrative analyst. Read the story summary below and write ONLY \
the sequence of plot events — what happens, in what order, and what causes \
what. Do NOT mention character names, specific locations, or themes. \
Do NOT write any introduction, heading, or label before your answer. \
Begin your response immediately with the first event. Write 2-4 sentences.

Story:
{story}

Response:"""

OUTCOMES_PROMPT = """\
You are a narrative analyst. Read the story summary below and write ONLY \
the final outcome and resolution. What is the end state? What did the \
protagonist ultimately achieve, lose, or experience? Do NOT describe how \
they got there. Do NOT write any introduction, heading, or label. \
Begin your response immediately with the outcome. Write 1-2 sentences.

Story:
{story}

Response:"""

THEME_PROMPT = """\
You are a narrative analyst. Read the story summary below and write ONLY \
the abstract themes and universal human experiences it explores. What \
fundamental aspects of human nature, society, or morality does it examine? \
Do NOT mention specific characters, places, or plot events. \
Do NOT write any introduction, heading, or label. \
Begin your response immediately with the theme. Write 1-3 sentences.

Story:
{story}

Response:"""

PROMPTS = {
    "coa":      COA_PROMPT,
    "outcomes": OUTCOMES_PROMPT,
    "theme":    THEME_PROMPT,
}


# Story collection — deduplicate across all datasets

def collect_stories(data_files):
    """
    Walk all available dataset files and collect every unique story text.
    Returns a list of dicts: [{text, title, source}, ...]
    Deduplication is by exact text string after stripping whitespace.
    """
    seen   = set()
    stories = []

    def _add(text, title, source):
        # Guard against explicit JSON null values
        if text is None:
            return
        key = _normalise_key(text)   # same normalisation as cache keys
        if key and key not in seen and len(key) >= 20:
            seen.add(key)
            stories.append({"text": key, "title": title, "source": source})

    for name, path in data_files.items():
        if not Path(path).exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  Reading {path} ...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                if name == "test_track_b":
                    # Track B input: one story per row with "text" field
                    _add(obj.get("text",""),
                         obj.get("story_details",{}).get("title",""),
                         name)
                else:
                    # Track A / synthetic: triples with anchor_text, text_a, text_b
                    details = {
                        "anchor_text": obj.get("story_details_anchor",{}).get("title",""),
                        "text_a":      obj.get("story_details_a",{}).get("title",""),
                        "text_b":      obj.get("story_details_b",{}).get("title",""),
                    }
                    for field, title_key in [
                        ("anchor_text", "anchor_text"),
                        ("text_a",      "text_a"),
                        ("text_b",      "text_b"),
                    ]:
                        _add(obj.get(field,""), details[title_key], name)

    print(f"\nTotal unique stories collected: {len(stories)}")
    return stories


# Cache helpers

def _normalise_key(text):
    """Canonical cache key: strip whitespace, normalise internal spaces."""
    return " ".join(str(text).split())

def load_cache(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Re-key with normalised keys so laptop and Kaggle entries match
        cache = {_normalise_key(k): v for k, v in raw.items()}
        print(f"Cache loaded: {len(cache)} entries from {path}")
        return cache
    return {}


def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def is_complete(entry):
    """An entry is complete if all three aspects are non-empty strings."""
    return (
        isinstance(entry, dict) and
        all(entry.get(k,"").strip() for k in ["coa", "outcomes", "theme"])
    )


# Backend: Ollama (local)

def _ollama_available():
    try:
        import urllib.request
        urllib.request.urlopen(OLLAMA_URL.replace("/api/generate", ""), timeout=2)
        return True
    except Exception:
        return False


def check_ollama_gpu():
    """
    Print Ollama GPU status. If Ollama is running on CPU instead of GPU,
    extraction will be ~10x slower. Run this before starting extraction.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(
            OLLAMA_URL.replace("/api/generate", "/api/ps"), timeout=5
        ) as resp:
            ps = json.loads(resp.read())
        models = ps.get("models", [])
        if models:
            for m in models:
                size_vram = m.get("size_vram", 0)
                print(f"  Ollama model: {m.get('name')}  "
                      f"VRAM: {size_vram/1e9:.1f}GB  "
                      f"({'GPU ✓' if size_vram > 0 else 'CPU ⚠ — slow!'})")
        else:
            print("  Ollama running but no model loaded yet (loads on first call)")
    except Exception:
        pass  # ps endpoint may not exist on older Ollama versions


def _ollama_generate(prompt, model=OLLAMA_MODEL, max_tokens=200):
    import urllib.request, urllib.error
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,   # low temp for factual extraction
            "top_p": 0.9,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result.get("response", "").strip()



# Response cleaning
# Llama-3.1-8B sometimes echoes a preamble like "Here is the sequence of
# plot events:" before the actual answer, despite explicit prompt instructions.
# This function removes all known preamble patterns and normalises whitespace.

_PREAMBLE_RE = re.compile(
    r'^(?:'
    r'(?:here (?:is|are) (?:the )?(?:a )?'
    r'(?:sequence of plot events|extracted plot events|following|'
    r'abstract themes?.*?|course of action.*?|outcome.*?|theme.*?))'
    r'[:\.!]?\s*\n+'
    r')',
    re.IGNORECASE | re.DOTALL
)

# Also strip leading label like "Course of Action:" or "Response:"
_LABEL_RE = re.compile(
    r'^(?:Course of Action|Outcome(?:s)?|Abstract Theme|Response)\s*[:：]\s*',
    re.IGNORECASE
)

def _clean_response(text):
    """Remove preamble echoes and leading labels from LLM responses."""
    text = text.strip()
    # Iteratively remove preamble lines (sometimes nested)
    for _ in range(3):
        cleaned = _PREAMBLE_RE.sub('', text).strip()
        if cleaned == text:
            break
        text = cleaned
    # Remove any remaining leading label
    text = _LABEL_RE.sub('', text).strip()
    # Normalise internal whitespace — collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def extract_with_ollama(story_text, title=""):
    aspects = {}
    for aspect, prompt_template in PROMPTS.items():
        prompt = prompt_template.format(story=story_text)
        try:
            response = _ollama_generate(prompt)
            # Clean up: remove any accidental repetition of the prompt keyword
            response = _clean_response(response)
            aspects[aspect] = response
        except Exception as e:
            aspects[aspect] = ""
            print(f"    [ollama error] {aspect}: {e}")
    aspects["title"] = title
    return aspects


SYSTEM_MSG = (
    "You are a precise narrative analyst. "
    "Always follow instructions exactly. "
    "Never add introductions, headings, labels, or phrases like "
    "'Here is...' or 'Here are...' before your answer. "
    "Start your response immediately with the requested content."
)


def _build_messages(prompt):
    return [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt}]


def _extract_reply(output):
    generated = output["generated_text"]
    if isinstance(generated, list):
        return generated[-1].get("content", "").strip()
    return str(generated).strip()

# Backend auto-detection and dispatch

def detect_backend(forced=None):
    if forced == "ollama":
        if not _ollama_available():
            print("ERROR: Ollama not running. Start it with: ollama serve")
            sys.exit(1)
        return "ollama"

    # Auto-detect: prefer Ollama if running locally
    if _ollama_available():
        print("Backend: Ollama (local) — detected running instance")
        return "ollama"


def extract_aspects(story_text, title, backend):
    if backend == "ollama":
        return extract_with_ollama(story_text, title)
    else:
        return 


# Quality check — print random samples from the cache

def quality_check(cache, n=10):
    print("\n" + "="*70)
    print(f"  QUALITY CHECK — {n} random samples from cache")
    print("="*70)

    sample_keys = random.sample(list(cache.keys()), min(n, len(cache)))
    for i, key in enumerate(sample_keys, 1):
        entry = cache[key]
        print(f"\n[{i}] Title: {entry.get('title', '(unknown)')}")
        print(f"    Story (first 120 chars): {key[:120]}...")
        print(f"    CoA:      {entry.get('coa','(empty)')[:150]}")
        print(f"    Outcomes: {entry.get('outcomes','(empty)')[:150]}")
        print(f"    Theme:    {entry.get('theme','(empty)')[:150]}")
        # Flag potential problems
        for asp in ["coa","outcomes","theme"]:
            val = entry.get(asp,"")
            if len(val) < 20:
                print(f"    ⚠  {asp} looks too short — may need manual fix")
            if any(w in val.lower() for w in ["romeo","juliet","hamlet"]):
                # proper names leaked into extraction
                print(f"    ⚠  {asp} may contain character names")

    print("\n" + "="*70)

    # Summary stats
    total = len(cache)
    complete = sum(1 for v in cache.values() if is_complete(v))
    print(f"  Total entries : {total}")
    print(f"  Complete      : {complete}  ({complete/total*100:.1f}%)")
    print(f"  Incomplete    : {total-complete}")

    # Average lengths
    for asp in ["coa","outcomes","theme"]:
        lengths = [len(v.get(asp,"")) for v in cache.values() if v.get(asp,"")]
        if lengths:
            print(f"  Avg {asp:8s} length: {sum(lengths)/len(lengths):.0f} chars")
    print("="*70 + "\n")


# Main extraction loop

def run_extraction(stories, cache, backend, dry_run=False):
    # Filter to stories not yet in cache (or incomplete)
    todo = [s for s in stories if not is_complete(cache.get(s["text"]))]

    if dry_run:
        todo = todo[:5]
        print(f"DRY RUN: processing {len(todo)} stories only\n")
    else:
        print(f"Stories to process: {len(todo)} "
              f"(already cached: {len(stories)-len(todo)})\n")

    if not todo:
        print("Nothing to do — all stories already cached.")
        return cache

    start_time = time.time()
    errors = 0

    batch_size = 1

    for i in tqdm(range(0, len(todo), batch_size),
                  desc=f"Extracting ({backend})",
                  total=(len(todo) + batch_size - 1) // batch_size):

        batch = todo[i : i + batch_size]

        try:
            for story in batch:
                aspects = extract_aspects(story["text"], story["title"], backend)
                cache[story["text"]] = aspects
        except Exception as e:
            print(f"\n  [ERROR] batch {i}: {e}")
            for story in batch:
                cache[story["text"]] = {
                    "title": story["title"], "coa": "", "outcomes": "", "theme": ""}
            errors += len(batch)

        # Save every 10 batches — crash-safe
        batch_num = i // batch_size
        if batch_num % 10 == 0 or i + batch_size >= len(todo):
            save_cache(cache, CACHE_PATH)

        # Progress estimate every 50 stories
        stories_done = min(i + batch_size, len(todo))
        if stories_done > 0 and stories_done % 50 < batch_size:
            elapsed   = time.time() - start_time
            rate      = stories_done / elapsed
            remaining = (len(todo) - stories_done) / rate / 60
            print(f"\n  [{stories_done}/{len(todo)}] "
                  f"rate={rate:.2f} stories/s  "
                  f"ETA={remaining:.1f} min  "
                  f"errors={errors}")

    save_cache(cache, CACHE_PATH)
    elapsed = time.time() - start_time
    print(f"\nExtraction complete in {elapsed/60:.1f} min  |  errors: {errors}")
    return cache


# ── Cache repair — clean existing cache entries ───────────────────────────────

def repair_cache(cache_path):
    """
    Run _clean_response on every entry in an existing cache file.
    Use this to fix preamble echoes in caches generated with the old prompts.

    Usage:  python extract_aspects_kaggle.py  (set quality_check=False, 
            then call repair_cache(CACHE_PATH) directly)
    """
    cache = load_cache(cache_path)
    if not cache:
        print("Cache is empty or not found.")
        return

    fixed = 0
    for text, entry in cache.items():
        for asp in ["coa", "outcomes", "theme"]:
            original = entry.get(asp, "")
            cleaned  = _clean_response(original)
            if cleaned != original:
                entry[asp] = cleaned
                fixed += 1

    save_cache(cache, cache_path)
    print(f"Cache repair complete — {fixed} fields cleaned across {len(cache)} entries.")
    return cache


class _Cfg:

    backend       = "ollama" 

    # Set True to process only 5 stories — verify output before full run
    dry_run       = False

    # Set True to inspect existing cache and exit without extracting
    quality_check = False

    # ── Paths ─────────────────────────────────────────────────────────────
    # Cache output file
    cache         = CACHE_PATH
    data_dir      = "data_files"

    hf_token      = ""   # leave empty

def parse_args():
    return _Cfg()


def main():
    args = parse_args()

    global CACHE_PATH
    CACHE_PATH = args.cache

    # Resolve data file paths
    # If data_dir is empty, DATA_FILES paths are used as-is (relative to cwd)
    if args.data_dir:
        data_files = {
            name: str(Path(args.data_dir) / fname)
            for name, fname in DATA_FILES.items()
        }
    else:
        data_files = dict(DATA_FILES)

    print("=" * 60)
    print("  Narrative Aspect Extraction")
    print("=" * 60)
    print(f"  Cache     : {CACHE_PATH}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Backend   : {args.backend}")
    print()

    # Load existing cache
    cache = load_cache(CACHE_PATH)

    # Quality check mode — just inspect existing cache and exit
    if args.quality_check:
        if not cache:
            print("Cache is empty. Run extraction first.")
        else:
            quality_check(cache)
        return

    # Collect all unique stories from all available datasets
    print("Collecting stories from datasets...")
    stories = collect_stories(data_files)
    print()

    # Detect or validate backend
    backend = detect_backend(
        forced=args.backend if args.backend != "auto" else None
    )
    print(f"Using backend: {backend}")
    if backend == "ollama":
        check_ollama_gpu()
    print()

    # Run extraction
    cache = run_extraction(stories, cache, backend, dry_run=args.dry_run)

    # Always show quality check at end of a real (non-dry) run
    if not args.dry_run and cache:
        quality_check(cache, n=5)

    print(f"\nCache saved to: {CACHE_PATH}")
    print(f"Total entries : {len(cache)}")
    complete = sum(1 for v in cache.values() if is_complete(v))
    print(f"Complete      : {complete}/{len(cache)}")


if __name__ == "__main__":
    main()