import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request

import streamlit as st


DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ASPECT_KEYS = ("coa", "outcomes", "theme")
MODE_TO_VERSION = {
    "Detailed": "v1",
    "Step-by-step": "v2",
    "Compact": "v3",
}
MODE_DESCRIPTIONS = {
    "Detailed": "Rich narrative analysis with prose explanations.",
    "Step-by-step": "Shows the extraction process in clear stages.",
    "Compact": "Returns concise structured results.",
}
VERSION_TO_MODE = {version: mode for mode, version in MODE_TO_VERSION.items()}
EXAMPLE_STORY = (
    "A young inventor discovers a hidden city beneath the desert after decoding "
    "a map left by her missing father. She enters the city with a reluctant guide, "
    "uncovers a machine that controls the region's water supply, and is forced to "
    "choose between restoring water to nearby villages or preserving the city's "
    "ancient secrecy. After sabotaging the machine's lock system, she escapes as "
    "water returns to the surface, but the city is exposed to the outside world."
)


V1_PROMPTS = {
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

V2_SYSTEM_MSG = (
    "You are a precise narrative analyst specialising in story structure. "
    "You follow instructions exactly. "
    "You always output valid JSON with no markdown fences, no extra keys, "
    "and no text outside the JSON object."
)

V2_COMBINED_PROMPT = """Analyse the story summary and extract three narrative aspects.
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

V3_SYSTEM_MSG = (
    "You are a precise narrative analyst specialising in story structure. "
    "You follow instructions exactly. "
    "You always output valid JSON with no markdown fences, no extra keys, "
    "and no text outside the JSON object. "
    "You never invent events, outcomes, or successful resolutions that are not clearly supported by the story."
)

V3_COMBINED_PROMPT = """Analyse the story summary and extract three narrative aspects.
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

V3_REPAIR_PROMPT = """You are repairing a noisy narrative-aspect extraction.
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


PREAMBLE_RE = re.compile(
    r"^(?:"
    r"here (?:is|are)(?: the| a)?[^\n:]{0,80}?[\s:.-]*\n+"
    r"|"
    r"(?:certainly|sure|of course|absolutely)[!,.]?[^\n]*\n*"
    r"|"
    r"(?:course of action|outcomes?|abstract theme|response|answer)\s*[:]\s*\n*"
    r")",
    re.IGNORECASE,
)


def fill_story(template, story_text):
    return template.replace("{story}", story_text.strip())


def call_ollama(host, model, prompt, max_tokens=500, json_mode=False, temperature=0.1):
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


def clean_response(text):
    if not text:
        return ""
    text = text.strip()
    for _ in range(4):
        cleaned = PREAMBLE_RE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
    text = re.sub(r"^\s*\n+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_json_response(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and all(key in obj for key in ASPECT_KEYS):
            return obj
    except json.JSONDecodeError:
        pass

    recovered = {}
    for key in ASPECT_KEYS:
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


def theme_parts(text):
    return [
        part.strip(" -.;,").lower()
        for part in re.split(r"[;\n,]", text)
        if part.strip(" -.;,")
    ]


def postprocess_coa(text):
    text = clean_response(text)
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


def postprocess_outcomes(text):
    text = clean_response(text)
    text = re.sub(
        r"\bresolved with partial resolution\b",
        "partially resolved",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bconflict is resolved with unresolved\b",
        "conflict remains unresolved",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bprotagonist succeeds in\b",
        "protagonist attempts to",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bultimately (solves|resolves)\b",
        "ultimately addresses",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bfully resolves\b", "largely resolves", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(personal transformation|systemic change|justice served|new hope|broader (threats|issues|relationships) remain)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def postprocess_theme(text):
    phrases = []
    seen = set()
    for part in theme_parts(clean_response(text)):
        part = re.sub(r"\b(the|a|an|story explores|themes? of)\b", "", part, flags=re.IGNORECASE)
        part = re.sub(
            r"\b(moral of the story|central theme|main theme)\b",
            "",
            part,
            flags=re.IGNORECASE,
        )
        part = re.sub(r"\s+", " ", part).strip(" -.;,")
        if part and len(part.split()) <= 5 and part not in seen:
            seen.add(part)
            phrases.append(part)
    return "; ".join(phrases[:4])


def aspect_issues(aspects):
    issues = []
    coa = aspects.get("coa", "")
    outcomes = aspects.get("outcomes", "")
    theme = aspects.get("theme", "")

    if len(coa) < 80:
        issues.append("coa is short")
    if re.search(r"(^|[\s;])\d+[.)]", coa) or "->" in coa or "\u2192" in coa:
        issues.append("coa has old formatting")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", outcomes) if s.strip()]
    if len(sentences) != 2:
        issues.append("outcomes should contain exactly 2 sentences")
    elif sentences[1].lower() not in {
        "conflict resolved.",
        "conflict unresolved.",
        "conflict partially resolved.",
    }:
        issues.append("outcomes resolution label is not plain")

    if re.search(
        r"\b(succeeds in|ultimately solves|fully resolves|justice served|systemic change|personal transformation|new hope)\b",
        outcomes,
        re.IGNORECASE,
    ):
        issues.append("outcomes may be overconfident")

    parts = theme_parts(theme)
    if len(parts) < 2 or any(len(part.split()) > 5 for part in parts):
        issues.append("theme format is loose")

    return issues


def extract_v1(story_text, host, model):
    aspects = {}
    for key, prompt_template in V1_PROMPTS.items():
        prompt = fill_story(prompt_template, story_text)
        aspects[key] = clean_response(
            call_ollama(
                host=host,
                model=model,
                prompt=prompt,
                max_tokens=220,
                json_mode=False,
                temperature=0.2,
            )
        )
    return aspects


def extract_v2(story_text, host, model):
    prompt = V2_SYSTEM_MSG + "\n\n" + fill_story(V2_COMBINED_PROMPT, story_text)
    raw = call_ollama(
        host=host,
        model=model,
        prompt=prompt,
        max_tokens=500,
        json_mode=True,
        temperature=0.1,
    )
    parsed = parse_json_response(raw)
    return {key: clean_response(parsed.get(key, "")) for key in ASPECT_KEYS}


def extract_v3(story_text, host, model):
    prompt = V3_SYSTEM_MSG + "\n\n" + fill_story(V3_COMBINED_PROMPT, story_text)
    raw = call_ollama(
        host=host,
        model=model,
        prompt=prompt,
        max_tokens=500,
        json_mode=True,
        temperature=0.0,
    )
    parsed = parse_json_response(raw)
    aspects = {
        "coa": postprocess_coa(parsed.get("coa", "")),
        "outcomes": postprocess_outcomes(parsed.get("outcomes", "")),
        "theme": postprocess_theme(parsed.get("theme", "")),
    }

    if aspect_issues(aspects):
        repair_prompt = fill_story(V3_REPAIR_PROMPT, story_text).replace(
            "{bad_json}", json.dumps(aspects, ensure_ascii=False)
        )
        raw_repair = call_ollama(
            host=host,
            model=model,
            prompt=V3_SYSTEM_MSG + "\n\n" + repair_prompt,
            max_tokens=350,
            json_mode=True,
            temperature=0.0,
        )
        repaired = parse_json_response(raw_repair)
        aspects = {
            "coa": postprocess_coa(repaired.get("coa", aspects["coa"])),
            "outcomes": postprocess_outcomes(repaired.get("outcomes", aspects["outcomes"])),
            "theme": postprocess_theme(repaired.get("theme", aspects["theme"])),
        }

    return aspects


EXTRACTORS = {
    "v1": extract_v1,
    "v2": extract_v2,
    "v3": extract_v3,
}


def run_extraction(version, story_text, host, model):
    return EXTRACTORS[version](story_text=story_text, host=host, model=model)


def get_executor():
    if "executor" not in st.session_state:
        st.session_state["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return st.session_state["executor"]


def active_job_running():
    job = st.session_state.get("active_job")
    return bool(job and not job["future"].done())


def start_extraction_job(version, story_text, host, model):
    if active_job_running():
        running_version = st.session_state["active_job"]["version"]
        st.session_state["last_error"] = f"Already running {running_version}."
        return

    st.session_state["last_result"] = None
    future = get_executor().submit(run_extraction, version, story_text, host, model)
    st.session_state["active_job"] = {
        "future": future,
        "version": version,
        "model": model,
        "started_at": time.time(),
    }
    st.session_state["last_error"] = None


def update_extraction_job():
    job = st.session_state.get("active_job")
    if not job:
        return False

    future = job["future"]
    if not future.done():
        return True

    try:
        result = future.result()
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
    else:
        st.session_state["last_error"] = None
        st.session_state["last_result"] = {
            "version": job["version"],
            "model": job["model"],
            "aspects": result,
        }

    st.session_state["active_job"] = None
    return False


def open_prompts(version):
    st.session_state["prompt_version"] = version
    st.session_state["prompt_open"] = True


def close_prompts():
    st.session_state["prompt_open"] = False


def use_example_story():
    st.session_state["story_input"] = EXAMPLE_STORY


def clear_story_input():
    st.session_state["story_input"] = ""
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None


def ollama_connected(host):
    try:
        request = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=1):
            return True
    except Exception:
        return False


def render_extraction_state():
    was_running = active_job_running()
    job_running = update_extraction_job()
    if was_running and not job_running and hasattr(st, "fragment"):
        st.rerun()

    with st.container(border=True):
        st.markdown("### Results")

        if job_running:
            job = st.session_state["active_job"]
            elapsed = int(time.time() - job["started_at"])
            mode = VERSION_TO_MODE.get(job["version"], job["version"])
            st.info(
                f"Extracting narrative aspects with {mode} mode... {elapsed}s\n\n"
                "This may take a few seconds depending on the model."
            )

        last_error = st.session_state.get("last_error")
        if last_error:
            configured_host = st.session_state.get("current_host", DEFAULT_HOST)
            st.error(
                "Could not complete the extraction.\n\n"
                f"{last_error}\n\n"
                f"Check that Ollama is running at {configured_host}."
            )

        last_result = st.session_state.get("last_result")
        if last_result:
            mode = VERSION_TO_MODE.get(last_result["version"], last_result["version"])
            st.success(f"Extraction complete. Mode: {mode}.")
            aspects = last_result["aspects"]

            out1, out2, out3 = st.columns(3)
            with out1:
                st.markdown("**Course of Action**")
                st.write(aspects.get("coa", ""))
            with out2:
                st.markdown("**Outcomes**")
                st.write(aspects.get("outcomes", ""))
            with out3:
                st.markdown("**Abstract Theme**")
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


def prompts_for_version(version, story_text):
    story_for_preview = story_text.strip() or "{story}"

    if version == "v1":
        return [
            ("CoA", fill_story(V1_PROMPTS["coa"], story_for_preview)),
            ("Outcomes", fill_story(V1_PROMPTS["outcomes"], story_for_preview)),
            ("Theme", fill_story(V1_PROMPTS["theme"], story_for_preview)),
        ]

    if version == "v2":
        return [
            ("System", V2_SYSTEM_MSG),
            ("Combined", fill_story(V2_COMBINED_PROMPT, story_for_preview)),
        ]

    if version == "v3":
        repair_preview = fill_story(V3_REPAIR_PROMPT, story_for_preview).replace(
            "{bad_json}", "{bad_extraction_json}"
        )
        return [
            ("System", V3_SYSTEM_MSG),
            ("Combined", fill_story(V3_COMBINED_PROMPT, story_for_preview)),
            ("Repair", repair_preview),
        ]

    raise ValueError(f"Unknown prompt version: {version}")


def render_prompt_preview(version, story_text):
    prompts = prompts_for_version(version, story_text)
    header_col, close_col = st.columns([0.94, 0.06])
    mode = VERSION_TO_MODE.get(version, version)
    header_col.subheader(f"Prompt template: {mode}")
    close_col.button(
        "X",
        key=f"close_prompts_{version}",
        help="Close prompts",
        on_click=close_prompts,
        use_container_width=True,
    )

    tabs = st.tabs([name for name, _ in prompts])
    for tab, (_, prompt_text) in zip(tabs, prompts):
        with tab:
            st.code(prompt_text, language="text")


st.set_page_config(page_title="Narrative Aspect Extractor", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 {
        font-size: 2.4rem !important;
        line-height: 1.1 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    h3 {
        font-size: 1.15rem !important;
        letter-spacing: 0 !important;
    }
    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.28);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--secondary-background-color);
        border-color: rgba(128, 128, 128, 0.28) !important;
        border-radius: 12px;
    }
    .secondary-copy {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.98rem;
        margin-bottom: 1.35rem;
    }
    .mode-help {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.9rem;
        margin-top: -0.35rem;
        margin-bottom: 0.85rem;
    }
    .settings-label {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("prompt_open", False)
st.session_state.setdefault("prompt_version", None)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_error", None)
st.session_state.setdefault("active_job", None)
st.session_state.setdefault("story_input", "")
st.session_state.setdefault("selected_mode", "Detailed")

update_extraction_job()

with st.sidebar:
    st.header("Settings")
    st.subheader("Model settings")
    st.text_input("Provider", value="Ollama", disabled=True)
    host = st.text_input("Host", DEFAULT_HOST)
    model = st.text_input("Model", DEFAULT_MODEL)
    st.session_state["current_host"] = host.strip() or DEFAULT_HOST

    connected = ollama_connected(st.session_state["current_host"])
    status_color = "#16a34a" if connected else "#dc2626"
    status_text = "Connected" if connected else "Not connected"
    st.markdown(
        "<div class='settings-label'>Connection status</div>"
        f"<div style='font-weight:600;color:{status_color};'>{status_text}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Setup commands"):
        st.code(f"ollama pull {model}", language="bash")
        st.code("ollama serve", language="bash")

title_col, settings_col = st.columns([0.78, 0.22])
with title_col:
    st.title("Narrative Aspect Extractor")
    st.markdown(
        "<div class='secondary-copy'>Extract course of action, outcomes, and abstract theme from story summaries.</div>",
        unsafe_allow_html=True,
    )
with settings_col:
    st.caption("Settings are available in the sidebar.")

input_col, config_col = st.columns([0.64, 0.36], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown("### Story summary")
        st.caption("Paste a synopsis, chapter outline, or scene.")

        example_col, clear_col = st.columns([0.5, 0.5])
        example_col.button(
            "Use example story",
            on_click=use_example_story,
            use_container_width=True,
        )
        clear_col.button(
            "Clear input",
            on_click=clear_story_input,
            use_container_width=True,
        )

        story = st.text_area(
            "Input story",
            key="story_input",
            height=300,
            label_visibility="collapsed",
            placeholder=(
                "Paste a story summary, synopsis, or chapter outline here.\n"
                "Example: A young inventor discovers a hidden city beneath the desert..."
            ),
        )
        st.caption(f"{len(story)} characters | Recommended length: 200-2,000 words")

with config_col:
    with st.container(border=True):
        st.markdown("### Extraction settings")
        selected_mode = st.radio(
            "Extraction style",
            list(MODE_TO_VERSION.keys()),
            key="selected_mode",
            horizontal=True,
        )
        st.markdown(
            f"<div class='mode-help'>{MODE_DESCRIPTIONS[selected_mode]}</div>",
            unsafe_allow_html=True,
        )

        requested_version = None
        if st.button(
            "Extract narrative aspects",
            type="primary",
            use_container_width=True,
            disabled=active_job_running(),
        ):
            requested_version = MODE_TO_VERSION[selected_mode]

        st.button(
            "View prompt template",
            use_container_width=True,
            on_click=open_prompts,
            args=(MODE_TO_VERSION[selected_mode],),
        )

if requested_version:
    if not story.strip():
        st.warning("Paste a story summary first.")
    else:
        start_extraction_job(
            requested_version,
            story.strip(),
            st.session_state["current_host"],
            model.strip(),
        )

if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def extraction_state_fragment():
        render_extraction_state()

    extraction_state_fragment()
else:
    render_extraction_state()

prompt_version = st.session_state.get("prompt_version")
if st.session_state.get("prompt_open") and prompt_version:
    st.divider()
    render_prompt_preview(prompt_version, story)
