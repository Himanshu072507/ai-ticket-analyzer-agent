"""Analyze a support-ticket DataFrame with Gemini, Groq, or local Ollama and return structured insights."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash-lite"]
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODELS = ["llama-3.1-8b-instant"]
MAX_SAMPLES = 30
SNIPPET_CHARS = 200
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a senior support operations analyst helping a support manager prioritize fixes. Be radically concise. Write like Slack messages, not paragraphs.

Hard rules per field:
- headline: ONE sentence, max 25 words. Lead with the biggest finding. Include numbers.
- summary (cluster): max 12 words. Name the issue, not the symptom.
- root_cause: ONE sentence, max 20 words. Plain language.
- suggested_owner: 1-4 words. Team name only (e.g. "Identity team", "Platform / Notifications").
- suggested_fix: max 15 words. MUST start with a verb. Specific and actionable (e.g. "Raise OTP TTL to 30 min", not "Investigate token expiry").

Other rules:
- Quote concrete numbers from the data ("23% of tickets", "412 of 1800").
- Every recommendation must tie to a volume number in the data.
- Prioritize by impact = volume x severity, not severity alone. A noisy low-severity issue often beats a rare critical one.
- Use the provided samples to name specific pain points, not abstract themes.
- No filler ("It is recommended that...", "We should consider..."). State the action.

Return your analysis as JSON that conforms exactly to the provided schema."""


_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}

OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": _STR,
        "ranked_categories": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"name": _STR, "count": _INT, "percent": _NUM},
                "required": ["name", "count", "percent"],
            },
        },
    },
    "required": ["headline", "ranked_categories"],
}

CATEGORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": _STR,
                    "summary": _STR,
                    "keywords": {"type": "array", "items": _STR, "minItems": 1, "maxItems": 8},
                    "root_cause": _STR,
                    "suggested_owner": _STR,
                    "suggested_fix": _STR,
                },
                "required": ["name", "summary", "keywords", "root_cause",
                             "suggested_owner", "suggested_fix"],
            },
        }
    },
    "required": ["clusters"],
}


def _build_stats(df: pd.DataFrame, m: dict) -> dict[str, Any]:
    cat, desc = m["category"], m["description"]
    stats: dict[str, Any] = {"total_tickets": int(len(df))}
    stats["top_categories"] = df[cat].value_counts().head(15).to_dict()
    if m.get("department"):
        stats["top_departments"] = df[m["department"]].value_counts().head(10).to_dict()
    if m.get("status"):
        stats["status_breakdown"] = df[m["status"]].value_counts().to_dict()
    if m.get("priority"):
        stats["priority_breakdown"] = df[m["priority"]].value_counts().to_dict()
    if m.get("date"):
        dates = pd.to_datetime(df[m["date"]], errors="coerce").dropna()
        if not dates.empty:
            last_30 = dates[dates >= dates.max() - pd.Timedelta(days=30)]
            stats["daily_volume_last_30d"] = {
                d.strftime("%Y-%m-%d"): int(c)
                for d, c in last_30.dt.date.value_counts().sort_index().items()
            }
    if m.get("department"):
        top_cats = df[cat].value_counts().head(10).index
        top_deps = df[m["department"]].value_counts().head(10).index
        sub = df[df[cat].isin(top_cats) & df[m["department"]].isin(top_deps)]
        crosstab = pd.crosstab(sub[cat], sub[m["department"]])
        stats["category_by_department"] = {
            c: {d: int(v) for d, v in row.items() if v > 0}
            for c, row in crosstab.iterrows()
        }
    stats["description_samples"] = _sample_descriptions(df, cat, desc)
    return stats


def _sample_descriptions(df: pd.DataFrame, cat: str, desc: str) -> dict[str, list[str]]:
    counts = df[cat].value_counts()
    total = int(counts.sum())
    samples: dict[str, list[str]] = {}
    for category, count in counts.items():
        quota = max(1, math.ceil(MAX_SAMPLES * (count / total)))
        picks = df[df[cat] == category][desc].dropna().astype(str).head(quota).tolist()
        samples[str(category)] = [s[:SNIPPET_CHARS] for s in picks]
        if sum(len(v) for v in samples.values()) >= MAX_SAMPLES:
            break
    return samples


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def _call_gemini(client: genai.Client, model: str, payload: str, schema: dict) -> str:
    response = client.models.generate_content(
        model=model,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=4096,
        ),
    )
    if not response.text:
        raise RuntimeError("Empty response.")
    return response.text


def _call_ollama(host: str, model: str, payload: str, schema: dict) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned empty content.")
    return content


def _call_groq(client: Any, model: str, payload: str, schema: dict) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=4096,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned empty content.")
    return content


def _analyze_groq(payload: str, api_key: str, model: str, schema: dict) -> dict:
    if not api_key:
        raise RuntimeError("Groq API key is required.")
    try:
        from groq import Groq
    except ImportError as e:
        raise RuntimeError("groq package is not installed. Run `pip install groq`.") from e

    # Inject schema hint into payload so the model conforms (no native schema arg in Groq SDK).
    schema_hint = (
        "\n\nReturn JSON conforming exactly to this schema (no extra fields):\n"
        + json.dumps(schema)
    )
    payload_with_schema = payload + schema_hint

    client = Groq(api_key=api_key)
    last_err: Exception | None = None
    for candidate in [model, *GROQ_FALLBACK_MODELS]:
        for attempt in range(MAX_RETRIES):
            try:
                text = _call_groq(client, candidate, payload_with_schema, schema)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Groq response was not valid JSON: {e}") from e
            except RuntimeError:
                raise
            except Exception as e:
                last_err = e
                if _is_transient(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
    raise RuntimeError(
        f"Groq API call failed after retrying {model} and fallbacks {GROQ_FALLBACK_MODELS}. "
        f"Last error: {last_err}"
    )


def _analyze_gemini(payload: str, api_key: str, schema: dict) -> dict:
    if not api_key:
        raise RuntimeError("Gemini API key is required.")
    client = genai.Client(api_key=api_key)
    last_err: Exception | None = None
    for model in [MODEL, *FALLBACK_MODELS]:
        for attempt in range(MAX_RETRIES):
            try:
                return json.loads(_call_gemini(client, model, payload, schema))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Gemini response was not valid JSON: {e}") from e
            except Exception as e:
                last_err = e
                if _is_transient(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
    raise RuntimeError(
        f"Gemini API call failed after retrying {MODEL} and fallbacks {FALLBACK_MODELS}. "
        f"Last error: {last_err}"
    )


def _analyze_ollama(payload: str, model: str, host: str, schema: dict) -> dict:
    try:
        text = _call_ollama(host, model, payload, schema)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is it running? (`ollama serve`). Details: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama response was not valid JSON: {e}") from e


def _build_category_payload(df: pd.DataFrame, mapping: dict, category_name: str) -> str:
    """Slice df to one category and serialize description samples for the cluster call."""
    cat_col, desc_col = mapping["category"], mapping["description"]
    sub = df[df[cat_col].astype(str) == str(category_name)]
    if sub.empty:
        raise RuntimeError(f"No tickets found for category {category_name!r}.")
    samples = (
        sub[desc_col].dropna().astype(str).head(MAX_SAMPLES).map(lambda s: s[:SNIPPET_CHARS]).tolist()
    )
    body = {
        "category": str(category_name),
        "ticket_count": int(len(sub)),
        "description_samples": samples,
    }
    return f"Single-category cluster payload:\n\n{json.dumps(body, indent=2, default=str)}"


def analyze_overview(
    df: pd.DataFrame,
    mapping: dict,
    provider: str = "gemini",
    api_key: str = "",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> dict:
    """Generate the headline narrative and ranked category list."""
    if df.empty or not mapping.get("category") or not mapping.get("description"):
        raise RuntimeError("DataFrame must be non-empty with 'category' and 'description' mapped.")
    payload = f"Ticket analytics payload:\n\n{json.dumps(_build_stats(df, mapping), indent=2, default=str)}"
    if provider == "ollama":
        return _analyze_ollama(payload, ollama_model, ollama_host, OVERVIEW_SCHEMA)
    if provider == "gemini":
        return _analyze_gemini(payload, api_key, OVERVIEW_SCHEMA)
    if provider == "groq":
        return _analyze_groq(payload, api_key, groq_model, OVERVIEW_SCHEMA)
    raise RuntimeError(f"Unknown provider: {provider!r}. Use 'gemini', 'groq', or 'ollama'.")


def analyze_category(
    df: pd.DataFrame,
    mapping: dict,
    category_name: str,
    provider: str = "gemini",
    api_key: str = "",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> dict:
    """Cluster a single category's tickets into 1-5 named sub-issues."""
    if df.empty or not mapping.get("category") or not mapping.get("description"):
        raise RuntimeError("DataFrame must be non-empty with 'category' and 'description' mapped.")
    payload = _build_category_payload(df, mapping, category_name)
    if provider == "ollama":
        return _analyze_ollama(payload, ollama_model, ollama_host, CATEGORY_SCHEMA)
    if provider == "gemini":
        return _analyze_gemini(payload, api_key, CATEGORY_SCHEMA)
    if provider == "groq":
        return _analyze_groq(payload, api_key, groq_model, CATEGORY_SCHEMA)
    raise RuntimeError(f"Unknown provider: {provider!r}. Use 'gemini', 'groq', or 'ollama'.")


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    """Return the names of locally-installed Ollama models, or [] if unreachable."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        return []
