import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import os
import random

try:
    from gpt4all import GPT4All
except Exception:  # pragma: no cover - optional dependency
    GPT4All = None

from .config import (
    LLM_MODEL_PATH,
    LLM_RATE_LIMIT_SECONDS,
    RESUME_CACHE_FILE,
    JD_CACHE_FILE,
)

# ---------- Logging ----------
logger = logging.getLogger(__name__)

# ---------- Load LLM ----------
_llm_instance: Optional[Any] = None


def get_llm() -> Any:
    global _llm_instance
    if _llm_instance is None:
        if GPT4All is None:
            raise RuntimeError("gpt4all is not available; install gpt4all or replace LLM backend")
        _llm_instance = GPT4All(model_name=LLM_MODEL_PATH)
    return _llm_instance


# Allow a simulated LLM mode via env var for easy CLI runs without gpt4all
SIMULATE_LLM = os.getenv("SIMULATE_LLM", "0") in ("1", "true", "True")


# ---------- Cache ----------
def _load_cache(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: Path, cache: Dict) -> None:
    # ensure parent dir exists
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ---------- LLM Call ----------
def call_llm_with_cache(
    prompt: str,
    cache_key: str,
    cache_file: Path,
    max_tokens: int = 512,
) -> Optional[str]:
    cache = _load_cache(cache_file)

    if cache_key in cache:
        return cache[cache_key]

    # First, try using a real local LLM if available
    if not SIMULATE_LLM and GPT4All is not None:
        try:
            llm = get_llm()
            response = llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.2,
            )

            # Ensure we store a JSON-serializable representation
            response_text = str(response)
            cache[cache_key] = response_text
            _save_cache(cache_file, cache)

            time.sleep(LLM_RATE_LIMIT_SECONDS)
            return response_text
        except Exception as exc:
            logger.warning("Real LLM failed, falling back to simulated LLM: %s", exc)

    # Simulated LLM path (for development/testing or fallback)
    try:
        response_text = _simulate_llm_response(prompt)
        cache[cache_key] = response_text
        _save_cache(cache_file, cache)
        # simulate delay
        time.sleep(LLM_RATE_LIMIT_SECONDS)
        return response_text
    except Exception as exc:
        logger.error("Simulated LLM failed", exc_info=exc)
        return None


def _simulate_llm_response(prompt: str) -> str:
    """Produce a simple deterministic simulated JSON response based on prompt content.

    This is intentionally lightweight: it extracts candidate-like fields from the
    prompt text heuristically so the rest of the pipeline can run without a real LLM.
    """
    text = prompt
    # If prompt contains 'Job Description' assume it's a JD
    if "Job Description:" in text or "Extract structured job" in text:
        # pick a few frequent words as skills
        words = [w.strip(".,()\"'`)[]") for w in text.split() if len(w) > 4]
        unique = list(dict.fromkeys(words))
        skills = unique[:5]
        # assign equal weights that sum to 1
        if skills:
            w = round(1.0 / len(skills), 3)
            req = [{"skill": s, "weight": w} for s in skills]
        else:
            req = []
        jd = {
            "title": "Generated Job",
            "required_skills": req,
            "preferred_location": None,
            "min_experience_years": 3,
        }
        return json.dumps(jd)

    # Otherwise treat as resume
    words = [w.strip(".,()\"'`)[]") for w in text.split() if len(w) > 4]
    unique = list(dict.fromkeys(words))
    skills = unique[:10]

    experience = [
        {"role": "Software Engineer", "company": "ACME", "duration_years": 3, "skills": skills[:3]}
    ]

    cand = {
        "candidate_id": f"sim-{abs(hash(text)) % (10 ** 8)}",
        "name": "Simulated Candidate",
        "email": "sim@example.com",
        "phone": "",
        "location": "",
        "skills": skills,
        "experience": experience,
        "education": [],
    }
    return json.dumps(cand)


# ---------- Convenience Wrappers ----------
def llm_resume_extract(prompt: str, resume_path: str) -> Optional[str]:
    return call_llm_with_cache(
        prompt=prompt,
        cache_key=resume_path,
        cache_file=RESUME_CACHE_FILE,
    )


def llm_jd_extract(prompt: str, job_id: str) -> Optional[str]:
    return call_llm_with_cache(
        prompt=prompt,
        cache_key=job_id,
        cache_file=JD_CACHE_FILE,
    )
