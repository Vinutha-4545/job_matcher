import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import os

from openai import AzureOpenAI

from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    LLM_RATE_LIMIT_SECONDS,
    RESUME_CACHE_FILE,
    JD_CACHE_FILE,
)

# ---------- Logging ----------
logger = logging.getLogger(__name__)

# ---------- Azure Client ----------
_azure_client: Optional[AzureOpenAI] = None


def get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is None:
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            raise RuntimeError("Azure OpenAI configuration is missing")
        _azure_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _azure_client


# Allow simulated LLM mode (for local/dev runs)
SIMULATE_LLM = os.getenv("SIMULATE_LLM", "0").lower() in ("1", "true", "yes")

# ---------- Cache ----------
def _load_cache(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: Path, cache: Dict) -> None:
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

    # -------- Azure OpenAI --------
    if not SIMULATE_LLM:
        try:
            client = get_azure_client()

            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )

            response_text = response.choices[0].message.content
            cache[cache_key] = response_text
            _save_cache(cache_file, cache)

            time.sleep(LLM_RATE_LIMIT_SECONDS)
            return response_text

        except Exception as exc:
            logger.warning(
                "Azure LLM failed, falling back to simulated LLM: %s", exc
            )

    # -------- Simulated LLM (fallback/dev) --------
    try:
        response_text = _simulate_llm_response(prompt)
        cache[cache_key] = response_text
        _save_cache(cache_file, cache)
        time.sleep(LLM_RATE_LIMIT_SECONDS)
        return response_text
    except Exception as exc:
        logger.error("Simulated LLM failed", exc_info=exc)
        return None


# ---------- Simulated LLM ----------
def _simulate_llm_response(prompt: str) -> str:
    text = prompt

    # Job Description simulation
    if "Job Description:" in text or "Extract structured job" in text:
        words = [w.strip(".,()\"'`)[]") for w in text.split() if len(w) > 4]
        unique = list(dict.fromkeys(words))
        skills = unique[:5]

        if skills:
            weight = round(1.0 / len(skills), 3)
            required_skills = [{"skill": s, "weight": weight} for s in skills]
        else:
            required_skills = []

        jd = {
            "title": "Generated Job",
            "required_skills": required_skills,
            "preferred_location": None,
            "min_experience_years": 3,
        }
        return json.dumps(jd)

    # Resume simulation
    words = [w.strip(".,()\"'`)[]") for w in text.split() if len(w) > 4]
    unique = list(dict.fromkeys(words))
    skills = unique[:10]

    experience = [
        {
            "role": "Software Engineer",
            "company": "ACME",
            "duration_years": 3,
            "skills": skills[:3],
        }
    ]

    candidate = {
        "candidate_id": f"sim-{abs(hash(text)) % (10 ** 8)}",
        "name": "Simulated Candidate",
        "email": "sim@example.com",
        "phone": "",
        "location": "",
        "skills": skills,
        "experience": experience,
        "education": [],
    }

    return json.dumps(candidate)


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
