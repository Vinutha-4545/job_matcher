# job_processor.py
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

from .db import MongoDBManager
from .utils import llm_jd_extract 
from .models import Job, JobSkill

logger = logging.getLogger(__name__)


# ---------- Prompt ----------
def build_jd_prompt(jd_text: str) -> str:
    return f"""
Extract structured job information from the description below.
Return ONLY valid JSON in this exact format:

{{
  "title": "",
  "required_skills": [
    {{ "skill": "", "weight": 0.0 }}
  ],
  "preferred_location": "",
  "min_experience_years": 0
}}

Rules:
- Skill weights must sum to 1.0
- Use only core technical skills
- If something is missing, use empty string or 0

Job Description:
{jd_text[:4000]}
"""


# ---------- Normalization ----------
def normalize_job(
    job_id: str,
    jd_text: str,
    llm_output: str
) -> Optional[Dict]:
    try:
        data = json.loads(llm_output)

        skills: List[JobSkill] = []
        for item in data.get("required_skills", []):
            skills.append(
                JobSkill(
                    skill=item.get("skill", "").strip(),
                    weight=float(item.get("weight", 0)),
                )
            )

        job = Job(
            job_id=job_id,
            title=data.get("title", ""),
            description=jd_text,
            required_skills=skills,
            preferred_location=data.get("preferred_location") or None,
            min_experience_years=int(
                data.get("min_experience_years", 0)
            ) or None,
        )

        return {
            "job_id": job.job_id,
            "title": job.title,
            "description": job.description,
            "required_skills": [
                {"skill": s.skill, "weight": s.weight}
                for s in job.required_skills
            ],
            "preferred_location": job.preferred_location,
            "min_experience_years": job.min_experience_years,
        }

    except Exception as exc:
        logger.error("Failed to normalize JD", exc_info=exc)
        return None


# ---------- Public API ----------
def add_job(job_id: str, jd_file: Path) -> None:
    db = MongoDBManager()

    if not jd_file.exists():
        raise FileNotFoundError(f"JD file not found: {jd_file}")

    # Support plain text, PDF and DOCX job descriptions
    if jd_file.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}:
        # lazy import to avoid heavy deps at module import time
        from .extractor import extract_raw_text

        jd_text = extract_raw_text(jd_file) or ""
    else:
        jd_text = jd_file.read_text(encoding="utf-8")

    prompt = build_jd_prompt(jd_text)
    llm_response = llm_jd_extract(prompt, job_id)

    if not llm_response:
        raise RuntimeError("LLM failed to extract JD")

    job_data = normalize_job(job_id, jd_text, llm_response)
    if not job_data:
        raise RuntimeError("Invalid JD structure")
    try:
        db.insert_job(job_data)
        logger.info("Inserted job %s into DB", job_id)
    except Exception:
        # If DB unavailable, print job_data so user can see result
        logger.exception("Failed to insert job into DB; printing job data")
        print("Job data (DB insert failed):")
        print(json.dumps(job_data, indent=2))
