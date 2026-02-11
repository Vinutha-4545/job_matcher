# scorer.py
from typing import Dict, List, Optional
import numpy as np

from .config import (
    SKILL_WEIGHT,
    EXPERIENCE_WEIGHT,
    LOCATION_WEIGHT,
)

# Lazy-loaded embedding model to avoid heavy import at module import time
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for scoring") from exc
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------- Skill Score ----------
def skill_match_score(
    candidate_skills: List[str],
    jd_skills: List[Dict],
) -> float:
    if not candidate_skills or not jd_skills:
        return 0.0
    # Try embedding-based semantic similarity if sentence-transformers is available.
    try:
        model = _get_embedding_model()
        cand_embeddings = model.encode(candidate_skills, convert_to_tensor=True)

        total_score = 0.0
        total_weight = 0.0

        for item in jd_skills:
            jd_skill = item["skill"]
            weight = float(item["weight"])

            jd_embedding = model.encode(jd_skill, convert_to_tensor=True)
            try:
                from sentence_transformers import util
                similarity = util.cos_sim(jd_embedding, cand_embeddings).max().item()
            except Exception:
                # fallback to numpy if util unavailable
                jd_np = np.array(jd_embedding)
                cand_np = np.array(cand_embeddings)
                denom = (np.linalg.norm(jd_np) * np.linalg.norm(cand_np))
                if denom == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.max(np.dot(jd_np, cand_np.T) / denom))

            total_score += similarity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(total_score / total_weight, 4)
    except Exception:
        # Fallback: simple overlap / heuristic when heavy deps unavailable.
        import difflib

        total_score = 0.0
        total_weight = 0.0

        cand_skills_norm = [s.lower() for s in candidate_skills]

        for item in jd_skills:
            jd_skill = item["skill"].lower()
            weight = float(item.get("weight", 1.0))

            # best similarity between jd_skill and any candidate skill
            best_sim = 0.0
            for cs in cand_skills_norm:
                if jd_skill in cs or cs in jd_skill:
                    sim = 1.0
                else:
                    sim = difflib.SequenceMatcher(None, jd_skill, cs).ratio()
                if sim > best_sim:
                    best_sim = sim

            total_score += best_sim * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(total_score / total_weight, 4)


# ---------- Experience Score ----------
def experience_score(
    candidate_years: float,
    min_required_years: int | None,
) -> float:
    if not min_required_years or min_required_years <= 0:
        return 1.0

    return round(min(candidate_years / min_required_years, 1.0), 4)


# ---------- Location Score ----------
def location_score(
    candidate_location: str,
    preferred_location: str | None,
) -> float:
    if not preferred_location:
        return 1.0

    if not candidate_location:
        return 0.0

    return 1.0 if preferred_location.lower() in candidate_location.lower() else 0.0


# ---------- Final Score ----------
def compute_final_score(
    candidate: Dict,
    job: Dict,
) -> Dict:
    skill_score = skill_match_score(
        candidate.get("skills", []),
        job.get("required_skills", []),
    )

    exp_score = experience_score(
        candidate.get("total_experience_years", 0.0),
        job.get("min_experience_years"),
    )

    loc_score = location_score(
        candidate.get("location", ""),
        job.get("preferred_location"),
    )

    final_score = (
        skill_score * SKILL_WEIGHT
        + exp_score * EXPERIENCE_WEIGHT
        + loc_score * LOCATION_WEIGHT
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "final_score": round(final_score, 4),
        "skill_score": skill_score,
        "experience_score": exp_score,
        "location_score": loc_score,
    }
