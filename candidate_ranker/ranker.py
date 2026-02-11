# ranker.py
from typing import List, Dict
def rank_candidates(job_id: str, top_k: int = 10) -> None:
    # Import heavy modules lazily
    try:
        from tabulate import tabulate
    except Exception:
        tabulate = None

    from .db import MongoDBManager
    from .scorer import compute_final_score

    db = MongoDBManager()

    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job ID not found: {job_id}")

    candidates = db.get_all_candidates()
    if not candidates:
        print("No candidates found.")
        return

    scored: List[Dict] = []

    for candidate in candidates:
        result = compute_final_score(candidate, job)
        scored.append({**candidate, **result})

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    top_candidates = scored[:top_k]

    db.insert_ranking(
        {
            "job_id": job_id,
            "top_candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "score": c["final_score"],
                    "skill_score": c["skill_score"],
                    "experience_score": c["experience_score"],
                    "location_score": c["location_score"],
                }
                for c in top_candidates
            ],
        }
    )

    table = [
        [
            idx + 1,
            c.get("name", ""),
            c.get("location", ""),
            c["final_score"],
            c["skill_score"],
            c["experience_score"],
            c["location_score"],
        ]
        for idx, c in enumerate(top_candidates)
    ]

    if tabulate:
        print(
            tabulate(
                table,
                headers=[
                    "Rank",
                    "Name",
                    "Location",
                    "Final",
                    "Skill",
                    "Exp",
                    "Loc",
                ],
                tablefmt="github",
            )
        )
    else:
        # Fallback plain text table
        print("Rank | Name | Location | Final | Skill | Exp | Loc")
        for row in table:
            print(" | ".join(str(x) for x in row))
