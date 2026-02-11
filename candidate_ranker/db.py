# db.py
from typing import List, Dict, Optional
from datetime import datetime
from pymongo import MongoClient, errors
import logging

from .config import (
    MONGO_URI,
    DB_NAME,
    RESUME_RAW_COLLECTION,
    CANDIDATE_COLLECTION,
    JOB_COLLECTION,
    RANKING_COLLECTION,
    LOG_COLLECTION,
)


class MongoDBManager:
    def __init__(self) -> None:
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]

            self.resumes_raw = self.db[RESUME_RAW_COLLECTION]
            self.candidates = self.db[CANDIDATE_COLLECTION]
            self.jobs = self.db[JOB_COLLECTION]
            self.rankings = self.db[RANKING_COLLECTION]
            self.logs = self.db[LOG_COLLECTION]

        except errors.PyMongoError as exc:
            raise RuntimeError(f"MongoDB connection failed: {exc}") from exc

    # ---------- Resume ----------
    def insert_raw_resume(self, data: Dict) -> None:
        data["ingested_at"] = datetime.utcnow()
        self.resumes_raw.insert_one(data)

    # ---------- Candidate ----------
    def insert_candidate(self, candidate: Dict) -> None:
        candidate["created_at"] = datetime.utcnow()
        # Use upsert to avoid duplicate candidate documents
        self.candidates.replace_one({"candidate_id": candidate.get("candidate_id")}, candidate, upsert=True)

    def get_all_candidates(self) -> List[Dict]:
        return list(self.candidates.find({}, {"_id": 0}))

    # ---------- Job ----------
    def insert_job(self, job: Dict) -> None:
        job["created_at"] = datetime.utcnow()
        self.jobs.replace_one(
            {"job_id": job["job_id"]},
            job,
            upsert=True
        )

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.find_one({"job_id": job_id}, {"_id": 0})

    # ---------- Ranking ----------
    def insert_ranking(self, ranking: Dict) -> None:
        ranking["generated_at"] = datetime.utcnow()
        self.rankings.insert_one(ranking)

    # ---------- Logs ----------
    def log(
        self,
        level: str,
        module: str,
        message: str,
        meta: Optional[Dict] = None
    ) -> None:
        log_entry = {
            "level": level,
            "module": module,
            "message": message,
            "meta": meta or {},
            "timestamp": datetime.utcnow(),
        }
        self.logs.insert_one(log_entry)
