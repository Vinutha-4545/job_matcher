# models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from uuid import uuid4


@dataclass
class Experience:
    role: str
    company: str
    duration_years: float
    skills: List[str]


@dataclass
class Candidate:
    candidate_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    skills: List[str] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    total_experience_years: float = 0.0
    education: List[Dict] = field(default_factory=list)
    resume_path: str = ""


@dataclass
class JobSkill:
    skill: str
    weight: float


@dataclass
class Job:
    job_id: str
    title: str
    description: str
    required_skills: List[JobSkill]
    preferred_location: Optional[str] = None
    min_experience_years: Optional[int] = None
