# extractor.py
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import re
from datetime import datetime, date

from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from tqdm import tqdm

from .db import MongoDBManager
from .utils import llm_resume_extract
from .models import Candidate, Experience

logger = logging.getLogger(__name__)


# ---------- Text Extraction ----------
def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_image(path: Path) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image)


def extract_raw_text(path: Path) -> Optional[str]:
    try:
        if path.suffix.lower() == ".pdf":
            return extract_text_from_pdf(path)
        if path.suffix.lower() == ".docx":
            return extract_text_from_docx(path)
        if path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return extract_text_from_image(path)
    except Exception as exc:
        logger.error(f"Text extraction failed for {path.name}", exc_info=exc)
    return None


# ---------- LLM Prompt ----------
def build_resume_prompt(text: str) -> str:
    return f"""
Extract structured candidate information from the resume below.
Return ONLY valid JSON in this exact format:

{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "skills": [],
  "experience": [
    {{
      "role": "",
      "company": "",
      "duration_years": 0.0,
      "skills": []
    }}
  ],
  "education": []
}}

Resume Text:
{text[:4000]}
"""


# ---------- Normalization ----------
def normalize_candidate(
    llm_output: str,
    resume_path: str
) -> Optional[Dict]:
    try:
        data = json.loads(llm_output)

        # Read raw resume text if available to extract date ranges
        raw_text = ""
        try:
            p = Path(resume_path)
            if p.exists():
                raw_text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw_text = ""

        # Helper: parse dates from tokens like 'Jan 2020', '2020', 'Present'
        def _parse_date_token(tok: str) -> Optional[date]:
            tok = tok.strip()
            if not tok:
                return None
            if tok.lower() in {"present", "current", "now"}:
                return datetime.utcnow().date()
            # Try Year only
            m = re.match(r"^(\d{4})$", tok)
            if m:
                return date(int(m.group(1)), 1, 1)
            # Try Month Year formats
            for fmt in ("%b %Y", "%B %Y", "%b. %Y"):
                try:
                    return datetime.strptime(tok, fmt).date()
                except Exception:
                    continue
            return None

        # Find date ranges in nearby text for a given role/company
        range_regexes = [
            re.compile(r"(\b\d{4})\s*[-–—to]{1,3}\s*(Present|Current|\d{4})", re.I),
            re.compile(r"([A-Za-z]{3,9} \d{4})\s*[-–—to]{1,3}\s*(Present|Current|[A-Za-z]{3,9} \d{4})", re.I),
        ]

        def _find_dates_near(text: str, phrase: str) -> Optional[tuple]:
            # Search for the phrase and look within the same line or nearby 2 lines for date ranges
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if phrase.lower() in line.lower():
                    window = "\n".join(lines[max(0, i - 2): i + 3])
                    for rx in range_regexes:
                        m = rx.search(window)
                        if m:
                            start_tok = m.group(1)
                            end_tok = m.group(2)
                            start_dt = _parse_date_token(start_tok)
                            end_dt = _parse_date_token(end_tok)
                            return (start_dt, end_dt)
            return None

        # Exclude internships/training/project roles unless explicitly full-time
        def _is_excluded_role(role: str, company: str) -> bool:
            s = " ".join(filter(None, [role or "", company or ""]) ).lower()
            if any(k in s for k in ("intern", "internship", "trainee", "training")):
                return True
            if "project" in s and "full-time" not in s:
                return True
            return False

        dated_intervals: List[tuple] = []  # (start_date, end_date)
        processed_experience: List[Dict] = []
        undated_duration_total = 0.0

        for exp in data.get("experience", []):
            role = exp.get("role") or None
            company = exp.get("company") or None
            duration = None
            try:
                if exp.get("duration_years") is not None:
                    duration = float(exp.get("duration_years") or 0)
            except Exception:
                duration = None

            # Skip excluded roles like internships
            if _is_excluded_role(role or "", company or ""):
                continue

            start_dt = None
            end_dt = None

            # Try to find dates near role/company in raw text
            phrase = " ".join(filter(None, [role or "", company or ""]))
            if phrase and raw_text:
                found = _find_dates_near(raw_text, phrase)
                if found:
                    start_dt, end_dt = found

            # If date tokens not found but duration exists, treat as undated duration
            if start_dt is None and end_dt is None and duration:
                undated_duration_total += duration
                processed_experience.append({
                    "role": role,
                    "company": company,
                    "start": None,
                    "end": None,
                    "duration_years": round(duration, 2),
                    "skills": exp.get("skills", []),
                })
                continue

            # If we have at least start or end, normalize and add dated interval
            if start_dt or end_dt:
                if start_dt is None:
                    # If only end is present, assume a 1-year duration ending at end_dt (conservative)
                    start_dt = date(end_dt.year - 1, end_dt.month, end_dt.day) if end_dt else None
                if end_dt is None:
                    end_dt = datetime.utcnow().date()

                # Normalize to dates (start at first day of year/month if needed)
                def _normalize_day(d: date) -> date:
                    return date(d.year, d.month if d.month else 1, 1)

                s_norm = date(start_dt.year, start_dt.month if start_dt.month else 1, 1)
                e_norm = date(end_dt.year, end_dt.month if end_dt.month else 1, 1)

                if e_norm < s_norm:
                    # swap if somehow reversed
                    s_norm, e_norm = e_norm, s_norm

                dated_intervals.append((s_norm, e_norm))

                # compute duration in years for this role
                days = (e_norm - s_norm).days
                dur_years = round(days / 365.25, 2) if days >= 0 else 0.0

                processed_experience.append({
                    "role": role,
                    "company": company,
                    "start": s_norm.isoformat(),
                    "end": ("Present" if end_dt == datetime.utcnow().date() else e_norm.isoformat()),
                    "duration_years": dur_years,
                    "skills": exp.get("skills", []),
                })
                continue

            # Fallback: include the role with whatever fields available (no dates)
            processed_experience.append({
                "role": role,
                "company": company,
                "start": None,
                "end": None,
                "duration_years": round(duration, 2) if duration else None,
                "skills": exp.get("skills", []),
            })

        # Merge overlapping dated intervals
        merged_days = 0
        if dated_intervals:
            intervals = sorted(dated_intervals, key=lambda x: x[0])
            merged = [intervals[0]]
            for s, e in intervals[1:]:
                last_s, last_e = merged[-1]
                if s <= last_e:
                    # overlap -> extend
                    merged[-1] = (last_s, max(last_e, e))
                else:
                    merged.append((s, e))
            # sum days
            for s, e in merged:
                merged_days += (e - s).days

        merged_years = round(merged_days / 365.25, 2) if merged_days else 0.0

        total_years = merged_years + round(undated_duration_total, 2)

        # Final rounding to one decimal per rules
        total_years = round(total_years, 1)

        candidate_doc = {
            "candidate_id": data.get("candidate_id") or Candidate().candidate_id,
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "phone": data.get("phone", ""),
            "location": data.get("location", ""),
            "skills": list(dict.fromkeys(data.get("skills", []))),
            "experience": processed_experience,
            "total_experience_years": total_years,
            "education": data.get("education", []),
            "resume_path": resume_path,
        }

        return candidate_doc

    except Exception as exc:
        logger.error("Failed to normalize LLM output", exc_info=exc)
        return None


# ---------- Async Worker ----------
async def process_resume(
    queue: asyncio.Queue,
    db: MongoDBManager
) -> None:
    while True:
        try:
            path: Path = await queue.get()
        except asyncio.CancelledError:
            break

        print(f"Worker processing: {path}")
        raw_text = extract_raw_text(path)
        if not raw_text:
            db.log(
                level="ERROR",
                module="extractor",
                message="No text extracted",
                meta={"file": path.name},
            )
            queue.task_done()
            continue

        prompt = build_resume_prompt(raw_text)
        llm_response = llm_resume_extract(prompt, str(path))

        if not llm_response:
            queue.task_done()
            continue

        candidate_data = normalize_candidate(llm_response, str(path))
        if not candidate_data:
            queue.task_done()
            continue

        try:
            db.insert_candidate(candidate_data)
            db.insert_raw_resume(
                {
                    "candidate_id": candidate_data["candidate_id"],
                    "file_name": path.name,
                    "file_path": str(path),
                    "status": "processed",
                }
            )
            logger.info("Inserted candidate %s into DB", candidate_data.get("candidate_id"))
            # Also print to stdout so the CLI user sees progress even if logging is redirected
            print(f"Inserted candidate: {candidate_data.get('candidate_id')} from {path.name}")
        except Exception:
            logger.exception("DB insert failed; printing candidate data")
            print("Candidate data (DB insert failed):")
            print(json.dumps(candidate_data, indent=2))

        queue.task_done()


# ---------- Public API ----------
async def ingest_resumes(folder: Path) -> None:
    db = MongoDBManager()
    queue: asyncio.Queue = asyncio.Queue()

    resume_files = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt"}
    ]

    for path in resume_files:
        queue.put_nowait(path)

    workers = [asyncio.create_task(process_resume(queue, db)) for _ in range(3)]

    # Wait for all items to be processed
    await queue.join()

    for w in workers:
        w.cancel()
