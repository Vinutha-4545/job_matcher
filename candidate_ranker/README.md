# candidate_ranker

Lightweight CLI candidate ranking tool. This project provides a command-line interface to ingest resumes, add job descriptions, and rank candidates for a job using local LLM placeholders and sentence-transformers embeddings.

Structure

candidate_ranker/
├── main.py               # CLI entry
├── config.py             # constants, DB URI, LLM model path
├── models.py             # dataclass for Candidate and Job
├── db.py                 # MongoDB manager class
├── extractor.py          # resume parsing + LLM normalization
├── job_processor.py      # JD parsing + skill extraction
├── scorer.py             # similarity, experience calc, total score
├── ranker.py             # top-10 logic
├── utils.py              # helpers (llm call wrapper with cache + rate limit)
└── requirements.txt

Usage

1. Install dependencies (create a virtualenv first):

```powershell
python -m pip install -r candidate_ranker/requirements.txt
```

2. Ensure MongoDB is running locally or set `MONGO_URI` env var.

3. Ingest resumes:

```powershell
python -m candidate_ranker.main ingest --folder C:\path\to\resumes
```

4. Add a job:

```powershell
python -m candidate_ranker.main add-job --id JD001 --file jd.txt
```

5. Rank candidates for the job:

```powershell
python -m candidate_ranker.main rank --job-id JD001
```

Notes

- The LLM calls are currently simulated. Replace `utils.simulated_llm_call` with a real
  local LLM invocation (gpt4all or transformers) for production use.
- Embedding model defaults to `all-MiniLM-L6-v2` (configurable via env).
- This is a starting point; further work is required to parse PDF/DOCX reliably with LLMs,
  improve extraction, add caching, and robust error handling for production.
