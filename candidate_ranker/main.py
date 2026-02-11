# main.py
import argparse
import asyncio
import logging
from pathlib import Path

from .config import LOG_DIR, LOG_FILE

# Heavy modules are imported lazily inside CLI branches to avoid import-time failures


# ---------- Logging Setup ----------
def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


# ---------- CLI ----------
def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="CLI Candidate Ranking System"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- ingest ----
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest resumes from a folder"
    )
    ingest_parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing resumes",
    )

    # ---- add-job ----
    job_parser = subparsers.add_parser(
        "add-job", help="Add a job description"
    )
    job_parser.add_argument(
        "--id",
        required=True,
        help="Job ID (e.g. JD001)",
    )
    job_parser.add_argument(
        "--file",
        required=True,
        help="Path to job description text file",
    )

    # ---- rank ----
    rank_parser = subparsers.add_parser(
        "rank", help="Rank candidates for a job"
    )
    rank_parser.add_argument(
        "--job-id",
        required=True,
        help="Job ID to rank against",
    )

    args = parser.parse_args()

    try:
        if args.command == "ingest":
            from .extractor import ingest_resumes

            folder = Path(args.folder)
            if not folder.exists():
                raise FileNotFoundError(
                    f"Resume folder not found: {folder}"
                )
            asyncio.run(ingest_resumes(folder))

        elif args.command == "add-job":
            from .job_processor import add_job

            add_job(args.id, Path(args.file))

        elif args.command == "rank":
            from .ranker import rank_candidates

            rank_candidates(args.job_id)

        else:
            parser.print_help()

    except Exception as exc:
        logging.error("Command failed", exc_info=exc)


if __name__ == "__main__":
    main()
