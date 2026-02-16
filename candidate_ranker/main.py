# main.py
import argparse
import asyncio
import logging
from pathlib import Path

# absolute import allows running as script or module without import errors
try:
    from candidate_ranker.config import LOG_DIR, LOG_FILE
except ImportError:
    # fallback if package not on path (e.g. executed inside package directory)
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
    rank_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top candidates to show (default 10)",
    )

    args = parser.parse_args()

    try:
        # if no sub‑command provided, execute automatic bulk load and
        # prompt the user to pick a job for ranking
        if args.command is None:
            # ---- bulk ingest from bundled data folders ----
            from .extractor import ingest_resumes
            from .job_processor import add_job

            # load all job description files found in data/jds
            jds_dir = Path(__file__).parent / "data" / "jds"
            if jds_dir.exists():
                for path in jds_dir.iterdir():
                    if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}:
                        job_id = path.stem
                        try:
                            add_job(job_id, path)
                        except Exception as exc:
                            logging.error("Failed to insert job %s: %s", job_id, exc)

            # ingest resumes from data/resumes (existing behaviour)
            resumes_dir = Path(__file__).parent / "data" / "resumes"
            if resumes_dir.exists():
                asyncio.run(ingest_resumes(resumes_dir))

            # prompt user for job id and optionally top-k
            job_id = input("Enter job ID to rank: ").strip()
            if not job_id:
                print("No job ID entered, exiting.")
            else:
                # default to 5 top candidates
                try:
                    top_str = input("How many top candidates to display [5]? ").strip()
                    top_k = int(top_str) if top_str else 5
                except ValueError:
                    top_k = 5

                from .ranker import rank_candidates
                rank_candidates(job_id, top_k)

        elif args.command == "ingest":
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

            rank_candidates(args.job_id, top_k=args.top)

        else:
            parser.print_help()

    except Exception as exc:
        logging.error("Command failed", exc_info=exc)


if __name__ == "__main__":
    main()
