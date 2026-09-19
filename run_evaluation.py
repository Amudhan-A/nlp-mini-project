"""
run_evaluation.py
-----------------
One command to run the full end-to-end evaluation for the mini project.

Stages, in dependency order:

  1. prepare_data.py       NFCorpus -> data/processed/chunks.json          (skipped if present)
  2. build_qa_dataset.py   -> data/qa_dataset.json                         (skipped if present)
  3. baseline.py --compare TF-IDF vs BM25 -> results/baseline_results.json
  4. dense_retrieval.py    builds the Chroma index in data/vector_db AND
                           evaluates dense retrieval -> results/dense_retrieval_results.json
  5. qa_reader.py          DistilBERT extractive QA  -> results/transformer_qa_results.json
  6. llama_qa.py           Llama 3.1 8B via Ollama   -> results/llama_qa_results.json
  7. error_analysis.py     10 worst cases            -> results/error_analysis_10.{json,md}
  8. final_comparison.py   the summary table         -> results/final_comparison.json

Notes
-----
* data/vector_db is gitignored, so stage 4 has to run at least once on this
  machine before stages 5-8 (they call DenseRetriever() without build_index()).
* Stage 6 needs Ollama running locally with llama3.1:8b pulled. If it is not
  reachable the stage is skipped with a warning (unless --require-llama).
* Stage 7 reads results/llama_qa_results.json. The repo ships one, so it works
  even when stage 6 is skipped.

Usage
-----
    python run_evaluation.py                     # full pipeline
    python run_evaluation.py --skip-llama        # no Ollama on this machine
    python run_evaluation.py --from dense        # resume from a stage
    python run_evaluation.py --only baseline dense
    python run_evaluation.py --list
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "llama3.1:8b"

# name, script, args, produces, needs
STAGES = [
    ("prepare",   "prepare_data.py",     [],            ["data/processed/chunks.json"], []),
    ("dataset",   "build_qa_dataset.py", [],            ["data/qa_dataset.json"],       ["data/processed/chunks.json"]),
    ("baseline",  "baseline.py",         ["--compare"], ["results/baseline_results.json"],
     ["data/processed/chunks.json", "data/qa_dataset.json"]),
    ("dense",     "dense_retrieval.py",  [],            ["results/dense_retrieval_results.json"],
     ["data/processed/chunks.json", "data/qa_dataset.json"]),
    ("reader",    "qa_reader.py",        [],            ["results/transformer_qa_results.json"],
     ["data/vector_db"]),
    ("llama",     "llama_qa.py",         [],            ["results/llama_qa_results.json"],
     ["data/vector_db"]),
    ("errors",    "error_analysis.py",   [],            ["results/error_analysis_10.json"],
     ["data/vector_db", "results/llama_qa_results.json"]),
    ("compare",   "final_comparison.py", [],            ["results/final_comparison.json"],
     ["results/baseline_results.json", "results/dense_retrieval_results.json",
      "results/transformer_qa_results.json", "results/llama_qa_results.json"]),
]
STAGE_NAMES = [s[0] for s in STAGES]

# stages that are expensive but idempotent: skipped if their output already exists
SKIP_IF_PRESENT = {"prepare", "dataset"}


def ollama_ready():
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
        return any(n.startswith(OLLAMA_MODEL.split(":")[0]) for n in names), names
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False, []


def check_deps():
    missing = []
    for mod, pkg in [("sklearn", "scikit-learn"), ("rank_bm25", "rank_bm25"),
                     ("sentence_transformers", "sentence-transformers"),
                     ("chromadb", "chromadb"), ("transformers", "transformers"),
                     ("torch", "torch")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def run_stage(name, script, args, produces, force=False):
    out_paths = [ROOT / p for p in produces]
    if not force and name in SKIP_IF_PRESENT and all(p.exists() for p in out_paths):
        print(f"\n### [{name}] output already present -> skipping ({produces[0]})")
        return {"stage": name, "status": "skipped-cached", "seconds": 0.0}

    cmd = [PY, script] + args
    print(f"\n{'=' * 70}\n### [{name}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"--- [{name}] {status} in {dt:.1f}s")
    return {"stage": name, "status": status, "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=STAGE_NAMES)
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES)
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--require-llama", action="store_true",
                    help="fail instead of skipping when Ollama is unreachable")
    ap.add_argument("--force", action="store_true",
                    help="re-run the cached preprocessing stages too")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, script, extra, produces, needs in STAGES:
            print(f"{name:<10} {script} {' '.join(extra):<10} -> {produces[0]}")
        return

    missing = check_deps()
    if missing:
        print("Missing packages: " + ", ".join(missing))
        print(f"Install with:  {PY} -m pip install -r requirements_updated.txt")
        sys.exit(1)

    selected = STAGES
    if args.only:
        selected = [s for s in STAGES if s[0] in args.only]
    elif args.from_stage:
        selected = STAGES[STAGE_NAMES.index(args.from_stage):]

    llama_ok, tags = ollama_ready()
    if not llama_ok and any(s[0] == "llama" for s in selected) and not args.skip_llama:
        if args.require_llama:
            raise SystemExit(
                f"Ollama not reachable at {OLLAMA_TAGS_URL} (models seen: {tags}).\n"
                f"Start it with `ollama serve` and `ollama pull {OLLAMA_MODEL}`.")
        print(f"\n!! Ollama not reachable at {OLLAMA_TAGS_URL} -- skipping the llama stage.")
        print(f"   To include it: `ollama serve` + `ollama pull {OLLAMA_MODEL}`, then re-run.")
        args.skip_llama = True
    if args.skip_llama:
        selected = [s for s in selected if s[0] != "llama"]

    report, t_start = [], time.perf_counter()
    for name, script, extra, produces, needs in selected:
        blockers = [n for n in needs if not (ROOT / n).exists()]
        if blockers:
            print(f"\n!! [{name}] missing prerequisite(s): {blockers} -- skipping. "
                  f"Run the earlier stages first.")
            report.append({"stage": name, "status": "skipped-missing-input", "seconds": 0.0})
            continue
        report.append(run_stage(name, script, extra, produces, force=args.force))

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for row in report:
        print(f"  {row['stage']:<10} {row['status']:<24} {row['seconds']:>8.1f}s")
    print(f"  {'TOTAL':<10} {'':<24} {time.perf_counter() - t_start:>8.1f}s")

    final = ROOT / "results" / "final_comparison.json"
    if final.exists():
        print(f"\nFinal comparison table -> {final}")
        try:
            print(json.dumps(json.loads(final.read_text(encoding='utf-8')), indent=2)[:2000])
        except json.JSONDecodeError:
            pass

    if any("FAILED" in r["status"] for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
