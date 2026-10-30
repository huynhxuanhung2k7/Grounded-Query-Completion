"""Evaluate a saved method on prepared development or official cases."""

import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from autocomplete.artifacts import (
    load_development_inputs,
    save_evaluation_artifact,
    verify_file_hash,
)
from autocomplete.config import (
    BEAM_WIDTH,
    CANDIDATE_BUDGET,
    CHECKPOINT,
    HISTORY_CATALOG,
    INPUT,
    MANIFEST,
    OFFICIAL_CASES,
    OFFICIAL_MANIFEST,
    OUTPUT,
)
from autocomplete.data import DevelopmentCase
from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.evaluation import evaluate_cases
from autocomplete.predictors import make_suggester

OUTPUT_ROOT = Path("artifacts/results")


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number


def arguments(argv, *, demo=False):
    """Read shared options for terminal suggestions or batch evaluation.

    Args:
        argv: Command-line arguments, or None for the terminal arguments.
        demo: Include prefix input instead of evaluation split/output options.
    Returns:
        Parsed options and their argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Try or evaluate the autocomplete models."
    )
    parser.add_argument(
        "--model",
        choices=("popularity", "neural", "ngram", "catalog_neural"),
        required=True,
    )
    parser.add_argument("--decoder", choices=("greedy", "beam"))
    parser.add_argument("--checkpoint", type=Path)

    if demo:
        parser.add_argument("--prefix", help="Omit for an interactive prompt")
    else:
        parser.add_argument(
            "--split", choices=("development", "official"), default="development"
        )
        parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    parser.add_argument("--beam-width", type=positive_int)
    parser.add_argument("--candidate-budget", type=positive_int)
    parser.add_argument("--grounded", action="store_true")

    args = parser.parse_args(argv)

    if args.model in ("popularity", "catalog_neural"):
        if args.decoder is not None:
            parser.error(f"{args.model} does not use a decoder")
    else:
        args.decoder = args.decoder or "beam"

    if args.decoder != "beam" and (
        args.beam_width is not None or args.candidate_budget is not None
    ):
        parser.error("beam settings require beam decoding")

    if args.checkpoint and args.model not in ("neural", "catalog_neural"):
        parser.error("checkpoint is only used with neural models")

    args.beam_width = args.beam_width or BEAM_WIDTH
    args.candidate_budget = args.candidate_budget or CANDIDATE_BUDGET

    if args.decoder == "beam" and args.candidate_budget < 10:
        parser.error("candidate-budget must be at least 10")

    if args.grounded and (
        args.model not in ("neural", "ngram") or args.decoder != "beam"
    ):
        parser.error("--grounded requires neural/ngram beam decoding")

    return args, parser


def load_official_cases():
    """Use supplied October prefixes, without changing September training inputs."""
    manifest = json.loads(OFFICIAL_MANIFEST.read_text())

    if manifest["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Official normalization version differs")

    verify_file_hash(OFFICIAL_CASES, manifest["cases_sha256"])
    cases = [DevelopmentCase(**row) for row in json.loads(OFFICIAL_CASES.read_text())]

    if len(cases) != manifest["request_count"]:
        raise ValueError("Official-case count differs from manifest")

    return cases


def main(argv: list[str] | None = None) -> None:
    args, _ = arguments(argv)

    if args.split == "development":
        _, cases = load_development_inputs(MANIFEST, INPUT, OUTPUT)
        cases_path = OUTPUT
    else:
        cases = load_official_cases()
        cases_path = OFFICIAL_CASES

    suggest = make_suggester(
        args.model,
        decoder=args.decoder,
        checkpoint_path=args.checkpoint or CHECKPOINT,
        raw_path=INPUT,
        manifest_path=MANIFEST,
        beam_width=args.beam_width,
        candidate_budget=args.candidate_budget,
        grounded=args.grounded,
    )
    latencies = []

    def timed_predict(prefix):
        started = perf_counter()
        prediction = suggest(prefix)
        latencies.append((perf_counter() - started) * 1000)

        return prediction

    def progress(done, total):
        if done == 1 or done % 100 == 0 or done == total:
            print(f"Completed {done}/{total}", flush=True)

    print(f"Starting {args.model}: {len(cases)} {args.split} cases", flush=True)
    result = evaluate_cases(cases, timed_predict, progress=progress)

    p95_ms = None

    if latencies:
        sorted_times = sorted(latencies)

        rank = ceil(0.95 * len(sorted_times))

        p95_ms = sorted_times[rank - 1]

    summary = {
        "dataset": "amazon/AmazonQAC",
        "split": args.split,
        "method": args.model,
        "cases_path": str(cases_path),
        **result["summary"],
        **result["prediction_summary"],
        "p95_ms": p95_ms,
    }
    run_id = args.model

    if args.model in ("neural", "catalog_neural"):
        summary["checkpoint_path"] = str(args.checkpoint or CHECKPOINT)

    if args.model in ("popularity", "catalog_neural") or args.grounded:
        summary["catalog_path"] = str(HISTORY_CATALOG)
        run_id += "_history_100k"

    if args.decoder:
        summary.update(decoder=args.decoder, grounded=args.grounded)
        run_id += f"_{args.decoder}"

    if args.decoder == "beam":
        summary.update(
            beam_width=args.beam_width, candidate_budget=args.candidate_budget
        )
        run_id += f"_b{args.beam_width}_m{args.candidate_budget}"

    if args.grounded:
        run_id += "_grounded"

    if args.model == "catalog_neural":
        summary["scoring"] = "batched_exhaustive"
        run_id += "_batched"

    path = args.output_root / args.split / run_id / "summary.json"
    created = save_evaluation_artifact(path, summary)
    print(f"{'Saved' if created else 'Reused'}: {path}")
    print(json.dumps(json.loads(path.read_text()), indent=2))


if __name__ == "__main__":
    main()
