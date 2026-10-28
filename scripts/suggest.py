"""Try the same predictions as evaluate.py, once or at an interactive prompt."""

from autocomplete.config import CHECKPOINT, INPUT, MANIFEST
from autocomplete.predictors import make_suggester
from scripts.evaluate import arguments


def main(argv: list[str] | None = None) -> None:
    # Reuse the method options instead of maintaining a second set of defaults.
    args, _ = arguments(argv, demo=True)
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
    while True:
        if args.prefix is None:
            try:
                prefix = input("Prefix (empty to quit): ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prefix:
                break
        else:
            prefix = args.prefix

        prediction = suggest(prefix)
        for rank, term in enumerate(prediction["suggestions"], start=1):
            print(f"{rank}. {term}")
        if not prediction["suggestions"]:
            print(f"No suggestions ({prediction['stop_reason'] or 'no_candidates'}).")
        if args.prefix is not None:
            break


if __name__ == "__main__":
    main()
