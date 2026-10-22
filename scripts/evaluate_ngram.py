import hashlib
import json
from pathlib import Path
from dataclasses import asdict

from collections import Counter
from autocomplete.normalization import NORMALIZATION_VERSION
from autocomplete.data import DevelopmentCase, partition_record
from autocomplete.normalization import normalize_term
from autocomplete.evaluation import score_case, summarize_scores
from autocomplete.contract import MAX_ADDED_LENGTH, MAX_INPUT_LENGTH
from autocomplete.ngram import NgramModel
from datasets import Dataset
from scripts.prepare_data import MANIFEST, OUTPUT
from scripts.probe_amazonqac import RAW_DATA
from scripts.evaluate_popularity import verify_file_hash

def main() -> None:
    manifest = json.loads(MANIFEST.read_bytes())
    verify_file_hash(RAW_DATA, manifest["hashes"]["raw_file SHA-256"])
    verify_file_hash(OUTPUT, manifest["hashes"]["development_cases SHA-256"])

    raw_data = Dataset.from_parquet(str(RAW_DATA))
    partition = partition_record(raw_data.to_list())

    model = NgramModel()
    model.fit(
        normalize_term(record["final_search_term"])
        for record in partition.train
    )
    
    saved_rows = json.loads(OUTPUT.read_bytes())
    cases = [DevelopmentCase(**row) for row in saved_rows]
    
    results = []
    scores = []
    
    for case in cases:
        prediction = model.greedy_complete(case.raw_prefix)
        completion = prediction["completion"]
        suggestions = [] if completion is None else [completion]
        
        hit, rr = score_case(
            suggestions,
            case.normalized_final_search_term
        )
        
        results.append({
            **asdict(case),
            "suggestions": suggestions,
            "log_score": prediction["log_score"],
            "stop_reason": prediction["stop_reason"],
            "hit_at_10": hit,
            "reciprocal_rank_at_10": rr,
        })
        scores.append((hit, rr))
    
    summary = summarize_scores(scores)
    run_id = "char_ngram_greedy_v1"

    artifact = {
        "run_id": run_id,
        "split": "september_development",
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()
        ).hexdigest(),
        "input_hashes": manifest["hashes"],
        "normalization_version": NORMALIZATION_VERSION,
        "model": {
            "type": "character_ngram",
            "order": model.order,
            "alpha": model.alpha,
            "training_interactions": len(partition.train),
            "alphabet_size_including_eos": len(model.alphabet),
        },
        "serving_policy": {
            "version": "contract-v1",
            "decoder": "greedy",
            "max_suggestions": 1,
            "max_input_length": MAX_INPUT_LENGTH,
            "max_added_length": MAX_ADDED_LENGTH,
        },
        "stop_reason_counts": dict(
            Counter(row["stop_reason"] for row in results)
        ),
        "summary": summary,
        "results": results,
    }

    result_path = (
        Path("artifacts")
        / run_id
        / "development_results.json"
    )

    payload = json.dumps(
        artifact,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")

    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        if result_path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite different results: {result_path}"
            )
        print(f"Reusing identical results: {result_path}")
    else:
        with result_path.open("xb") as output_file:
            output_file.write(payload)
        print(f"Saved results: {result_path}")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()