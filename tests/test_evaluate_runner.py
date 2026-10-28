"""Offline command checks: tiny synthetic cases and checkpoints only."""

import json
import math
from dataclasses import asdict

import pytest
import torch

from autocomplete import artifacts, predictors
from autocomplete.artifacts import canonical_json_bytes, file_sha256
from autocomplete.baseline.ngram import NgramModel
from autocomplete.contract import is_compatible_completion, prepare_prefix
from autocomplete.data import CUTOFF, DevelopmentCase, loading
from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.decoding import (
    NeuralPredictor,
    beam_candidates,
    greedy_complete,
)
from autocomplete.evaluation import (
    beam_prediction,
    completion_prediction,
    evaluate_cases,
    evaluate_popularity,
    prediction_summary,
)
from autocomplete.model.checkpoint import load_checkpoint
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import EOS, CharVocabulary
from datasets import Dataset
from scripts import evaluate as runner
from scripts import suggest as cli


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    def record(query_id, term, date):
        return {
            "query_id": query_id,
            "session_id": str(query_id),
            "prefixes": [term[:1]],
            "first_prefix_typed_time": f"2023-09-{date}T00:00:00",
            "final_search_term": term,
            "search_time": f"2023-09-{date}T00:00:01",
            "popularity": 999,
        }

    raw = tmp_path / "raw.parquet"
    Dataset.from_list(
        [record(1, "cat", "01"), record(2, "car", "02"), record(3, "zzz", "26")]
    ).to_parquet(str(raw))
    cases = [
        DevelopmentCase(3, 0, "ca", "ca", "zzz", "zzz"),
        DevelopmentCase(4, 0, "cat", "cat", "cat", "cat"),
        DevelopmentCase(5, 0, " ", "", "cat", "cat"),
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_bytes(canonical_json_bytes([asdict(case) for case in cases]))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "counts": {"development_cases": len(cases)},
                "hashes": {
                    "raw_file SHA-256": file_sha256(raw),
                    "development_cases SHA-256": file_sha256(cases_path),
                },
            }
        )
    )
    vocabulary = CharVocabulary(["cat", "car"])
    config = {
        "vocab_size": len(vocabulary),
        "context_length": 3,
        "embedding_dim": 2,
        "hidden_dim": 4,
    }
    model = NeuralCharModel(**config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias[vocabulary.token_to_id[EOS]] = 2
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "normalization_version": NORMALIZATION_VERSION,
            "id_to_token": vocabulary.id_to_token,
            "model_config": config,
            "model_state": model.state_dict(),
            "epoch": 4,
            "manifest_sha256": file_sha256(manifest_path),
        },
        checkpoint,
    )
    for name, path in (
        ("INPUT", raw),
        ("OUTPUT", cases_path),
        ("MANIFEST", manifest_path),
        ("CHECKPOINT", checkpoint),
    ):
        monkeypatch.setattr(runner, name, path)
    expanded = tmp_path / "catalog.json"
    expanded.write_bytes(
        canonical_json_bytes(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "cutoff": CUTOFF.isoformat(),
                "term_counts": {"cat": 1, "car": 1},
            }
        )
    )
    monkeypatch.setattr(runner, "HISTORY_CATALOG", expanded)
    monkeypatch.setattr(predictors, "HISTORY_CATALOG", expanded)
    return tmp_path, cases, checkpoint


def saved_summary(root):
    paths = list(root.rglob("summary.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text())


@pytest.mark.parametrize(
    "method,flags",
    [
        ("popularity", []),
        ("ngram", []),
        ("neural", []),
        ("neural", ["--grounded"]),
        ("catalog_neural", []),
    ],
)
def test_compact_summary_all_routes_preserve_checkpoint(local_run, method, flags):
    directory, cases, checkpoint = local_run
    output = directory / "results"
    before = file_sha256(checkpoint)
    args = ["--model", method, "--output-root", str(output), *flags]
    runner.main(args)
    path, summary = saved_summary(output)
    assert summary["request_count"] == len(cases)
    assert summary["method"] == method and summary["split"] == "development"
    assert summary["cases_path"] == str(runner.OUTPUT)
    assert 0 <= summary["mrr_at_10"] <= summary["success_at_10"] <= 1
    assert summary["response_coverage"] == 2 / 3
    assert summary["p95_ms"] >= 0
    assert "results" not in summary and "serving_policy" not in summary
    original = path.read_bytes()
    runner.main(args)
    assert path.read_bytes() == original  # Also preserves the original p95.
    assert file_sha256(checkpoint) == before


@pytest.mark.parametrize("method", ["ngram", "neural"])
def test_free_and_grounded_use_same_pool_and_selected_catalog(
    local_run, monkeypatch, method
):
    pool = {
        "candidates": [
            {"completion": "cat", "log_score": -1.0},
            {"completion": "cart", "log_score": -2.0},
        ],
        "stop_reason": "exhausted",
    }
    monkeypatch.setattr(predictors, "beam_candidates", lambda *_: pool)
    monkeypatch.setattr(predictors, "load_catalog", lambda: ({"cart"}, {"cart": 1}))
    outputs = []
    for grounded in (False, True):
        suggest = predictors.make_suggester(
            method,
            raw_path=runner.INPUT,
            manifest_path=runner.MANIFEST,
            checkpoint_path=runner.CHECKPOINT,
            grounded=grounded,
        )
        outputs.append(suggest("ca"))
    assert outputs[0]["suggestions"] == ["cat", "cart"]
    assert outputs[1]["suggestions"] == ["cart"]
    assert outputs[0]["candidates"] == outputs[1]["candidates"] == pool["candidates"]


def test_changed_beam_settings_use_separate_paths(local_run):
    directory, _, _ = local_run
    output = directory / "results"
    for width in (1, 2):
        runner.main(
            [
                "--model",
                "neural",
                "--output-root",
                str(output),
                "--beam-width",
                str(width),
                "--candidate-budget",
                "10",
            ]
        )
    summaries = [json.loads(p.read_text()) for p in output.rglob("summary.json")]
    assert {s["beam_width"] for s in summaries} == {1, 2}


@pytest.mark.parametrize("method", ["popularity", "ngram", "neural"])
def test_evaluation_verifies_raw_source_once(local_run, monkeypatch, method):
    directory, _, _ = local_run
    verified = []
    original = artifacts.verify_file_hash

    def verify(path, expected):
        verified.append(path)
        original(path, expected)

    monkeypatch.setattr(artifacts, "verify_file_hash", verify)
    monkeypatch.setattr(loading, "verify_file_hash", verify)
    runner.main(["--model", method, "--output-root", str(directory)])
    assert verified.count(runner.INPUT) == 1


@pytest.mark.parametrize("method", ["popularity", "ngram", "neural"])
def test_lookup_and_greedy_metrics_unchanged(local_run, method):
    directory, cases, checkpoint = local_run
    args = ["--model", method, "--output-root", str(directory)]
    if method == "popularity":
        expected = evaluate_popularity({"cat", "car"}, {"cat": 1, "car": 1}, cases)
    else:
        args += ["--decoder", "greedy"]
        if method == "ngram":
            model = NgramModel()
            model.fit(["cat", "car"])
            predict = model.next_log_probs
        else:
            model, vocabulary, _ = load_checkpoint(checkpoint)
            predict = NeuralPredictor(model, vocabulary).next_log_probs
        expected = evaluate_cases(
            cases,
            lambda prefix: completion_prediction(greedy_complete(prefix, predict)),
        )
    runner.main(args)
    _, summary = saved_summary(directory)
    for key, value in expected["summary"].items():
        assert summary[key] == value


@pytest.mark.parametrize(
    "extra",
    [
        ["--beam-width", "0"],
        ["--candidate-budget", "9"],
        ["--model", "ngram", "--checkpoint", "unused.pt"],
        ["--model", "popularity", "--decoder", "greedy"],
        ["--decoder", "greedy", "--beam-width", "3"],
        ["--decoder", "greedy", "--candidate-budget", "10"],
        ["--decoder", "greedy", "--grounded"],
        ["--model", "popularity", "--grounded"],
        ["--model", "catalog_neural", "--decoder", "beam"],
    ],
)
def test_invalid_cli_fails_before_loading_data(monkeypatch, extra):
    monkeypatch.setattr(
        runner,
        "load_development_inputs",
        lambda *_: pytest.fail("Invalid args must not read data"),
    )
    with pytest.raises(SystemExit) as error:
        runner.main(["--model", "neural", *extra])
    assert error.value.code == 2


def test_generators_default_to_beam():
    args, _ = runner.arguments(["--model", "neural"])
    assert args.decoder == "beam"


def test_checkpoint_manifest_mismatch_creates_no_results(local_run):
    directory, _, checkpoint = local_run
    saved = torch.load(checkpoint, weights_only=True)
    saved["manifest_sha256"] = "wrong"
    torch.save(saved, checkpoint)
    output = directory / "results"
    with pytest.raises(ValueError, match="different manifest"):
        runner.main(["--model", "neural", "--output-root", str(output)])
    assert not output.exists()


def test_bad_predictor_publishes_no_metrics(local_run, monkeypatch):
    directory, _, _ = local_run
    monkeypatch.setattr(NeuralPredictor, "next_log_probs", lambda *_: {EOS: math.nan})
    with pytest.raises(ValueError, match="log probabilities"):
        runner.main(["--model", "neural", "--output-root", str(directory)])
    assert not list(directory.rglob("summary.json"))


def test_static_scoring_does_not_generate(local_run, monkeypatch):
    def unexpected(*_):
        pytest.fail("Static scoring must not generate candidates")

    monkeypatch.setattr(predictors, "beam_candidates", unexpected)
    monkeypatch.setattr(predictors, "greedy_complete", unexpected)
    suggest = predictors.make_suggester(
        "catalog_neural",
        raw_path=runner.INPUT,
        manifest_path=runner.MANIFEST,
        checkpoint_path=runner.CHECKPOINT,
    )
    assert set(suggest("ca")["suggestions"]) == {"cat", "car"}
    assert suggest(" ")["suggestions"] == []


def scalar_catalog_reference(prefix, catalog, predictor):
    """Small independent oracle, not a second production catalog scorer."""
    prefix, reason = prepare_prefix(prefix)
    if reason:
        return []
    rows = []
    for term in sorted(catalog):
        if not is_compatible_completion(prefix, term):
            continue
        total = 0.0
        for offset, token in enumerate([*term[len(prefix) :], EOS]):
            context = term[: len(prefix) + offset]
            total += predictor.next_log_probs(context).get(token, -math.inf)
        if math.isfinite(total):
            rows.append({"completion": term, "log_score": total})
    return sorted(rows, key=lambda row: (-row["log_score"], row["completion"]))


@pytest.mark.parametrize("batch_size", [1, 3, 256])
@pytest.mark.parametrize("prefix", ["ca", "cat", "ωca", "zz", " "])
def test_static_batches_match_scalar_reference(prefix, batch_size):
    torch.manual_seed(7)
    vocabulary = CharVocabulary(["ca", "cat", "car"])
    model = NeuralCharModel(
        len(vocabulary), context_length=3, embedding_dim=2, hidden_dim=4
    )
    predictor = NeuralPredictor(model, vocabulary)
    catalog = {"ca", "cat", "car", "ca?", "ωcat", "ωcar", "cat" + "a" * 65}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    reference = scalar_catalog_reference(prefix, catalog, predictor)
    batched = predictor.score_catalog_terms(prefix, catalog, batch_size)
    assert [row["completion"] for row in batched] == [
        row["completion"] for row in reference
    ]
    assert [row["log_score"] for row in batched] == pytest.approx(
        [row["log_score"] for row in reference], abs=1e-5
    )
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )


def test_static_batches_bound_windows_and_keep_final_partial_batch():
    vocabulary = CharVocabulary(["ca", "cat", "car"])
    model = NeuralCharModel(
        len(vocabulary), context_length=3, embedding_dim=2, hidden_dim=4
    )
    predictor = NeuralPredictor(model, vocabulary)
    sizes = []
    hook = model.register_forward_pre_hook(
        lambda _model, inputs: sizes.append(inputs[0].shape[0])
    )
    try:
        predictor.score_catalog_terms("ca", {"ca", "cat", "car"}, 3)
    finally:
        hook.remove()
    assert sizes == [3, 2]  # EOS only for ca, plus two windows each for car/cat.


def test_static_evaluator_and_cli_never_use_scalar_inference(local_run, monkeypatch):
    directory, _, _ = local_run
    calls = []
    original = NeuralPredictor.next_log_probs_batch

    def batch(self, texts):
        calls.append(len(texts))
        return original(self, texts)

    def unexpected(*_):
        pytest.fail("catalog_neural must use the batch scorer, not scalar inference")

    monkeypatch.setattr(NeuralPredictor, "next_log_probs", unexpected)
    monkeypatch.setattr(NeuralPredictor, "next_log_probs_batch", batch)
    runner.main(
        ["--model", "catalog_neural", "--output-root", str(directory / "batched")]
    )
    path, summary = saved_summary(directory / "batched")
    assert summary["request_count"] == 3
    assert summary["scoring"] == "batched_exhaustive"
    assert path.parent.name.endswith("_batched")
    assert max(calls) > 1

    for name in ("INPUT", "MANIFEST", "CHECKPOINT"):
        monkeypatch.setattr(cli, name, getattr(runner, name))
    calls.clear()
    cli.main(["--model", "catalog_neural", "--prefix", "ca"])
    assert calls == [4]


@pytest.mark.parametrize("score", [math.nan, math.inf, 0.01])
def test_invalid_static_batch_publishes_no_metrics(local_run, monkeypatch, score):
    directory, _, _ = local_run

    def invalid(self, texts):
        return torch.full((len(texts), len(self.vocabulary)), score)

    monkeypatch.setattr(NeuralPredictor, "next_log_probs_batch", invalid)
    with pytest.raises(ValueError, match="log probabilities"):
        runner.main(
            ["--model", "catalog_neural", "--output-root", str(directory / "invalid")]
        )
    assert not list((directory / "invalid").rglob("summary.json"))


def test_official_synthetic_cases_keep_september_training(local_run, monkeypatch):
    directory, _, _ = local_run
    official_cases = directory / "fake_october.json"
    official_cases.write_text(
        json.dumps(
            [
                asdict(DevelopmentCase(800, 0, "ca", "ca", "zzz", "zzz")),
            ]
        )
    )
    official_manifest = directory / "fake_october_manifest.json"
    official_manifest.write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "cases_sha256": file_sha256(official_cases),
                "request_count": 1,
            }
        )
    )
    monkeypatch.setattr(runner, "OFFICIAL_CASES", official_cases)
    monkeypatch.setattr(runner, "OFFICIAL_MANIFEST", official_manifest)
    fitted = []
    original = NgramModel.fit

    def fit(self, terms):
        fitted.extend(terms)
        return original(self, terms)

    monkeypatch.setattr(NgramModel, "fit", fit)
    runner.main(
        [
            "--model",
            "ngram",
            "--split",
            "official",
            "--output-root",
            str(directory / "official_result"),
        ]
    )
    _, summary = saved_summary(directory / "official_result")
    assert fitted == ["cat", "car"]  # Neither September D nor October truth trains it.
    assert summary["request_count"] == 1 and summary["split"] == "official"
    assert summary["cases_path"] == str(official_cases)


def test_p95_times_predictions_only_and_empty_data(local_run, monkeypatch):
    directory, cases, _ = local_run
    times = iter([0, 0.001, 1, 1.002, 2, 2.010])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        runner,
        "make_suggester",
        lambda *_a, **_k: lambda _: {"suggestions": [], "stop_reason": None},
    )
    runner.main(["--model", "popularity", "--output-root", str(directory / "timed")])
    _, summary = saved_summary(directory / "timed")
    assert summary["p95_ms"] == pytest.approx(10)
    assert summary["success_at_10"] == 0 and summary["request_count"] == len(cases)
    monkeypatch.setattr(runner, "load_development_inputs", lambda *_: ({}, []))
    runner.main(["--model", "popularity", "--output-root", str(directory / "empty")])
    _, summary = saved_summary(directory / "empty")
    assert summary["p95_ms"] is None and summary["mrr_at_10"] is None


@pytest.mark.parametrize("interactive", [False, True])
def test_user_cli_loads_once_and_prints_suggestions(monkeypatch, capsys, interactive):
    loaded, prefixes = [], []

    def load(*args, **kwargs):
        loaded.append(args)

        def suggest(prefix):
            prefixes.append(prefix)
            return {"suggestions": ["cat"], "stop_reason": None}

        return suggest

    monkeypatch.setattr(cli, "make_suggester", load)
    if interactive:
        answers = iter(["ca", "cat", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        cli.main(["--model", "neural"])
    else:
        cli.main(["--model", "neural", "--prefix", "ca"])
    assert len(loaded) == 1
    assert prefixes == (["ca", "cat"] if interactive else ["ca"])
    assert "1. cat" in capsys.readouterr().out


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 0.01])
@pytest.mark.parametrize("decode", [beam_candidates, greedy_complete])
def test_invalid_scores_fail_inside_decoder(score, decode):
    with pytest.raises(ValueError, match="log probabilities"):
        decode("cat", lambda prefix: {EOS: score})


def test_neural_predictor_emits_only_training_characters_and_eos(local_run):
    _, _, checkpoint = local_run
    model, vocabulary, _ = load_checkpoint(checkpoint)
    scores = NeuralPredictor(model, vocabulary).next_log_probs("ca")
    assert set(scores) == set(vocabulary.id_to_token[2:])


def test_decoders_allow_impossible_tokens_without_changing_scores():
    scores = {EOS: 0.0, "a": -math.inf}
    result = beam_prediction(beam_candidates("cat", lambda _: scores))
    assert result["suggestions"] == ["cat"]
    assert result["candidates"] == [{"completion": "cat", "log_score": 0.0}]
    assert greedy_complete("cat", lambda _: scores)["completion"] == "cat"
    assert scores == {EOS: 0.0, "a": -math.inf}


def test_beam_matches_tiny_exhaustive_oracle_with_eos_scores():
    tree = {
        "c": {"a": math.log(0.6), "o": math.log(0.4)},
        "ca": {EOS: math.log(0.5), "t": math.log(0.5)},
        "co": {EOS: math.log(0.75), "t": math.log(0.25)},
        "cat": {EOS: 0.0},
        "cot": {EOS: 0.0},
    }
    expected = []

    def enumerate_paths(text, score):
        for token, log_score in tree[text].items():
            if token == EOS:
                expected.append({"completion": text, "log_score": score + log_score})
            else:
                enumerate_paths(text + token, score + log_score)

    enumerate_paths("c", 0.0)
    expected.sort(
        key=lambda candidate: (-candidate["log_score"], candidate["completion"])
    )
    pool = beam_candidates("c", tree.__getitem__, 10, 10)
    assert pool["candidates"] == expected


def test_coverage_counts_empty_predictions():
    results = [
        {
            "suggestions": ["a"],
            "candidates": [{"completion": "a"}, {"completion": "b"}],
            "normalized_final_search_term": "b",
            "candidate_count": 2,
            "generation_seconds": 0.1,
        },
        {
            "suggestions": [],
            "candidates": [],
            "normalized_final_search_term": "c",
            "candidate_count": 0,
            "generation_seconds": 0.3,
        },
    ]
    assert prediction_summary(results) == {"response_coverage": 0.5}
    assert prediction_summary([]) == {"response_coverage": None}
