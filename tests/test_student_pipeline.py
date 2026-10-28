"""Process tiny manual local files; tests never access AmazonQAC remotely."""

import json
from pathlib import Path

import pytest

from autocomplete import config
from autocomplete.artifacts import file_sha256
from autocomplete.data.loading import read_parquet
from datasets import Dataset
from scripts import process_data as prepare


def record(query_id, *, later=False):
    day = "26" if later else "01"
    return {
        "query_id": query_id,
        "session_id": str(query_id),
        "prefixes": ["c"],
        "first_prefix_typed_time": f"2023-09-{day} 00:00:00",
        "search_time": f"2023-09-{day} 00:00:01",
        "final_search_term": f"cat {query_id}",
        "popularity": 999,
    }


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    indices = (0, 6, 13, 19)
    rows = {i: [record(100 * i + j, later=j == 1) for j in range(4)] for i in indices}
    root = tmp_path / "sample"
    raw, manifest, cases = (
        tmp_path / name for name in ("raw.parquet", "manifest.json", "cases.json")
    )
    for name, value in (
        ("INPUT", raw),
        ("MANIFEST", manifest),
        ("OUTPUT", cases),
        ("SHARD_INDICES", indices),
        ("SAMPLE_ROWS_PER_SHARD", 2),
        ("SAMPLE_ROOT", root),
    ):
        monkeypatch.setattr(prepare, name, value)
    root.mkdir()
    for i, records in rows.items():
        prepare.shard_path(root, i, 2).write_text(
            "".join(json.dumps(row) + "\n" for row in records[:2])
        )
    return tmp_path, rows, raw, manifest, cases


def test_manual_sample_processing_and_safe_rerun(pipeline):
    _, _, raw, manifest, cases = pipeline
    prepare.main(["--dataset", "development"])
    assert len(read_parquet(raw)) == 8
    assert json.loads(manifest.read_text())["counts"] == {
        "source": 8,
        "train": 4,
        "development_cases": 4,
        "quarantine": 0,
    }
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in (raw, manifest, cases)}
    prepare.main(["--dataset", "development"])
    assert before == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in before}


def test_history_includes_sample_once_and_filters_later_rows(pipeline, monkeypatch):
    tmp, rows, raw, manifest, cases = pipeline
    prepare.main(["--dataset", "development"])
    protected = {p: file_sha256(p) for p in (raw, manifest, cases)}
    root = tmp / "history"
    root.mkdir()
    for i, records in rows.items():
        prepare.shard_path(root, i, 4).write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    catalog_path = tmp / "catalog.json"
    monkeypatch.setattr(prepare, "HISTORY_ROOT", root)
    monkeypatch.setattr(prepare, "HISTORY_ROWS_PER_SHARD", 4)
    monkeypatch.setattr(prepare, "HISTORY_CATALOG", catalog_path)
    prepare.main(["--dataset", "history"])
    result = json.loads(catalog_path.read_text())
    assert result["counts"]["source_rows"] == 16
    assert result["counts"]["eligible_history_interactions"] == 12
    assert result["counts"]["additional_eligible_interactions"] == 8
    assert len(set(result["history_query_ids"])) == 12
    assert all(f"cat {100 * i + 1}" not in result["term_counts"] for i in rows)
    assert protected == {p: file_sha256(p) for p in protected}


def test_real_targets_include_original_20k():
    assert len(config.SHARD_INDICES) * config.SAMPLE_ROWS_PER_SHARD == 20_000
    assert len(config.SHARD_INDICES) * config.HISTORY_ROWS_PER_SHARD == 100_000


@pytest.fixture
def official_raw(tmp_path, monkeypatch):
    path = tmp_path / "fake_official.parquet"
    Dataset.from_list(
        [
            {"query_id": 2, "prefix": "Cat ", "final_search_term": "CAT FOOD"},
            {"query_id": 1, "prefix": "dog", "final_search_term": "cat toy"},
        ]
    ).to_parquet(str(path))
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(prepare, "EXPECTED_SHA256", file_sha256(path))
    return path


def test_official_preserves_supplied_prefix_rewrites_order_and_reuse(
    tmp_path, official_raw
):
    cases_path, manifest_path = tmp_path / "cases.json", tmp_path / "manifest.json"
    prepare.prepare_official(official_raw, cases_path, manifest_path)
    cases = json.loads(cases_path.read_text())
    assert [c["query_id"] for c in cases] == [2, 1]
    assert cases[0]["raw_prefix"] == "Cat " and cases[0]["normalized_prefix"] == "cat "
    assert cases[1]["normalized_final_search_term"] == "cat toy"
    info = json.loads(manifest_path.read_text())
    assert info["cases_sha256"] == file_sha256(cases_path)
    assert info["prior_exploratory_probe"]["exposed_query_ids"] == [2, 1]
    before = (cases_path.read_bytes(), manifest_path.read_bytes())
    prepare.prepare_official(official_raw, cases_path, manifest_path)
    assert before == (cases_path.read_bytes(), manifest_path.read_bytes())


def test_official_duplicate_ids_fail_before_writing(
    tmp_path, monkeypatch, official_raw
):
    rows = read_parquet(official_raw)
    for row in rows:
        row["query_id"] = 1
    Dataset.from_list(rows).to_parquet(str(official_raw))
    monkeypatch.setattr(prepare, "EXPECTED_SHA256", file_sha256(official_raw))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare.prepare_official(
            official_raw, tmp_path / "cases.json", tmp_path / "manifest.json"
        )
    assert not (tmp_path / "cases.json").exists()


def test_official_wrong_file_fails_before_writing(tmp_path, official_raw):
    official_raw.write_bytes(b"not the downloaded file")
    with pytest.raises(ValueError, match="checksum"):
        prepare.prepare_official(
            official_raw, tmp_path / "cases.json", tmp_path / "manifest.json"
        )
    assert not (tmp_path / "cases.json").exists()


def test_existing_training_run_is_protected(tmp_path, monkeypatch):
    from scripts import train_model

    monkeypatch.setattr(train_model, "RUN_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        train_model, "load_partition", lambda *args: pytest.fail("Started training")
    )
    with pytest.raises(FileExistsError, match="already exists"):
        train_model.main()


def test_tiny_training_saves_history_and_reloadable_best_model(pipeline, monkeypatch):
    import torch

    from autocomplete.model.checkpoint import load_checkpoint
    from scripts import train_model

    tmp, _, raw, manifest, _ = pipeline
    prepare.main(["--dataset", "development"])
    run = tmp / "tiny_training"
    monkeypatch.setattr(train_model, "INPUT", raw)
    monkeypatch.setattr(train_model, "MANIFEST", manifest)
    monkeypatch.setattr(train_model, "RUN_DIRECTORY", run)
    monkeypatch.setattr(train_model, "MAX_EPOCHS", 1)
    monkeypatch.setattr(
        train_model,
        "MODEL_CONFIG",
        {
            "context_length": 3,
            "embedding_dim": 2,
            "hidden_dim": 4,
            "activation": "relu",
        },
    )
    train_model.main()
    history = json.loads((run / "history.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert Path(summary["best_checkpoint"]) == run / "best.pt"
    assert len(history) == 1 and history[0]["epoch"] == 1
    assert summary["best_validation_loss"] == history[0]["validation"]["loss"]
    model, vocabulary, checkpoint = load_checkpoint(Path(summary["best_checkpoint"]))
    assert (
        not model.training
        and len(vocabulary) == checkpoint["model_config"]["vocab_size"]
    )
    assert checkpoint["manifest_sha256"] == file_sha256(manifest)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
