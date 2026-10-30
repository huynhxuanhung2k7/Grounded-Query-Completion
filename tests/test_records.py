from dataclasses import replace

import pytest

from autocomplete.data.records import (
    DevelopmentCase,
    choose_development_prefix,
    development_cases_digest,
    make_development_cases,
    partition_record,
    write_development_cases,
)


def record(query_id=1, **changes):
    result = {
        "query_id": query_id,
        "session_id": "toy",
        "prefixes": ["c", "ca"],
        "final_search_term": "cat",
        "popularity": 1,
        "first_prefix_typed_time": "2023-09-24 12:00:00",
        "search_time": "2023-09-24 12:00:01",
    }
    result.update(changes)

    return result


def test_partition_uses_whole_interactions_and_retains_quarantine_reasons():
    records = [
        record(1),
        record(
            2,
            first_prefix_typed_time="2023-09-25 00:00:00",
            search_time="2023-09-25 00:00:01",
        ),
        record(3, search_time="2023-09-25 00:00:00"),
        record(4, search_time="2023-09-24 11:59:59"),
        record(5, first_prefix_typed_time="invalid"),
    ]

    partition = partition_record(records)

    assert [row["query_id"] for row in partition.train] == [1]
    assert [row["query_id"] for row in partition.development] == [2]
    assert [reason for _, reason in partition.quarantine] == [
        "crosses_cutoff",
        "reversed_time",
        "invalid_timestamp",
    ]
    assert len(make_development_cases(partition, seed=42)) == 1


def test_duplicate_interaction_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicated"):
        partition_record([record(), record()])


def test_prefix_selection_is_repeatable_and_independent_of_the_answer():
    original = record()
    changed = record(final_search_term="different answer")
    selected = choose_development_prefix(original, seed=42)

    assert choose_development_prefix(original, seed=42) == selected
    assert choose_development_prefix(changed, seed=42) == selected
    assert selected[1] == original["prefixes"][selected[0]]


def test_empty_prefix_history_is_rejected():
    with pytest.raises(ValueError, match="no prefixes"):
        choose_development_prefix(record(prefixes=[]), seed=42)


def test_case_hash_is_order_independent_and_saved_cases_are_protected(tmp_path):
    first = DevelopmentCase(1, 0, "ca", "ca", "cat", "cat")
    second = DevelopmentCase(2, 0, "ca", "ca", "car", "car")
    path = tmp_path / "cases.json"

    digest = write_development_cases(path, [second, first])
    original_bytes = path.read_bytes()

    assert digest == development_cases_digest([first, second])
    assert write_development_cases(path, [first, second]) == digest

    with pytest.raises(FileExistsError):
        write_development_cases(path, [replace(first, raw_prefix="c"), second])

    assert path.read_bytes() == original_bytes
