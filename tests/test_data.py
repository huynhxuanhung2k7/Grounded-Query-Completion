import pytest

from autocomplete.data import (
    PartitionResult,
    choose_development_prefix,
    make_development_cases,
    partition_record,
    validate_train_record,
)
from autocomplete.normalization import (
    normalize_prefix,
    normalize_term,
)

VALID_TRAIN_RECORD = {
    "query_id": 99,
    "session_id": "test-session-001",
    "prefixes": [
        "w",
        "wr",
        "wri",
        "wrin",
        "wrink",
        "wrinkl",
        "wrinkle",
        "wrinkle ",
        "wrinkle p",
        "wrinkle pa",
    ],
    "first_prefix_typed_time": "2023-09-01 10:00:00",
    "final_search_term": "wrinkle patch",
    "search_time": "2023-09-01 10:00:10",
    "popularity": 1,
}


def make_record(**changes: object) -> dict[str, object]:
    record = VALID_TRAIN_RECORD.copy()
    record.update(changes)
    return record


def test_missing_query_id() -> None:
    record = VALID_TRAIN_RECORD.copy()
    del record["query_id"]

    with pytest.raises(ValueError, match="missing required fields: query_id"):
        validate_train_record(record)


def test_wrong_type_query_id() -> None:
    record = VALID_TRAIN_RECORD.copy()
    record["query_id"] = "99"
    with pytest.raises(TypeError, match="query_id expected int, received str"):
        validate_train_record(record)


def test_valid_query_id() -> None:
    assert validate_train_record(VALID_TRAIN_RECORD) is None


def test_partition_record_sorts_each_time_case() -> None:
    result = partition_record(
        [
            make_record(
                query_id=1,
                first_prefix_typed_time="2023-09-24 23:00:00",
                search_time="2023-09-24 23:00:10",
            ),
            make_record(
                query_id=2,
                first_prefix_typed_time="2023-09-25 00:00:00",
                search_time="2023-09-25 00:00:10",
            ),
            make_record(
                query_id=3,
                first_prefix_typed_time="2023-09-24 23:59:59",
                search_time="2023-09-25 00:00:00",
            ),
            make_record(
                query_id=4,
                first_prefix_typed_time="2023-09-25 00:00:10",
                search_time="2023-09-25 00:00:00",
            ),
            make_record(
                query_id=5,
                first_prefix_typed_time="not a timestamp",
            ),
        ]
    )

    assert [record["query_id"] for record in result.train] == [1]
    assert [record["query_id"] for record in result.development] == [2]
    assert [(record["query_id"], reason) for record, reason in result.quarantine] == [
        (3, "crosses_cutoff"),
        (4, "reversed_time"),
        (5, "invalid_timestamp"),
    ]


def test_partition_record_rejects_duplicate_query_id() -> None:
    records = [make_record(query_id=1), make_record(query_id=1)]

    with pytest.raises(ValueError, match="duplicated"):
        partition_record(records)


def test_choose_development_prefix_is_repeatable() -> None:
    record = make_record()

    index, prefix = choose_development_prefix(record, seed=42)

    assert (index, prefix) == choose_development_prefix(record, seed=42)
    assert prefix == record["prefixes"][index]


def test_choose_development_prefix_normal_expectation() -> None:
    record = make_record(
        query_id=1, prefixes=["w", "wi", "wire", "wire;;;;", "wireless"]
    )
    index, prefix = choose_development_prefix(record, seed=42)

    assert record["prefixes"][index] in record["prefixes"]
    assert prefix in record["prefixes"]
    assert index < len(record["prefixes"])


def test_choose_development_prefix_edge_cases() -> None:
    record = make_record(query_id=1, prefixes=[])

    with pytest.raises(ValueError, match="has no prefixes"):
        choose_development_prefix(record, seed=42)


def test_prefix_choice_does_not_use_final_answer() -> None:
    original = make_record(
        query_id=123,
        prefixes=["w", "wi", "wire"],
        final_search_term="wireless mouse",
    )
    changed = {
        **original,
        "final_search_term": "completely different answer",
    }

    index_original, prefix_original = choose_development_prefix(original, 42)
    index_change, prefix_change = choose_development_prefix(changed, 42)

    assert (index_original, prefix_original) == (index_change, prefix_change)


def test_prefix_choice_is_independent_of_record_order() -> None:
    records = [
        make_record(query_id=101, prefixes=["w", "wi", "wire"]),
        make_record(query_id=202, prefixes=["c", "ca", "cat"]),
    ]

    normal_results = {
        record["query_id"]: choose_development_prefix(record, 42) for record in records
    }
    reversed_results = {
        record["query_id"]: choose_development_prefix(record, 42)
        for record in reversed(records)
    }

    assert normal_results == reversed_results


def test_make_development_cases_uses_only_development_records() -> None:
    development_record = make_record(
        query_id=10,
        prefixes=["w", "wi", "wire"],
        final_search_term="wireless mouse",
    )

    partition = PartitionResult(
        train=[make_record(query_id=20)],
        development=[development_record],
        quarantine=[(make_record(query_id=30), "reversed_time")],
    )

    cases = make_development_cases(partition, seed=42)

    assert len(cases) == 1

    case = cases[0]
    expected_index, expected_prefix = choose_development_prefix(
        development_record, seed=42
    )

    assert case.query_id == 10
    assert case.prefix_index == expected_index
    assert case.raw_prefix == expected_prefix
    assert case.normalized_prefix == normalize_prefix(expected_prefix)
    assert case.raw_final_search_term == development_record["final_search_term"]
    assert case.normalized_final_search_term == normalize_term(
        development_record["final_search_term"]
    )


def test_make_development_cases_preserves_index_for_repeated_prefixes() -> None:
    record = make_record(
        query_id=123,
        prefixes=["w", "wi", "wi", "win"],
        final_search_term="winter jacket",
    )
    partition = PartitionResult(train=[], development=[record], quarantine=[])

    expected_index, expected_prefix = choose_development_prefix(record, seed=42)
    [case] = make_development_cases(partition, seed=42)

    assert case.prefix_index == expected_index
    assert case.raw_prefix == expected_prefix
    assert record["prefixes"][case.prefix_index] == case.raw_prefix


def test_make_development_cases_allows_no_development_records() -> None:
    partition = PartitionResult(
        train=[make_record(query_id=1)],
        development=[],
        quarantine=[],
    )

    assert make_development_cases(partition, seed=42) == []
