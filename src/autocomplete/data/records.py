import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from autocomplete.data.normalization import normalize_prefix, normalize_term
from datasets import Dataset

CUTOFF = datetime.fromisoformat("2023-09-25 00:00:00")

QUERY_ID: Final = "query_id"
SESSION_ID: Final = "session_id"
PREFIXES: Final = "prefixes"
FIRST_PREFIX_TYPED_TIME: Final = "first_prefix_typed_time"
FINAL_SEARCH_TERM: Final = "final_search_term"
SEARCH_TIME: Final = "search_time"
POPULARITY: Final = "popularity"

REQUIRED_TRAIN_FIELDS: Final = (
    QUERY_ID,
    SESSION_ID,
    PREFIXES,
    FIRST_PREFIX_TYPED_TIME,
    FINAL_SEARCH_TERM,
    SEARCH_TIME,
    POPULARITY,
)

TRAIN_FIELD_TYPE: Final = {
    QUERY_ID: int,
    SESSION_ID: str,
    PREFIXES: list,
    FIRST_PREFIX_TYPED_TIME: str,
    FINAL_SEARCH_TERM: str,
    SEARCH_TIME: str,
    POPULARITY: int,
}


@dataclass
class PartitionResult:
    train: Dataset | list[dict[str, object]]
    development: Dataset | list[dict[str, object]]
    quarantine: list[tuple[dict[str, object], str]]


@dataclass(frozen=True)
class DevelopmentCase:
    query_id: int
    prefix_index: int
    raw_prefix: str
    normalized_prefix: str
    raw_final_search_term: str
    normalized_final_search_term: str


def validate_train_record(record: dict[str, object]) -> None:
    """Check required source fields and their types.

    Args:
        record: One September interaction.
    """

    missing_keys = [key for key in REQUIRED_TRAIN_FIELDS if key not in record]

    if missing_keys:
        raise ValueError(
            f"record is missing required fields: {', '.join(missing_keys)}"
        )

    for field, expected_type in TRAIN_FIELD_TYPE.items():
        value = record[field]

        if type(value) is not expected_type:
            raise TypeError(
                f"{field} expected {expected_type.__name__}, "
                f"received {type(value).__name__}"
            )


def partition_record(records: list[dict[str, object]]) -> PartitionResult:
    """Split interactions at the cutoff; quarantine invalid or crossing times.

    Args:
        records: Source interactions with unique query IDs.
    Returns:
        Training, development and quarantined records.
    """
    result = PartitionResult(train=[], development=[], quarantine=[])
    seen_ids = set()

    for record in records:
        validate_train_record(record)

        query_id = record[QUERY_ID]

        if query_id in seen_ids:
            raise ValueError(f"Query ID: {query_id} duplicated!")

        seen_ids.add(query_id)

        try:
            first_prefix_time = datetime.fromisoformat(record[FIRST_PREFIX_TYPED_TIME])
            search_time = datetime.fromisoformat(record[SEARCH_TIME])
        except (TypeError, ValueError):
            result.quarantine.append((record, "invalid_timestamp"))
            continue

        if search_time < first_prefix_time:
            result.quarantine.append((record, "reversed_time"))
        elif first_prefix_time < CUTOFF <= search_time:
            result.quarantine.append((record, "crosses_cutoff"))
        elif search_time < CUTOFF:
            result.train.append(record)
        else:
            result.development.append(record)

    return result


def choose_development_prefix(
    record: dict[str, object],
    seed: int,
) -> tuple[int, str]:
    """Choose an observed prefix using seed and query ID, never the answer.

    Args:
        record: Interaction containing observed prefixes.
        seed: Fixed sampling seed.
    Returns:
        Selected position and unchanged prefix text.
    """
    rng = random.Random(f"{seed}:{record[QUERY_ID]}")

    prefix_num = len(record[PREFIXES])

    if prefix_num == 0:
        raise ValueError(f"Record {record[QUERY_ID]} has no prefixes")

    index = rng.randrange(prefix_num)

    return index, record[PREFIXES][index]


def make_development_case(record: dict[str, object], seed: int) -> DevelopmentCase:
    """Keep raw and normalized text for one evaluation request.

    Args:
        record: Development interaction.
        seed: Fixed prefix-selection seed.
    Returns:
        One development case.
    """
    prefix_index, raw_prefix = choose_development_prefix(record, seed)
    normalized_prefix = normalize_prefix(raw_prefix)
    raw_final_search_term = record[FINAL_SEARCH_TERM]
    normalized_final_search_term = normalize_term(raw_final_search_term)

    return DevelopmentCase(
        record[QUERY_ID],
        prefix_index,
        raw_prefix,
        normalized_prefix,
        raw_final_search_term,
        normalized_final_search_term,
    )


def make_development_cases(
    partition_result: PartitionResult,
    seed: int,
) -> list[DevelopmentCase]:
    """Build one case per development interaction.

    Args:
        partition_result: Chronologically split records.
        seed: Fixed prefix-selection seed.
    Returns:
        Development cases in interaction order.
    """
    cases = []

    for record in partition_result.development:
        cases.append(make_development_case(record, seed))

    return cases


def _canonical_cases_bytes(
    cases: list[DevelopmentCase],
) -> bytes:
    """Sort cases and encode them consistently for hashing.

    Args:
        cases: Cases with unique query IDs.
    Returns:
        Compact UTF-8 JSON bytes ordered by query ID.
    """
    sorted_cases = sorted(cases, key=lambda x: x.query_id)

    case_data = [asdict(case) for case in sorted_cases]

    return json.dumps(
        case_data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def development_cases_digest(
    cases: list[DevelopmentCase],
) -> str:
    """Fingerprint every saved case field, independent of input order.

    Args:
        cases: Development cases.
    Returns:
        SHA-256 text.
    """
    payload = _canonical_cases_bytes(cases)

    return hashlib.sha256(payload).hexdigest()


def write_development_cases(
    path: Path,
    cases: list[DevelopmentCase],
) -> str:
    """Save cases or reuse an identical existing file.

    Args:
        path: Output file.
        cases: Development cases.
    Returns:
        Fingerprint of the saved bytes.
    """
    payload = _canonical_cases_bytes(cases)
    digest = hashlib.sha256(payload).hexdigest()

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite different development cases: {path}"
            )

        return digest

    path.write_bytes(payload)

    return digest


if __name__ == "__main__":
    sample = {
        "session_id": "toy",
        "prefixes": ["c", "ca"],
        "final_search_term": "cat",
        "popularity": 1,
    }
    times = [
        ("2023-09-24 12:00:00", "2023-09-24 12:00:01"),
        ("2023-09-25 12:00:00", "2023-09-25 12:00:01"),
        ("2023-09-24 23:59:59", "2023-09-25 00:00:01"),
    ]
    records = [
        dict(sample, query_id=i, first_prefix_typed_time=start, search_time=end)
        for i, (start, end) in enumerate(times, start=1)
    ]
    partition = partition_record(records)
    print("Cutoff:", CUTOFF)
    print("Training IDs:", [row["query_id"] for row in partition.train])
    print("Development IDs:", [row["query_id"] for row in partition.development])
    print(
        "Quarantine:",
        [(row["query_id"], reason) for row, reason in partition.quarantine],
    )
    print("Chosen development case:", make_development_cases(partition, seed=42)[0])
