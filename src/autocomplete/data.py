import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from autocomplete.normalization import normalize_prefix, normalize_term
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
    """Check required training fields and their exact outer Python types.

    This checks that prefixes is a list, but does not validate its elements,
    parse timestamps, or check field values.

    Args:
        record: One raw AmazonQAC training interaction.

    Raises:
        ValueError: A required field is missing.
        TypeError: A field does not have its declared Python type.
    """
    # Check presence before accessing field values.
    missing_keys = [key for key in REQUIRED_TRAIN_FIELDS if key not in record]
    if missing_keys:
        raise ValueError(
            f"record is missing required fields: {', '.join(missing_keys)}"
        )

    # Require exact types without silently converting source values.
    for field, expected_type in TRAIN_FIELD_TYPE.items():
        value = record[field]

        if type(value) is not expected_type:
            raise TypeError(
                f"{field} expected {expected_type.__name__}, "
                f"received {type(value).__name__}"
            )


def partition_record(records: list[dict[str, object]]) -> PartitionResult:
    """Split complete interactions around CUTOFF and retain quarantine reasons.

    Interactions ending before the cutoff train; those starting at or after
    it develop. Invalid timestamps, reversed times, and cutoff crossings are
    quarantined. Records are preserved without modification.

    Args:
        records: Raw training interactions with unique query IDs.

    Raises:
        ValueError: A required field is missing or a query ID is duplicated.
        TypeError: A required field has the wrong Python type.

    Returns:
        Training and development records, plus (record, reason) quarantine pairs.
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
    """Choose an observed prefix position using the seed and query ID.

    Selection is independent of the final search term and record-processing
    order. Repeated prefix values remain separate eligible positions.

    Args:
        record: A valid interaction containing a query ID and observed prefixes.
        seed: Fixed seed used to reproduce this interaction's selection.

    Raises:
        ValueError: The observed prefix list is empty.

    Returns:
        The zero-based position and its unchanged raw prefix string.
    """
    rng = random.Random(f"{seed}:{record[QUERY_ID]}")

    prefix_num = len(record[PREFIXES])
    if prefix_num == 0:
        raise ValueError(f"Record {record[QUERY_ID]} has no prefixes")

    index = rng.randrange(prefix_num)
    return index, record[PREFIXES][index]


def make_development_case(record: dict[str, object], seed: int) -> DevelopmentCase:
    """Create an immutable evaluation case from one development interaction.

    Preserve the selected position and raw strings for auditing, alongside
    normalized prefix and final-term strings for matching and evaluation.

    Args:
        record: A valid interaction from the development partition.
        seed: Fixed seed passed to the observed-prefix chooser.

    Raises:
        ValueError: The observed prefix list is empty.

    Returns:
        A case containing the query ID, selected position, and raw/normalized text.
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
    """Build one case per development interaction, preserving input order.

    Training and quarantined records are excluded. An empty development
    partition produces an empty list.

    Args:
        partition_result: The result of splitting complete interactions.
        seed: Fixed seed used for each interaction's prefix selection.

    Raises:
        ValueError: A development interaction has an empty prefix list.

    Returns:
        Development cases with raw and normalized text.
    """
    cases = []
    for record in partition_result.development:
        cases.append(make_development_case(record, seed))

    return cases

def _canonical_cases_bytes(
    cases: list[DevelopmentCase],
) -> bytes:
    """Serialize all six case fields as canonical UTF-8 JSON bytes.

    Sort cases by query ID and dictionary keys alphabetically, retain Unicode
    characters, and omit formatting spaces and a trailing newline. Hashing and
    writing share this exact byte representation.

    Args:
        cases: Development cases with unique query IDs, in any order.

    Returns:
        Compact JSON bytes ordered by query ID, without modifying the input list.
    """
    sorted_cases = sorted(cases, key = lambda x: x.query_id)
    
    case_data = [
        {
            "query_id": case.query_id,
            "prefix_index": case.prefix_index,
            "raw_prefix": case.raw_prefix,
            "normalized_prefix": case.normalized_prefix,
            "raw_final_search_term": case.raw_final_search_term,
            "normalized_final_search_term": case.normalized_final_search_term,
        }
        for case in sorted_cases
    ]
    
    return json.dumps(
        case_data, 
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")

def development_cases_digest(
    cases: list[DevelopmentCase],
) -> str:
    """Compute the SHA-256 fingerprint of the canonical case bytes.

    Args:
        cases: Development cases with unique query IDs, in any order.

    Returns:
        A lowercase, 64-character hexadecimal digest. Reordering cases does not
        change it; every serialized field contributes to the fingerprint.
    """
    payload = _canonical_cases_bytes(cases)
    return hashlib.sha256(payload).hexdigest()

def write_development_cases(
    path: Path,
    cases: list[DevelopmentCase],
) -> str:
    """Write canonical case bytes, reusing an identical existing file.

    Create parent directories as needed. If the destination already contains
    identical bytes, return its digest without rewriting it.

    Args:
        path: Destination JSON file.
        cases: Development cases with unique query IDs, in any order.

    Raises:
        FileExistsError: The destination exists with different bytes.

    Returns:
        The SHA-256 digest of the file bytes, matching development_cases_digest.
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
