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
    """Validate only query_id for now

    Args:
        record (dict[str, object]): _description_
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
    rng = random.Random(f"{seed}:{record[QUERY_ID]}")

    prefix_num = len(record[PREFIXES])
    if prefix_num == 0:
        raise ValueError(f"Record {record[QUERY_ID]} has no prefixes")

    index = rng.randrange(prefix_num)
    return index, record[PREFIXES][index]


def make_development_case(record: dict[str, object], seed: int) -> DevelopmentCase:
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
    cases = []
    for record in partition_result.development:
        cases.append(make_development_case(record, seed))

    return cases

def _canocial_case_data(
    cases: list[DevelopmentCase],
) -> list[dict[str, object]]:
    sorted_cases = sorted(cases, key = lambda x: x.query_id)
    
    return [
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

def development_cases_digest(
    cases: list[DevelopmentCase],
) -> str:
    case_data = _canocial_case_data(cases)
    
    payload = json.dumps(
        case_data, 
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )
    
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def write_development_cases(
    path: Path,
    cases: list[DevelopmentCase],
) -> None:
    case_data = _canocial_case_data(cases)
    content_digest = development_cases_digest(cases)
    
    artifact = {
        "content_digest": content_digest,
        "cases": case_data,
    }
    
    artifact_text = json.dumps(
        artifact,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )
    
    expected_text = artifact_text + "\n"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        
        if path.read_text(encoding="utf-8") != artifact_text:
            raise FileExistsError(
            f"Refusing to overwrite different development cases: {path}"
            )

        return
    
    path.write_text(expected_text, encoding="utf-8")    