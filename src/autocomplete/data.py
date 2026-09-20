from typing import Final
import unicodedata
from datetime import datetime
import random
from dataclasses import dataclass

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

TRAIN_FIELD_TYPE : Final = {
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
        
def validate_train_record(record: dict[str, object]) -> None:
    """Validate only query_id for now

    Args:
        record (dict[str, object]): _description_
    """
    missing_keys = [
        key for key in REQUIRED_TRAIN_FIELDS
        if key not in record
    ]
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
    
    result = PartitionResult(
        train=[],
        development=[],
        quarantine=[]
    )
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

    