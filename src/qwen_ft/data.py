"""Dataset utilities: JSONL (chat ``messages`` format) validation + splitting.

The expected record format matches what ``trl.SFTTrainer`` consumes for
conversational language modeling::

    {"messages": [
        {"role": "user", "content": "សួស្តី! អ្នកជាអ្វី?"},
        {"role": "assistant", "content": "ខ្ញុំជាជំនួយការ AI ..."}
    ]}

Rules enforced by :func:`validate_messages`:

* ``messages`` is a non-empty list;
* every message is ``{"role": ..., "content": ...}`` with a non-empty string
  ``content``;
* roles come from {system, user, assistant, tool};
* at most one leading ``system`` message;
* the conversation contains at least one ``assistant`` turn (otherwise there
  is nothing to learn from);
* no two ``assistant`` turns in a row and no two ``user`` turns in a row
  (strict alternation after the optional system prompt — the common case for
  SFT data; relax with ``strict_alternation=False``).

Pure standard library — safe to import anywhere.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

__all__ = [
    "DataValidationError",
    "validate_messages",
    "validate_record",
    "validate_jsonl_file",
    "split_dataset",
    "write_jsonl",
    "read_jsonl",
]

VALID_ROLES = {"system", "user", "assistant", "tool"}


class DataValidationError(ValueError):
    """Raised when a dataset record or file violates the expected schema."""

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        where = f" (line {line})" if line is not None else ""
        super().__init__(f"{message}{where}")


def validate_messages(
    messages: Sequence[Any],
    strict_alternation: bool = True,
) -> None:
    """Validate one conversation. Raises :class:`DataValidationError`."""
    if not isinstance(messages, list):
        raise DataValidationError("'messages' must be a list")
    if len(messages) == 0:
        raise DataValidationError("'messages' must not be empty")

    roles: List[str] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise DataValidationError(f"message #{i} must be an object, got {type(msg).__name__}")
        extra = set(msg) - {"role", "content"}
        if extra:
            raise DataValidationError(
                f"message #{i} has unsupported key(s) {sorted(extra)}; "
                "only 'role' and 'content' are allowed"
            )
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise DataValidationError(
                f"message #{i} has invalid role {role!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if not isinstance(content, str) or not content.strip():
            raise DataValidationError(f"message #{i} ({role}) has empty or non-string 'content'")
        roles.append(role)

    if roles.count("system") > 1:
        raise DataValidationError("at most one 'system' message is allowed")
    if "system" in roles and roles.index("system") != 0:
        raise DataValidationError("the 'system' message must come first")
    if "assistant" not in roles:
        raise DataValidationError(
            "conversation has no 'assistant' turn — nothing for SFT to learn"
        )
    if roles[-1] != "assistant":
        raise DataValidationError(
            f"conversation must end with an 'assistant' turn (ends with {roles[-1]!r})"
        )

    if strict_alternation:
        prev_non_system: str | None = None
        for idx, role in enumerate(roles):
            if role == "system":
                continue
            if role == "tool":
                # tool responses follow an assistant turn
                if prev_non_system != "assistant":
                    raise DataValidationError(
                        f"message #{idx}: 'tool' message must follow an 'assistant' message"
                    )
                prev_non_system = "user"  # next expected speaker flips back
                continue
            if prev_non_system is not None and role == prev_non_system:
                raise DataValidationError(
                    f"message #{idx}: two consecutive {role!r} turns "
                    "(strict user/assistant alternation expected)"
                )
            prev_non_system = role


def validate_record(record: Any, line: int | None = None) -> Dict[str, Any]:
    """Validate a single JSONL record (dict with a 'messages' key)."""
    if not isinstance(record, dict):
        raise DataValidationError("record must be a JSON object", line=line)
    if "messages" not in record:
        if "prompt" in record and "completion" in record:
            raise DataValidationError(
                "prompt/completion format detected — this course uses the "
                "'messages' conversational format; convert it first "
                "(see docs/chapters/ch04-datasets.md)",
                line=line,
            )
        if "text" in record:
            raise DataValidationError(
                "plain-text format detected — this course uses the 'messages' "
                "conversational format (see docs/chapters/ch04-datasets.md)",
                line=line,
            )
        raise DataValidationError("record has no 'messages' key", line=line)
    extra = set(record) - {"messages", "meta"}
    if extra:
        raise DataValidationError(
            f"unsupported top-level key(s) {sorted(extra)}; allowed: messages, meta",
            line=line,
        )
    try:
        validate_messages(record["messages"])
    except DataValidationError as exc:
        raise DataValidationError(str(exc), line=line) from None
    return record


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"dataset file not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"invalid JSON: {exc.msg}", line=lineno) from None
    return records


def write_jsonl(records: Sequence[Dict[str, Any]], path: Union[str, Path]) -> Path:
    """Write records to a JSONL file (UTF-8, one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def validate_jsonl_file(
    path: Union[str, Path],
    max_records: int | None = None,
) -> Dict[str, Any]:
    """Validate a whole JSONL dataset file.

    Returns a small report dict (counts, warnings). Raises
    :class:`DataValidationError` on the first invalid record.
    """
    records = read_jsonl(path)
    if not records:
        raise DataValidationError(f"{path}: file contains no records")
    n = len(records) if max_records is None else min(len(records), max_records)
    for i, rec in enumerate(records[:n], start=1):
        validate_record(rec, line=i)

    turns = [len(r["messages"]) for r in records[:n]]
    chars = [
        sum(len(m["content"]) for m in r["messages"]) for r in records[:n]
    ]
    report = {
        "path": str(path),
        "records": len(records),
        "validated": n,
        "min_turns": min(turns),
        "max_turns": max(turns),
        "avg_chars": round(sum(chars) / len(chars), 1),
        "max_chars": max(chars),
    }
    if max(chars) > 24000:
        report["warnings"] = [
            "some conversations exceed ~24k characters; they will be truncated "
            "at the configured max_length — consider splitting long dialogues"
        ]
    return report


def split_dataset(
    records: Sequence[Dict[str, Any]],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic shuffled train/eval split."""
    if not 0.0 <= eval_ratio < 0.5:
        raise DataValidationError(f"eval_ratio must be in [0, 0.5), got {eval_ratio}")
    if len(records) == 0:
        raise DataValidationError("cannot split an empty dataset")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(round(len(shuffled) * eval_ratio))
    if eval_ratio > 0 and n_eval == 0 and len(shuffled) >= 2:
        n_eval = 1  # always keep at least one eval example when a ratio is set
    return shuffled[n_eval:], shuffled[:n_eval]
