"""Tests for qwen_ft.data — dataset validation and splitting."""

import json
from pathlib import Path

import pytest

from qwen_ft.data import (
    DataValidationError,
    split_dataset,
    validate_jsonl_file,
    validate_messages,
    validate_record,
    write_jsonl,
    read_jsonl,
)


def m(role, content):
    return {"role": role, "content": content}


# ---------------------------------------------------------------- happy path

def test_sample_dataset_validates(sample_dataset: Path):
    report = validate_jsonl_file(sample_dataset)
    assert report["records"] >= 20
    assert report["min_turns"] >= 2


def test_sample_dataset_is_utf8_khmer(sample_dataset: Path):
    text = sample_dataset.read_text(encoding="utf-8")
    # Khmer Unicode block U+1780–U+17FF must appear (course dataset is Khmer)
    assert any("\u1780" <= ch <= "\u17FF" for ch in text)


def test_valid_conversations():
    validate_messages([m("user", "hi"), m("assistant", "hello")])
    validate_messages([
        m("system", "you are helpful"),
        m("user", "hi"),
        m("assistant", "hello"),
    ])


# ------------------------------------------------------------- invalid cases

def test_empty_messages():
    with pytest.raises(DataValidationError, match="not be empty"):
        validate_messages([])


def test_bad_role():
    with pytest.raises(DataValidationError, match="invalid role"):
        validate_messages([m("bot", "hi")])


def test_empty_content():
    with pytest.raises(DataValidationError, match="empty or non-string"):
        validate_messages([m("user", "   "), m("assistant", "ok")])


def test_missing_assistant():
    with pytest.raises(DataValidationError, match="no 'assistant' turn"):
        validate_messages([m("user", "hi"), m("user", "hello?")], strict_alternation=False)


def test_not_ending_with_assistant():
    with pytest.raises(DataValidationError, match="must end with an 'assistant'"):
        validate_messages([m("assistant", "hi"), m("user", "hello")])


def test_double_user_turn():
    with pytest.raises(DataValidationError, match="consecutive"):
        validate_messages([
            m("user", "a"), m("user", "b"), m("assistant", "c"),
        ])


def test_double_assistant_turn():
    with pytest.raises(DataValidationError, match="consecutive"):
        validate_messages([
            m("user", "a"), m("assistant", "b"), m("assistant", "c"),
        ])


def test_system_not_first():
    with pytest.raises(DataValidationError, match="must come first"):
        validate_messages([m("user", "a"), m("system", "s"), m("assistant", "b")])


def test_two_system_messages():
    with pytest.raises(DataValidationError, match="at most one"):
        validate_messages([
            m("system", "a"), m("system", "b"), m("user", "u"), m("assistant", "x"),
        ])


def test_extra_message_keys():
    with pytest.raises(DataValidationError, match="unsupported key"):
        validate_messages([
            {"role": "user", "content": "hi", "name": "bob"},
            m("assistant", "hello"),
        ])


def test_tool_without_assistant():
    with pytest.raises(DataValidationError, match="must follow an 'assistant'"):
        validate_messages([m("user", "a"), m("tool", "result"), m("assistant", "b")])


def test_record_missing_messages_gives_format_hint():
    with pytest.raises(DataValidationError, match="messages"):
        validate_record({"text": "hello"})


def test_prompt_completion_format_hint():
    with pytest.raises(DataValidationError, match="prompt/completion"):
        validate_record({"prompt": "a", "completion": "b"})


def test_record_extra_top_level_key():
    with pytest.raises(DataValidationError, match="unsupported top-level key"):
        validate_record({"messages": [m("user", "a"), m("assistant", "b")], "label": 1})


# ------------------------------------------------------------------- files

def test_jsonl_roundtrip(tmp_path):
    records = [
        {"messages": [m("user", "សួស្តី"), m("assistant", "សួស្តី! ជំរាបសួរ")]},
        {"messages": [m("user", "1+1?"), m("assistant", "2")]},
    ]
    p = tmp_path / "ds.jsonl"
    write_jsonl(records, p)
    assert read_jsonl(p) == records


def test_bad_json_line_reports_line_number(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text('{"messages": []}\nNOT-JSON\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="line 2"):
        validate_jsonl_file(p)


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no records"):
        validate_jsonl_file(p)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        validate_jsonl_file(tmp_path / "nope.jsonl")


# ------------------------------------------------------------------ splits

def test_split_deterministic(sample_dataset: Path):
    records = read_jsonl(sample_dataset)
    a1, b1 = split_dataset(records, eval_ratio=0.1, seed=42)
    a2, b2 = split_dataset(records, eval_ratio=0.1, seed=42)
    assert a1 == a2 and b1 == b2
    assert len(a1) + len(b1) == len(records)


def test_split_ratio(sample_dataset: Path):
    records = read_jsonl(sample_dataset)
    train, ev = split_dataset(records, eval_ratio=0.1, seed=7)
    assert len(ev) == round(len(records) * 0.1)
    assert len(train) == len(records) - len(ev)


def test_split_disjoint(sample_dataset: Path):
    records = read_jsonl(sample_dataset)
    train, ev = split_dataset(records, eval_ratio=0.2, seed=3)
    train_s = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in train}
    ev_s = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in ev}
    assert train_s.isdisjoint(ev_s)


def test_split_bad_ratio():
    with pytest.raises(DataValidationError, match="eval_ratio"):
        split_dataset([{"messages": [m("user", "a"), m("assistant", "b")]}], eval_ratio=0.9)


def test_split_empty():
    with pytest.raises(DataValidationError, match="empty dataset"):
        split_dataset([], 0.1)
