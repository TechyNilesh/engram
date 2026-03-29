from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..observability import ScoredMemory
from ..schema import MemoryRecord, parse_datetime

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in WORD_RE.findall(text):
        token = raw.lower()
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _normalize_datetime(value: Any) -> datetime | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _matches_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        if isinstance(actual, list):
            return any(item in expected for item in actual)
        return actual in expected
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def record_matches_filters(record: MemoryRecord, filters: dict[str, Any] | None) -> bool:
    for key, expected in (filters or {}).items():
        if expected is None or key in {"query", "top_k"}:
            continue
        if key == "valid_before":
            valid_before = _normalize_datetime(expected)
            if record.valid_from and valid_before and record.valid_from > valid_before:
                return False
            continue
        if key == "valid_after":
            valid_after = _normalize_datetime(expected)
            if record.valid_to and valid_after and record.valid_to < valid_after:
                return False
            continue
        actual = getattr(record, key, None)
        if not _matches_value(actual, expected):
            return False
    return True


def score_record(record: MemoryRecord, query: str) -> tuple[float, float, float, list[str]]:
    query_terms = tokenize(query)
    content_terms = tokenize(record.content)
    if record.payload:
        content_terms |= tokenize(str(record.payload))
    matched = sorted(query_terms & content_terms)
    lexical = len(matched) / max(len(query_terms), 1) if query_terms else 0.0
    importance = max(0.0, min(1.0, record.importance_score))
    freshness = max(0.0, min(1.0, record.decay_factor))
    access_boost = 1.0 - math.exp(-min(record.access_count, 20) / 5.0)
    if not query_terms:
        score = 0.55 * importance + 0.35 * freshness + 0.10 * access_boost
    else:
        score = 0.58 * lexical + 0.20 * importance + 0.17 * freshness + 0.05 * access_boost
    return max(0.0, min(1.0, score)), lexical, freshness, matched


def _policy_filters(record: MemoryRecord, filters: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    for key in ("type", "scope", "user_id", "agent_id", "session_id", "sensitivity_level"):
        value = (filters or {}).get(key)
        if value is not None and getattr(record, key) == value:
            labels.append(f"{key}={value}")
    return labels


def score_records(records: list[MemoryRecord], query: str, filters: dict[str, Any] | None) -> list[ScoredMemory]:
    scored: list[ScoredMemory] = []
    for record in records:
        if not record_matches_filters(record, filters):
            continue
        score, lexical, freshness, matched = score_record(record, query)
        if query.strip() and score <= 0 and not matched:
            continue
        scored.append(
            ScoredMemory(
                record=record,
                score=score,
                lexical_score=lexical,
                importance_component=max(0.0, min(1.0, record.importance_score)),
                decay_component=freshness,
                matched_terms=matched,
                policy_filters=_policy_filters(record, filters),
            )
        )
    scored.sort(key=lambda item: (item.score, item.record.timestamp_updated), reverse=True)
    return scored


def decay_record(
    record: MemoryRecord,
    days: float,
    *,
    half_life_days: float = 7.0,
    floor_score: float = 0.1,
) -> MemoryRecord:
    if days <= 0:
        return record
    half_life_days = max(half_life_days, 0.001)
    decay = 0.5 ** (days / half_life_days)
    return replace(record, decay_factor=max(floor_score, min(1.0, record.decay_factor * decay)))
