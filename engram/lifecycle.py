from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Sequence

from .config import GovernanceRule, RetentionRule, SummarizationRule
from .models import MemorySignal
from .schema import MemoryRecord, new_memory_id, parse_datetime, utcnow


def _to_datetime(value: datetime | str | None) -> datetime:
    return parse_datetime(value) or utcnow()


def _matches_selector(record: MemoryRecord, selector: dict[str, Any]) -> bool:
    for key, expected in selector.items():
        actual = getattr(record, key, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _content_matches(content: str, patterns: Sequence[str]) -> bool:
    lowered = content.lower()
    return all(pattern.lower() in lowered for pattern in patterns)


def _safe_formula(formula: str, signal: MemorySignal) -> float:
    allowed_names = {
        "retry_count": signal.retry_count,
        "content_length": len(signal.content),
    }
    try:
        value = eval(formula, {"__builtins__": {}}, allowed_names)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, float(value)))


def build_memory_record(
    *,
    type: str,
    scope: str,
    content: str,
    source: str,
    signal: MemorySignal,
    create: dict[str, Any],
    derived_from: list[str] | None = None,
    provenance: list[str] | None = None,
    timestamp: datetime | None = None,
) -> MemoryRecord:
    now = _to_datetime(timestamp or signal.timestamp)
    importance = create.get("importance_score")
    if importance is None and "importance_score_formula" in create:
        importance = _safe_formula(str(create["importance_score_formula"]), signal)
    payload = dict(create.get("payload") or {})
    payload.update(signal.metadata.get("payload", {}))
    return MemoryRecord(
        id=new_memory_id(),
        type=type,
        scope=scope,
        content=create.get("content", content),
        agent_id=create.get("agent_id", signal.agent_id),
        user_id=create.get("user_id", signal.user_id),
        session_id=create.get("session_id", signal.session_id),
        payload=payload or None,
        timestamp_created=now,
        timestamp_updated=now,
        valid_from=_to_datetime(create["valid_from"]) if create.get("valid_from") else None,
        valid_to=_to_datetime(create["valid_to"]) if create.get("valid_to") else None,
        source=create.get("source", source),
        provenance=provenance or [str(signal.metadata.get("event_id", signal.type))],
        derived_from=derived_from or [],
        importance_score=float(importance if importance is not None else 0.5),
        sensitivity_level=create.get("sensitivity_level", "public"),
        retention_policy=create.get("retention_policy"),
        access_roles=list(create.get("access_roles", [])),
        embedding_model=create.get("embedding_model"),
    )


def extract_records(
    rules: Iterable[Any],
    signal: MemorySignal,
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    extracted: list[MemoryRecord] = []
    timestamp = _to_datetime(now or signal.timestamp)
    for rule in rules:
        if rule.trigger != signal.type:
            continue
        conditions = rule.conditions or []
        matches = True
        for condition in conditions:
            if "content_matches" in condition and not _content_matches(signal.content, condition["content_matches"]):
                matches = False
                break
        if not matches:
            continue
        create = dict(rule.create)
        extracted.append(
            build_memory_record(
                type=create.get("type", "episodic"),
                scope=create.get("scope", "session"),
                content=signal.content,
                source=signal.source,
                signal=signal,
                create=create,
                timestamp=timestamp,
            )
        )
    return extracted


def compute_decay_factor(
    record: MemoryRecord,
    rules: Iterable[RetentionRule],
    *,
    now: datetime | None = None,
) -> float:
    current = _to_datetime(now)
    applicable = [rule for rule in rules if _matches_selector(record, rule.applies_to)]
    if not applicable:
        return record.decay_factor
    age_days = max((current - _to_datetime(record.timestamp_created)).total_seconds() / 86400.0, 0.0)
    factors: list[float] = []
    for rule in applicable:
        if rule.decay_function == "none" or rule.half_life_days is None:
            factors.append(1.0)
            continue
        half_life = max(float(rule.half_life_days), 0.001)
        if rule.decay_function == "exponential":
            factor = 0.5 ** (age_days / half_life)
        elif rule.decay_function == "linear":
            factor = max(0.0, 1.0 - (age_days / half_life))
        elif rule.decay_function == "step":
            factor = 0.5 if age_days >= half_life else 1.0
        else:
            factor = 1.0
        factors.append(max(rule.floor_score, factor))
    return min(factors) if factors else record.decay_factor


def apply_governance_deletion(
    records: Iterable[MemoryRecord],
    rules: Iterable[GovernanceRule],
    *,
    user_id: str | None = None,
    reason: str = "user_deletion",
    now: datetime | None = None,
) -> tuple[list[MemoryRecord], int]:
    current = _to_datetime(now)
    kept: list[MemoryRecord] = []
    deleted = 0
    for record in records:
        should_delete = False
        for rule in rules:
            if not _matches_selector(record, rule.applies_to):
                continue
            if user_id is not None and record.user_id != user_id:
                continue
            if reason == "user_deletion" and rule.on_user_deletion == "delete_all":
                should_delete = True
            elif rule.retention_days is not None:
                age_days = (current - _to_datetime(record.timestamp_created)).total_seconds() / 86400.0
                if age_days >= rule.retention_days:
                    should_delete = True
            if should_delete:
                deleted += 1
                break
        if not should_delete:
            kept.append(record)
    return kept, deleted


def promote_repeated_success(
    records: Iterable[MemoryRecord],
    rules: Iterable[SummarizationRule],
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    current = _to_datetime(now)
    promoted: list[MemoryRecord] = []
    by_signature: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        if record.type != "episodic":
            continue
        payload = record.payload or {}
        signature = payload.get("action_signature") or payload.get("action") or record.content
        if payload.get("success") is True or payload.get("outcome") == "success":
            by_signature[str(signature)].append(record)

    for rule in rules:
        if not rule.trigger.startswith("same_action_success_count >="):
            continue
        threshold = int(rule.trigger.split(">=", 1)[1].strip())
        for signature, group in by_signature.items():
            if len(group) < threshold or not _matches_selector(group[0], rule.applies_to):
                continue
            source_payload = dict(group[0].payload or {})
            source_payload["action_signature"] = signature
            source_payload["source_record_ids"] = [item.id for item in group]
            promoted.append(
                MemoryRecord(
                    id=new_memory_id(),
                    type=rule.promote_to.get("type", "procedural"),
                    scope=rule.promote_to.get("scope", "agent"),
                    content=rule.promote_to.get("content") or group[0].content,
                    agent_id=group[0].agent_id,
                    user_id=group[0].user_id,
                    session_id=group[0].session_id,
                    payload=source_payload,
                    timestamp_created=current,
                    timestamp_updated=current,
                    source="reflection",
                    provenance=[item.id for item in group],
                    derived_from=[item.id for item in group],
                    importance_score=max(item.importance_score for item in group),
                    sensitivity_level=group[0].sensitivity_level,
                    retention_policy=group[0].retention_policy,
                    access_roles=list(group[0].access_roles),
                )
            )
    return promoted
