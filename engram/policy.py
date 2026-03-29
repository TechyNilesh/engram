from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

from .config import PolicyConfig, load_policy_config
from .lifecycle import (
    apply_governance_deletion,
    compute_decay_factor,
    extract_records,
    promote_repeated_success,
)
from .models import MemorySignal
from .schema import MemoryRecord, utcnow


@dataclass(slots=True)
class PolicyOutcome:
    extracted: list[MemoryRecord] = field(default_factory=list)
    promoted: list[MemoryRecord] = field(default_factory=list)
    policies_fired: list[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, config: PolicyConfig | Mapping[str, object] | str):
        self.config = config if isinstance(config, PolicyConfig) else load_policy_config(config)

    def process_event(
        self,
        event: MemorySignal | Mapping[str, object],
        *,
        now: datetime | None = None,
    ) -> PolicyOutcome:
        signal = event if isinstance(event, MemorySignal) else MemorySignal.from_mapping(dict(event))
        extracted = extract_records(self.config.extraction, signal, now=now)
        promoted = promote_repeated_success(extracted, self.config.summarization, now=now)
        policies_fired = [rule.name for rule in self.config.extraction if rule.trigger == signal.type]
        if promoted:
            policies_fired.extend(rule.name for rule in self.config.summarization)
        return PolicyOutcome(extracted=extracted, promoted=promoted, policies_fired=policies_fired)

    def apply_decay(
        self,
        record: MemoryRecord,
        *,
        now: datetime | None = None,
    ) -> float:
        return compute_decay_factor(record, self.config.retention, now=now)

    def apply_retention(
        self,
        records: Iterable[MemoryRecord],
        *,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        updated: list[MemoryRecord] = []
        for record in records:
            payload = record.to_dict()
            payload["decay_factor"] = self.apply_decay(record, now=now)
            payload["timestamp_updated"] = (now or utcnow()).isoformat()
            updated.append(MemoryRecord.from_dict(payload))
        return updated

    def apply_governance(
        self,
        records: Iterable[MemoryRecord],
        *,
        user_id: str | None = None,
        reason: str = "user_deletion",
        now: datetime | None = None,
    ) -> tuple[list[MemoryRecord], int]:
        return apply_governance_deletion(
            records,
            self.config.governance,
            user_id=user_id,
            reason=reason,
            now=now,
        )

    def promote(
        self,
        records: Iterable[MemoryRecord],
        *,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        return promote_repeated_success(records, self.config.summarization, now=now)
