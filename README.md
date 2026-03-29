# Engram

Framework-agnostic agent memory for Python: canonical memory records, policy-driven lifecycle rules, pluggable storage, and retrieval observability.

## Features

- Canonical `MemoryRecord` schema for episodic, semantic, procedural, and meta memory
- Async-first `MemoryClient` with simple sync wrappers
- In-memory and SQLite drivers
- Policy engine for extraction, retention decay, governance deletion, and procedural promotion
- Retrieval traces and memory timelines for observability
- Test harness for memory behavior in CI

## Installation

### Install from PyPI

```bash
pip install engram
```

### Install the latest version from GitHub

```bash
pip install "engram @ git+https://github.com/TechyNilesh/engram.git"
```

### Install from a local clone

```bash
git clone https://github.com/TechyNilesh/engram.git
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick Start

```python
import asyncio

from engram import MemoryClient


async def main() -> None:
    client = MemoryClient(driver="memory")

    memory_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User prefers metric units for measurements.",
        source="conversation",
        importance_score=0.8,
        sensitivity_level="internal",
    )

    results = await client.search(
        query="unit preferences",
        filters={"scope": "user", "user_id": "u-123", "type": "semantic"},
        top_k=5,
    )

    trace = await client.explain(
        query="unit preferences",
        filters={"user_id": "u-123"},
    )

    print(memory_id)
    print(results[0].record.content)
    print(trace.to_markdown())


asyncio.run(main())
```

## Policy Configuration

Create an `engram.yaml` file:

```yaml
driver:
  kind: sqlite
  path: .engram/engram.db

policies:
  extraction:
    - name: extract-user-preferences
      trigger: conversation_turn
      conditions:
        - content_matches: ["prefer", "always", "never", "like"]
      create:
        type: semantic
        scope: user
        importance_score: 0.8
        sensitivity_level: internal

  retention:
    - name: session-episode-decay
      applies_to:
        type: episodic
        scope: session
      decay_function: exponential
      half_life_days: 7
      floor_score: 0.1

  summarization:
    - name: promote-repeated-success
      applies_to:
        type: episodic
      trigger: same_action_success_count >= 3
      promote_to:
        type: procedural
        scope: agent
        content: Confirm prerequisites before executing repeated workflows.

  governance:
    - name: gdpr-user-data
      applies_to:
        scope: user
        sensitivity_level: [confidential, restricted]
      retention_days: 30
      on_user_deletion: delete_all
      audit_log: true
```

Then load it:

```python
from engram import MemoryClient

client = MemoryClient(config="engram.yaml")
```

## Core API

```python
from engram import MemoryClient

client = MemoryClient(driver="memory")

memory_id = client.add_sync(
    type="episodic",
    scope="session",
    session_id="s-1",
    content="Agent attempted payment and received a 402 error.",
    source="tool_call",
    importance_score=0.85,
    sensitivity_level="internal",
)

results = client.search_sync(
    "payment failure",
    filters={"scope": "session", "type": "episodic"},
)
```

Async methods:

- `add()`
- `search()`
- `list()`
- `get()`
- `update()`
- `delete()`
- `explain()`
- `timeline()`
- `ingest_event()`
- `promote()`
- `apply_decay()`
- `forget_user()`

## Testing

```bash
python -m pytest -q
```

The repository ships with coverage for storage behavior, the test harness, and policy engine workflows.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
```

## Repository

- Source: [https://github.com/TechyNilesh/engram](https://github.com/TechyNilesh/engram)
- Install latest from GitHub: `pip install "engram @ git+https://github.com/TechyNilesh/engram.git"`
