"""Append-only SQLite event store.

Two tables joined by ``decision_id``: ``decisions`` is written once per recommendation;
``outcomes`` is append-only and 1:N -- a correction is a *new* row, never an ``UPDATE``, and
readers take the latest by autoincrement ``id``. This preserves a tamper-evident audit trail and
keeps the propensity log honest for off-policy evaluation.

The full ``ProspectContext`` is stored as JSON (``model_dump_json``) so ``load_events``
reconstructs the exact contexts that training (Phase 3) and OPE (Phase 5) consume.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from nba.schema import Action, BanditEvent, Outcome, ProspectContext, reward_for


class UnknownDecisionError(KeyError):
    """Raised when an outcome references a ``decision_id`` that was never recorded."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id  TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    address_id   TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    context_json TEXT NOT NULL,
    action       TEXT NOT NULL,
    propensity   REAL NOT NULL,
    policy_name  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
    ts          TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    reward      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON outcomes(decision_id);
"""


class EventStore:
    """An append-only, SQLite-backed log of decisions and their outcomes."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: a uvicorn worker may touch the connection from a thread pool;
        # a single lock around writes keeps the append-only invariant race-free.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append_decision(
        self,
        *,
        context: ProspectContext,
        action: Action,
        propensity: float,
        policy_name: str,
    ) -> str:
        """Record one recommendation and return its freshly minted ``decision_id`` (uuid4)."""
        if not 0.0 < propensity <= 1.0:
            raise ValueError("propensity must be in (0, 1]")
        decision_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions (decision_id, ts, address_id, lat, lon, context_json, "
                "action, propensity, policy_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    datetime.now(UTC).isoformat(),
                    context.address_id,
                    context.lat,
                    context.lon,
                    context.model_dump_json(),
                    action.value,
                    float(propensity),
                    policy_name,
                ),
            )
            self._conn.commit()
        return decision_id

    def append_outcome(self, decision_id: str, outcome: Outcome) -> None:
        """Append an outcome row for ``decision_id`` (reward derived from the reward map).

        Raises:
            UnknownDecisionError: if ``decision_id`` was never recorded.
        """
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
            if exists is None:
                raise UnknownDecisionError(decision_id)
            self._conn.execute(
                "INSERT INTO outcomes (decision_id, ts, outcome, reward) VALUES (?, ?, ?, ?)",
                (
                    decision_id,
                    datetime.now(UTC).isoformat(),
                    outcome.value,
                    reward_for(outcome),
                ),
            )
            self._conn.commit()

    def load_events(self) -> list[BanditEvent]:
        """Reconstruct one :class:`BanditEvent` per decision, joined to its latest outcome."""
        rows = self._conn.execute(
            """
            SELECT d.decision_id, d.ts, d.context_json, d.action, d.propensity,
                   o.outcome AS outcome, o.reward AS reward
            FROM decisions d
            LEFT JOIN (
                SELECT t.decision_id, t.outcome, t.reward
                FROM outcomes t
                JOIN (
                    SELECT decision_id, MAX(id) AS mid FROM outcomes GROUP BY decision_id
                ) latest ON t.id = latest.mid
            ) o ON o.decision_id = d.decision_id
            ORDER BY d.ts, d.decision_id
            """
        ).fetchall()

        events: list[BanditEvent] = []
        for row in rows:
            outcome = Outcome(row["outcome"]) if row["outcome"] is not None else None
            events.append(
                BanditEvent(
                    context=ProspectContext.model_validate_json(row["context_json"]),
                    action=Action(row["action"]),
                    propensity=float(row["propensity"]),
                    reward=float(row["reward"]) if row["reward"] is not None else None,
                    outcome=outcome,
                    timestamp=datetime.fromisoformat(row["ts"]),
                    decision_id=row["decision_id"],
                )
            )
        return events

    def decision_count(self) -> int:
        """Number of recorded decisions."""
        return int(self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    def outcome_count(self) -> int:
        """Number of recorded outcome rows (corrections included)."""
        return int(self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])

    def ingest_bandit_events(
        self, events: list[BanditEvent], *, policy_name: str = "logged"
    ) -> int:
        """Bulk-insert labeled :class:`BanditEvent`s (demo/replay path).

        Preserves each event's ``decision_id`` and timestamp. Skips rows without
        outcomes. Uses ``INSERT OR IGNORE`` on decisions so re-ingesting is safe.
        """
        n = 0
        with self._lock:
            for event in events:
                if event.outcome is None or event.reward is None:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO decisions "
                    "(decision_id, ts, address_id, lat, lon, context_json, "
                    "action, propensity, policy_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.decision_id,
                        event.timestamp.isoformat(),
                        event.context.address_id,
                        event.context.lat,
                        event.context.lon,
                        event.context.model_dump_json(),
                        event.action.value,
                        float(event.propensity),
                        policy_name,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO outcomes (decision_id, ts, outcome, reward) VALUES (?, ?, ?, ?)",
                    (
                        event.decision_id,
                        event.timestamp.isoformat(),
                        event.outcome.value,
                        float(event.reward),
                    ),
                )
                n += 1
            self._conn.commit()
        return n

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
