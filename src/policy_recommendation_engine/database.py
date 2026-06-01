from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from policy_recommendation_engine.models import PipelineResult


def default_database_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "policy_engine.sqlite"


def initialize_database(database_path: str | Path | None = None) -> Path:
    path = Path(database_path) if database_path is not None else default_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        create_tables(connection)
        ensure_document_metadata_column(connection)
        connection.commit()
    finally:
        connection.close()

    return path


def create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            analysis_mode TEXT NOT NULL,
            policy_priorities_json TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            theme_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            author TEXT,
            timestamp TEXT,
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            tokens_json TEXT NOT NULL,
            named_entities_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            document_indexes_json TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE
        )
        """
    )


def ensure_document_metadata_column(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(documents)").fetchall()
    column_names = {row[1] for row in rows}
    if "metadata_json" not in column_names:
        connection.execute("ALTER TABLE documents ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            theme TEXT NOT NULL,
            dominant_emotion TEXT NOT NULL,
            intensity TEXT NOT NULL,
            scores_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            theme TEXT NOT NULL,
            public_share REAL NOT NULL,
            policy_share REAL NOT NULL,
            gap_score REAL NOT NULL,
            severity TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            insight_text TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE
        )
        """
    )


def save_pipeline_result(
    result: PipelineResult,
    *,
    analysis_mode: str,
    policy_priorities: dict[str, float],
    database_path: str | Path | None = None,
) -> int:
    path = initialize_database(database_path)

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            """
            INSERT INTO analysis_runs (
                created_at,
                analysis_mode,
                policy_priorities_json,
                document_count,
                theme_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                analysis_mode,
                json.dumps(policy_priorities, sort_keys=True),
                len(result.documents),
                len(result.themes),
            ),
        )
        run_id = int(cursor.lastrowid)

        save_documents(connection, run_id, result)
        save_themes(connection, run_id, result)
        save_emotions(connection, run_id, result)
        save_policy_gaps(connection, run_id, result)
        save_insights(connection, run_id, result)
        connection.commit()
    finally:
        connection.close()

    return run_id


def get_database_summary(database_path: str | Path | None = None) -> dict[str, int]:
    path = initialize_database(database_path)
    table_names = ("analysis_runs", "documents", "themes", "emotions", "policy_gaps", "insights")
    summary: dict[str, int] = {}

    connection = sqlite3.connect(path)
    try:
        for table_name in table_names:
            row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            summary[table_name] = int(row[0])
    finally:
        connection.close()

    return summary


def list_analysis_runs(database_path: str | Path | None = None, limit: int = 10) -> list[dict[str, object]]:
    path = initialize_database(database_path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                analysis_mode,
                document_count,
                theme_count
            FROM analysis_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        runs: list[dict[str, object]] = []
        for row in rows:
            run_id = int(row["id"])
            runs.append(
                {
                    "id": run_id,
                    "created_at": row["created_at"],
                    "analysis_mode": row["analysis_mode"],
                    "document_count": int(row["document_count"]),
                    "theme_count": int(row["theme_count"]),
                    "themes": get_theme_names(connection, run_id),
                    "insights": get_insight_texts(connection, run_id),
                }
            )
    finally:
        connection.close()

    return runs


def get_theme_names(connection: sqlite3.Connection, run_id: int, limit: int = 3) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM themes
        WHERE run_id = ?
        ORDER BY score DESC, name ASC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [str(row["name"]) for row in rows]


def get_insight_texts(connection: sqlite3.Connection, run_id: int, limit: int = 2) -> list[str]:
    rows = connection.execute(
        """
        SELECT insight_text
        FROM insights
        WHERE run_id = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [str(row["insight_text"]) for row in rows]


def save_documents(connection: sqlite3.Connection, run_id: int, result: PipelineResult) -> None:
    for processed in result.documents:
        document = processed.document
        timestamp = document.timestamp.isoformat() if document.timestamp else None
        connection.execute(
            """
            INSERT INTO documents (
                run_id,
                source,
                author,
                timestamp,
                raw_text,
                normalized_text,
                tokens_json,
                named_entities_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                document.source,
                document.author,
                timestamp,
                document.text,
                processed.normalized_text,
                json.dumps(processed.tokens),
                json.dumps(processed.named_entities),
                json.dumps(document.metadata, sort_keys=True),
            ),
        )


def save_themes(connection: sqlite3.Connection, run_id: int, result: PipelineResult) -> None:
    for theme in result.themes:
        connection.execute(
            """
            INSERT INTO themes (
                run_id,
                name,
                keywords_json,
                document_indexes_json,
                score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                theme.name,
                json.dumps(theme.keywords),
                json.dumps(theme.document_indexes),
                theme.score,
            ),
        )


def save_emotions(connection: sqlite3.Connection, run_id: int, result: PipelineResult) -> None:
    for theme, signal in result.emotions_by_theme.items():
        connection.execute(
            """
            INSERT INTO emotions (
                run_id,
                theme,
                dominant_emotion,
                intensity,
                scores_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                theme,
                signal.dominant_emotion,
                signal.intensity,
                json.dumps(signal.scores, sort_keys=True),
            ),
        )


def save_policy_gaps(connection: sqlite3.Connection, run_id: int, result: PipelineResult) -> None:
    for gap in result.policy_gaps:
        connection.execute(
            """
            INSERT INTO policy_gaps (
                run_id,
                theme,
                public_share,
                policy_share,
                gap_score,
                severity
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                gap.theme,
                gap.public_share,
                gap.policy_share,
                gap.gap_score,
                gap.severity,
            ),
        )


def save_insights(connection: sqlite3.Connection, run_id: int, result: PipelineResult) -> None:
    for insight in result.insights:
        connection.execute(
            """
            INSERT INTO insights (run_id, insight_text)
            VALUES (?, ?)
            """,
            (run_id, insight),
        )


def seed_database(database_path: str | Path | None = None) -> int:
    from policy_recommendation_engine.ingestion import document_from_text
    from policy_recommendation_engine.pipeline import PolicyIntelligencePipeline

    documents = (
        document_from_text("Water shortages are unbearable and residents are angry.", source="seed"),
        document_from_text("Healthcare delays make families afraid.", source="seed"),
        document_from_text("Transport delays are frustrating for workers.", source="seed"),
    )
    priorities = {"water": 0.05, "healthcare": 0.2, "transport": 0.1}
    result = PolicyIntelligencePipeline().run(documents, policy_priorities=priorities)

    return save_pipeline_result(
        result,
        analysis_mode="seed",
        policy_priorities=priorities,
        database_path=database_path,
    )
