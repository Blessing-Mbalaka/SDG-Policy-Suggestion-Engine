from __future__ import annotations

import sqlite3

from policy_recommendation_engine.database import default_database_path


def main() -> None:
    database_path = default_database_path()
    connection = sqlite3.connect(database_path)
    try:
        for table_name in ("analysis_runs", "documents", "themes", "emotions", "policy_gaps", "insights"):
            row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            print(f"{table_name}: {row[0]}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
