from __future__ import annotations

from policy_recommendation_engine.database import default_database_path, seed_database


def main() -> None:
    run_id = seed_database()
    print(f"Seeded SQLite database at {default_database_path()}")
    print(f"Created analysis run #{run_id}")


if __name__ == "__main__":
    main()
