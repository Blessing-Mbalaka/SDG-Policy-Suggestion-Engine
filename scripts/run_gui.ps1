$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\python.exe") {
    .\.venv\Scripts\python.exe -m policy_recommendation_engine.web --host 127.0.0.1 --port 8000
} else {
    py -3.12 -m policy_recommendation_engine.web --host 127.0.0.1 --port 8000
}
