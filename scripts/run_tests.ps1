$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\python.exe") {
    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
} else {
    py -3.12 -m unittest discover -s tests -p "test_*.py"
}
