@echo off
setlocal

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m unittest discover -s tests
) else (
  py -3.12 -m unittest discover -s tests
)
