@echo off
REM Run satellite precompute using the project's virtualenv
set PYTHON=%~dp0\..\.venv\Scripts\python.exe
if exist %PYTHON% (
  echo Using virtualenv python: %PYTHON%
) else (
  set PYTHON=python
)
%PYTHON% -m gtauav_loc.satellite_precompute --dataset-root dataset --out-dir data/precompute_res18_sift --use-gpu
pause