@echo off
setlocal
set "REAL_CMAKE=D:\proj\.venv\Lib\site-packages\cmake\data\bin\cmake.exe"
if /I "%~1"=="--version" (
  "%REAL_CMAKE%" --version
  exit /b %errorlevel%
)
"%REAL_CMAKE%" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 %*
exit /b %errorlevel%
