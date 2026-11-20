@echo off
REM ============================================================================
REM MahaMonsoon-X Windows Setup Script (Python 3.12.10 required)
REM ============================================================================

set REQUIRED_PY=3.12.10
echo [INFO] Initializing MahaMonsoon-X Environment Setup...
echo [INFO] Required Python Version: %REQUIRED_PY%

REM Check if .venv312 already exists
if exist ".venv312\Scripts\python.exe" (
    echo [OK] Existing virtual environment found at .venv312
    goto activate_env
)

REM Check py launcher for Python 3.12
echo [INFO] Searching for Python 3.12 launcher...
py -3.12 -c "import sys; print('[OK] Found Python:', sys.version)" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.12.10 was not found on your system!
    echo [ACTION] Please download and install Python 3.12.10 64-bit from:
    echo          https://www.python.org/downloads/release/python-31210/
    echo          Ensure 'Add python.exe to PATH' is checked during installation.
    pause
    exit /b 1
)

echo [INFO] Creating virtual environment at .venv312...
py -3.12 -m venv .venv312
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:activate_env
echo [INFO] Activating virtual environment...
call .venv312\Scripts\activate.bat

echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

if exist "requirements.txt" (
    echo [INFO] Installing required dependencies from requirements.txt...
    pip install -r requirements.txt
)

echo.
echo ============================================================================
echo [SUCCESS] MahaMonsoon-X environment is ready!
echo Python executable:
python -c "import sys; print(sys.executable, sys.version)"
echo.
echo To activate in PowerShell:
echo   .\.venv312\Scripts\Activate.ps1
echo To activate in Command Prompt:
echo   call .venv312\Scripts\activate.bat
echo ============================================================================
pause
