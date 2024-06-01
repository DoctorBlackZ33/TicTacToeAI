@echo off
REM Ensure you are in the TicTacToeAI directory
cd /d "%~dp0"

REM Path to the specific Python version (adjust this path as needed)
set PYTHON_PATH=C:\path\to\python3.9\python.exe

REM Create virtual environment named TicTacToe using Python 3.9.13
%PYTHON_PATH% -m venv TicTacToe

REM Activate the virtual environment
call TicTacToe\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements from requirements.txt
if exist requirements.txt (
    pip install -r requirements.txt
) else 
    echo requirements.txt not found in the current directory.
    pause
    exit /b 1
)

echo Virtual environment 'TicTacToe' is ready and requirements are installed.
pause
exit /b 0
