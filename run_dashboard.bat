@echo off
title VoxGuard Dashboard Server
color 0A
echo ===================================================
echo           VoxGuard Dashboard Launcher
echo ===================================================
echo.
echo Starting Python Backend and Web Interface...
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please make sure you are in the voxguard project directory.
    pause
    exit /b 1
)

echo Opening http://localhost:8000 in your browser...
start http://localhost:8000

echo.
echo ===================================================
echo VoxGuard Server is running!
echo Keep this terminal window open while using the app.
echo Press Ctrl+C to stop the server when finished.
echo ===================================================
echo.

".venv\Scripts\python.exe" -m src.dashboard.backend.main

pause
