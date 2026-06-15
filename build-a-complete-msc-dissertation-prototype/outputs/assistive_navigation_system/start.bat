@echo off
title Assistive Navigation System
echo.
echo  ====================================================
echo   AI-Powered Assistive Navigation System
echo   MSc Dissertation Prototype
echo  ====================================================
echo.
echo  Starting... (first run installs dependencies)
echo.
python start.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] Something went wrong. Make sure Python 3.10+
    echo  is installed: https://www.python.org/downloads/
    pause
)
