@echo off
title NeuroScan AI - Web Application Server
echo ========================================================
echo   NeuroScan AI - HTML+CSS+JS Dyslexia Screening Portal
echo ========================================================
echo.
echo Starting Flask Server on http://localhost:5000 ...
echo (Press Ctrl+C to terminate)
echo.
start "" http://localhost:5000
"C:\Users\dprin\anaconda3\python.exe" server.py
pause
