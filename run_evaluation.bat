@echo off
title Ice-Risk Route DSS - Automated Evaluation
cd /d "%~dp0"
echo ===================================================
echo Running Route Evaluation Benchmark (Table 7)...
echo ===================================================
"C:\Users\asus tuf gaming\AppData\Local\Programs\Python\Python313\python.exe" run_evaluation.py
echo.
echo Evaluation complete!
pause
