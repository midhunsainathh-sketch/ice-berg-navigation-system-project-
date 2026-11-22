@echo off
title Ice-Risk Route DSS - Automated Tests
cd /d "%~dp0"
echo ===================================================
echo Running Automated Unit and Integration Tests...
echo ===================================================
"C:\Users\asus tuf gaming\AppData\Local\Programs\Python\Python313\python.exe" -m unittest discover -s tests -v
echo.
echo Tests complete!
pause
