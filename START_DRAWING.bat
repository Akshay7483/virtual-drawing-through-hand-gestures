@echo off
echo ==========================================
echo   Virtual Drawing - AI Enhanced
echo   Starting webcam drawing...
echo ==========================================
echo.
cd /d "%~dp0src\backend"
python draw_hand_gestures.py
pause
