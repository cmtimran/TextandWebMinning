@echo off
title MediMine AI - 3D Three.js Clinical Intelligence Platform
cd /d "%~dp0"

echo =================================================================
echo       Starting MediMine AI - Native 3D Three.js + Python Platform
echo =================================================================
echo.
echo Launching High-Performance Flask Backend on http://localhost:5000 ...
echo.

python server.py

pause

