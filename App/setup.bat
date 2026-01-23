@echo off
echo ============================================================
echo Bed Quality Checker - Complete Setup Script
echo ============================================================
echo.

echo Step 1: Installing Python dependencies...
pip install flask flask-cors torch ultralytics opencv-python pillow numpy
if %ERRORLEVEL% NEQ 0 (
    echo Error installing Python packages!
    pause
    exit /b 1
)
echo ✅ Python dependencies installed
echo.

echo Step 2: Installing Node.js dependencies...
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo Error installing npm packages!
    pause
    exit /b 1
)
echo ✅ Node.js dependencies installed
echo.

echo Step 3: Checking models...
if exist "Stage1_Binary.pth" (
    echo ✅ Stage1_Binary.pth found
) else (
    echo ❌ Stage1_Binary.pth NOT found
)

if exist "Stage2_Detection.pt" (
    echo ✅ Stage2_Detection.pt found
) else (
    echo ❌ Stage2_Detection.pt NOT found
)

if exist "Stage3_BedPillow.pt" (
    echo ✅ Stage3_BedPillow.pt found
) else (
    echo ❌ Stage3_BedPillow.pt NOT found
)
echo.

echo Step 4: Getting your computer's IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP=%%a
    set IP=!IP:~1!
    echo Found IP: !IP!
)
echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next Steps:
echo 1. Start the Python API server in one terminal:
echo    python api_server.py
echo.
echo 2. In another terminal, start the React Native app:
echo    npx react-native start
echo.
echo 3. In a third terminal, build and run on Android:
echo    npx react-native run-android
echo.
echo 4. Configure the app with your computer's IP address
echo    Settings → Enter: http://YOUR_IP:5000
echo.
echo ============================================================
pause
