@echo off
echo ================================================================
echo Bed Quality Checker - Android Build Script
echo ================================================================
echo.

REM Set Android environment
echo [1/5] Setting Android environment...
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
set PATH=%ANDROID_HOME%\platform-tools;%ANDROID_HOME%\emulator;%PATH%

REM Verify ADB
echo [2/5] Checking ADB...
adb version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ADB not found!
    echo Please install Android Studio and SDK
    pause
    exit /b 1
)
echo OK - ADB found

REM Check for connected device
echo [3/5] Checking for connected device...
adb devices | findstr /C:"device" /C:"emulator" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: No device connected!
    echo Please connect your phone via USB and enable USB debugging
    adb devices
    pause
    exit /b 1
)
echo OK - Device connected
adb devices

REM Clean and build
echo [4/5] Cleaning previous build...
cd android
call gradlew.bat clean
cd ..

REM Install and run
echo [5/5] Building and installing app...
npx react-native run-android

echo.
echo ================================================================
echo Build complete!
echo ================================================================
pause
