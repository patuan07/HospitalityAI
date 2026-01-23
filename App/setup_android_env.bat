@echo off
REM Setup Android Environment Variables for This Session
echo Setting up Android development environment...

REM Set ANDROID_HOME
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
echo ANDROID_HOME=%ANDROID_HOME%

REM Add Android SDK tools to PATH for this session
set PATH=%ANDROID_HOME%\platform-tools;%PATH%
set PATH=%ANDROID_HOME%\emulator;%PATH%
set PATH=%ANDROID_HOME%\tools;%PATH%
set PATH=%ANDROID_HOME%\tools\bin;%PATH%

echo.
echo ===================================================
echo Android Environment Set (for this terminal only)
echo ===================================================
echo ANDROID_HOME: %ANDROID_HOME%
echo.

REM Verify adb is accessible
where adb >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] ADB found
    adb version
) else (
    echo [FAIL] ADB not found - check if platform-tools exists
)

echo.
echo ===================================================
echo Next steps:
echo 1. Connect your phone via USB
echo 2. Enable USB debugging on phone
echo 3. Run: adb devices
echo 4. Run: npx react-native run-android
echo ===================================================
echo.
