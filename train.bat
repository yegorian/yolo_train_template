@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =============================================================================
REM YOLO Template - Training
REM =============================================================================
REM ИНСТРУКЦИЯ:
REM   1. Укажите путь к конфигурации ниже
REM   2. Запустите этот файл
REM =============================================================================

REM Путь к файлу конфигурации обучения
set "CONFIG_PATH=configs/detection/train_config.yaml"

REM =============================================================================
REM КОНЕЦ НАСТРОЕК
REM =============================================================================

echo =============================================================================
echo YOLO Template - Model Training
echo =============================================================================
echo.
echo Config file: %CONFIG_PATH%
echo.

REM Проверка, что конфиг существует
if not exist "%CONFIG_PATH%" (
    echo [ERROR] Config file not found: %CONFIG_PATH%
    echo.
    pause
    exit /b 1
)

echo Starting training...
echo.

REM Запуск скрипта обучения
python scripts\train.py --config "%CONFIG_PATH%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Training failed!
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =============================================================================
echo Training completed successfully!
echo =============================================================================
echo.
pause
