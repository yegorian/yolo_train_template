@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =============================================================================
REM YOLO Template - Model Benchmarking
REM =============================================================================
REM ИНСТРУКЦИЯ:
REM   1. Укажите путь к конфигурации ниже
REM   2. Запустите этот файл
REM
REM Режимы валидации (указываются в конфиге):
REM   mode: standard  - стандартная валидация через model.val() (Ultralytics)
REM   mode: custom    - кастомная валидация с метриками из yolo_template.metrics
REM =============================================================================

REM Путь к файлу конфигурации бенчмаркинга
set "CONFIG_PATH=configs/benchmark/benchmark.yaml"

REM =============================================================================
REM КОНЕЦ НАСТРОЕК
REM =============================================================================

echo =============================================================================
echo YOLO Template - Model Benchmarking
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

echo Starting benchmarking...
echo.

REM Запуск скрипта бенчмаркинга
python scripts\benchmark.py --config "%CONFIG_PATH%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Benchmarking failed!
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =============================================================================
echo Benchmarking completed successfully!
echo =============================================================================
echo.
pause
