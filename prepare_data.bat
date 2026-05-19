@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =============================================================================
REM YOLO Template - Скрипт для подготовки данных
REM =============================================================================
REM ИНСТРУКЦИЯ:
REM   1. Реализуйте загрузчик данных в my_data_loader.py
REM   2. Отредактируйте конфиг configs/dataset/data_config.json под ваш датасет
REM   3. Запустите этот файл
REM =============================================================================

REM Путь к конфигурационному файлу
set "CONFIG_PATH=configs/dataset/data_config.json"

REM =============================================================================
REM КОНЕЦ НАСТРОЕК - ДАЛЕЕ МОЖНО НЕ МЕНЯТЬ
REM =============================================================================

echo =============================================================================
echo YOLO Template - Data Preparation
echo =============================================================================
echo.
echo Config file: %CONFIG_PATH%
echo.

REM Проверяем наличие конфига
if not exist "%CONFIG_PATH%" (
    echo [ERROR] Config file not found: %CONFIG_PATH%
    echo.
    echo Create config file first or edit the path above.
    echo.
    pause
    exit /b 1
)

REM Запускаем скрипт подготовки с конфигом
python -m yolo_template.data_preparation.prepare_yolo_data --config %CONFIG_PATH%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Data preparation failed!
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =============================================================================
echo Data preparation completed successfully!
echo =============================================================================
echo.
pause
