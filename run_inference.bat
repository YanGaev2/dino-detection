@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo   ИНФЕРЕНС DINO С OCR МЕТАДАННЫМИ
echo   HSE Zapovednik Project
echo ================================================================================
echo.

REM Проверка наличия папки venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Виртуальное окружение не найдено!
    echo         Создайте его командой: python -m venv venv
    echo         Затем установите зависимости: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Проверка наличия checkpoint
if not exist "output_dino\checkpoint.pth" (
    echo [ERROR] Checkpoint не найден: output_dino\checkpoint.pth
    echo         Скачайте checkpoint и поместите в папку output_dino
    pause
    exit /b 1
)

REM Проверка параметров
set IMAGES_DIR=%1
if "%IMAGES_DIR%"=="" (
    echo [INFO] Использование: run_inference.bat ^<путь_к_изображениям^>
    echo.
    echo        Пример: run_inference.bat my_images
    echo.
    echo        Папка должна содержать изображения (.jpg, .png)
    echo.
    pause
    exit /b 1
)

REM Проверка папки с изображениями
if not exist "%IMAGES_DIR%" (
    echo [ERROR] Папка с изображениями не найдена: %IMAGES_DIR%
    pause
    exit /b 1
)

echo [INFO] Checkpoint: output_dino\checkpoint.pth
echo [INFO] Images: %IMAGES_DIR%
echo [INFO] OCR: Включен
echo.

REM Запуск инференса
python run_inference.py --checkpoint output_dino/checkpoint.pth --images_dir %IMAGES_DIR% --use_ocr

echo.
echo ================================================================================
echo   ИНФЕРЕНС ЗАВЕРШЁН
echo   Результаты сохранены в папке: inference_results
echo ================================================================================
echo.
echo   Созданные файлы:
echo     - summary_by_species.csv    (Таблица 1: количество особей)
echo     - detailed_detections.csv   (Таблица 2: анализ по фреймам)
echo     - inference_results.json    (полные данные)
echo.
pause

