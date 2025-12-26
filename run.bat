@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo   АНАЛИЗ МОДЕЛИ DINO ДЛЯ ДЕТЕКЦИИ ЖИВОТНЫХ
echo   HSE Zapovednik Project
echo ================================================================================
echo.

REM Проверка наличия Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден! Установите Python 3.10-3.12
    pause
    exit /b 1
)

REM Создание venv если не существует
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Виртуальное окружение не найдено. Создаю...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Не удалось создать виртуальное окружение
        pause
        exit /b 1
    )
    echo [SETUP] Виртуальное окружение создано
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Проверка и установка зависимостей
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Установка зависимостей... (это может занять 5-10 минут)
    echo [SETUP] Устанавливаю PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo [SETUP] Устанавливаю остальные зависимости...
    pip install -r requirements.txt --quiet
    echo [SETUP] Установка завершена!
)

REM Проверка EasyOCR
python -c "import easyocr" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Установка EasyOCR...
    pip install easyocr --quiet
    echo [SETUP] EasyOCR установлен
)

REM Проверка наличия checkpoint
if not exist "output_dino\checkpoint.pth" (
    echo.
    echo [ERROR] Checkpoint не найден: output_dino\checkpoint.pth
    echo         Скачайте checkpoint и поместите в папку output_dino
    echo.
    pause
    exit /b 1
)

REM Проверка параметров
set IMAGES_DIR=%1
set ANNOTATIONS=%2

if "%IMAGES_DIR%"=="" (
    echo [INFO] Использование:
    echo.
    echo   run.bat ^<путь_к_изображениям^> [путь_к_аннотациям]
    echo.
    echo   Примеры:
    echo     run.bat my_images
    echo     run.bat my_images annotations.json
    echo.
    echo   Папка должна содержать изображения (.jpg, .png)
    echo   Аннотации (опционально) - для расчёта ROC AUC
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
if not "%ANNOTATIONS%"=="" (
    echo [INFO] Annotations: %ANNOTATIONS%
)
echo [INFO] OCR: Включен
echo.

REM Запуск анализа
if "%ANNOTATIONS%"=="" (
    python run_analysis.py --checkpoint output_dino/checkpoint.pth --images_dir %IMAGES_DIR% --use_ocr
) else (
    python run_analysis.py --checkpoint output_dino/checkpoint.pth --images_dir %IMAGES_DIR% --annotations %ANNOTATIONS% --use_ocr
)

echo.
echo ================================================================================
echo   АНАЛИЗ ЗАВЕРШЁН
echo   Результаты сохранены в папке: results
echo ================================================================================
echo.
echo   Созданные файлы:
echo     - summary_by_species.csv      (Таблица 1: количество особей)
echo     - detailed_detections.csv     (Таблица 2: анализ по фреймам)
echo     - analysis_results.json       (ROC AUC + полные данные)
echo.
pause
