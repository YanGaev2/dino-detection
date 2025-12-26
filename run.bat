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
set DATASET_PATH=%1

if "%DATASET_PATH%"=="" (
    REM Если нет аргумента, проверяем coco_dataset_final
    if exist "coco_dataset_final\val2017" (
        set DATASET_PATH=coco_dataset_final
        echo [INFO] Использую датасет по умолчанию: coco_dataset_final
    ) else (
        echo [INFO] Использование:
        echo.
        echo   run.bat ^<путь_к_датасету^>
        echo.
        echo   Примеры:
        echo     run.bat coco_dataset_final
        echo     run.bat my_dataset
        echo.
        echo   Датасет должен иметь структуру:
        echo     dataset/
        echo       ├── annotations/
        echo       │   └── instances_val2017.json
        echo       └── val2017/
        echo           ├── image1.jpg
        echo           └── image2.png
        echo.
        pause
        exit /b 1
    )
)

REM Определяем пути к изображениям и аннотациям
set IMAGES_DIR=%DATASET_PATH%\val2017
set ANNOTATIONS=%DATASET_PATH%\annotations\instances_val2017.json

REM Проверка папки с изображениями
if not exist "%IMAGES_DIR%" (
    echo [ERROR] Папка с изображениями не найдена: %IMAGES_DIR%
    pause
    exit /b 1
)

REM Проверка файла аннотаций
if not exist "%ANNOTATIONS%" (
    echo [WARN] Файл аннотаций не найден: %ANNOTATIONS%
    echo [WARN] ROC AUC не будет рассчитан
    set ANNOTATIONS=
)

echo [INFO] Checkpoint: output_dino\checkpoint.pth
echo [INFO] Dataset: %DATASET_PATH%
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
