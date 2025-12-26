@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo   ОЦЕНКА МОДЕЛИ DINO ДЛЯ ДЕТЕКЦИИ ЖИВОТНЫХ
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
set DATASET_PATH=%1
if "%DATASET_PATH%"=="" (
    echo [INFO] Использование: run_evaluation.bat ^<путь_к_датасету^>
    echo.
    echo        Пример: run_evaluation.bat my_dataset
    echo.
    echo        Датасет должен иметь структуру:
    echo          my_dataset/
    echo            ├── annotations/
    echo            │   └── instances_val2017.json
    echo            └── val2017/
    echo                ├── image1.jpg
    echo                └── image2.png
    echo.
    
    REM Если есть coco_dataset_final, используем его
    if exist "coco_dataset_final\annotations\instances_val2017.json" (
        echo [INFO] Найден датасет по умолчанию: coco_dataset_final
        set DATASET_PATH=coco_dataset_final
    ) else (
        pause
        exit /b 1
    )
)

REM Проверка структуры датасета
if not exist "%DATASET_PATH%\annotations\instances_val2017.json" (
    echo [ERROR] Файл аннотаций не найден: %DATASET_PATH%\annotations\instances_val2017.json
    pause
    exit /b 1
)

if not exist "%DATASET_PATH%\val2017" (
    echo [ERROR] Папка с изображениями не найдена: %DATASET_PATH%\val2017
    pause
    exit /b 1
)

echo [INFO] Checkpoint: output_dino\checkpoint.pth
echo [INFO] Dataset: %DATASET_PATH%
echo.

REM Запуск оценки
python evaluate_model.py --checkpoint output_dino/checkpoint.pth --coco_path %DATASET_PATH%

echo.
echo ================================================================================
echo   ОЦЕНКА ЗАВЕРШЕНА
echo   Результаты сохранены в папке: evaluation_results
echo ================================================================================
echo.
pause

