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

echo.
echo ================================================================================
echo   ВЫБЕРИТЕ ФОРМАТ ДАТАСЕТА
echo ================================================================================
echo.
echo   1. COCO (JSON аннотации)
echo      Структура:
echo        dataset/
echo          ├── annotations/
echo          │   └── instances_val2017.json
echo          └── val2017/
echo              └── *.jpg, *.png
echo.
echo   2. YOLO (TXT аннотации - Ultralytics формат)
echo      Структура:
echo        dataset/
echo          ├── labels/
echo          │   └── *.txt (class x_center y_center w h)
echo          └── images/
echo              └── *.jpg, *.png
echo.
echo   3. Только изображения (без аннотаций, без ROC AUC)
echo.
echo ================================================================================

set /p FORMAT_CHOICE="Введите номер формата (1/2/3): "

if "%FORMAT_CHOICE%"=="1" (
    set FORMAT=coco
    echo [INFO] Выбран формат: COCO
) else if "%FORMAT_CHOICE%"=="2" (
    set FORMAT=yolo
    echo [INFO] Выбран формат: YOLO
) else if "%FORMAT_CHOICE%"=="3" (
    set FORMAT=none
    echo [INFO] Выбран режим: только изображения
) else (
    echo [ERROR] Неверный выбор. Используйте 1, 2 или 3
    pause
    exit /b 1
)

echo.

REM Запрос пути к датасету
set DATASET_PATH=%1

if "%DATASET_PATH%"=="" (
    echo ================================================================================
    echo   ВВЕДИТЕ ПУТЬ К ДАТАСЕТУ
    echo ================================================================================
    echo.
    
    if "%FORMAT%"=="coco" (
        echo   Для COCO формата укажите корневую папку датасета.
        echo   Пример: coco_dataset_final
        echo.
        if exist "coco_dataset_final" (
            echo   [Найден датасет по умолчанию: coco_dataset_final]
            echo   Нажмите Enter для использования или введите другой путь.
            set /p DATASET_PATH="Путь к датасету: "
            if "!DATASET_PATH!"=="" set DATASET_PATH=coco_dataset_final
        ) else (
            set /p DATASET_PATH="Путь к датасету: "
        )
    ) else if "%FORMAT%"=="yolo" (
        echo   Для YOLO формата укажите корневую папку датасета.
        echo   Папка должна содержать подпапки images/ и labels/
        echo   Пример: my_yolo_dataset
        echo.
        set /p DATASET_PATH="Путь к датасету: "
    ) else (
        echo   Укажите папку с изображениями.
        echo   Пример: my_images
        echo.
        set /p DATASET_PATH="Путь к изображениям: "
    )
)

REM Проверка что путь не пустой
if "%DATASET_PATH%"=="" (
    echo [ERROR] Путь к датасету не указан!
    pause
    exit /b 1
)

REM Определяем пути в зависимости от формата
if "%FORMAT%"=="coco" (
    REM COCO формат
    if exist "%DATASET_PATH%\val2017" (
        set IMAGES_DIR=%DATASET_PATH%\val2017
    ) else if exist "%DATASET_PATH%\images\val" (
        set IMAGES_DIR=%DATASET_PATH%\images\val
    ) else if exist "%DATASET_PATH%\images" (
        set IMAGES_DIR=%DATASET_PATH%\images
    ) else (
        set IMAGES_DIR=%DATASET_PATH%
    )
    
    if exist "%DATASET_PATH%\annotations\instances_val2017.json" (
        set ANNOTATIONS=%DATASET_PATH%\annotations\instances_val2017.json
    ) else if exist "%DATASET_PATH%\annotations\instances_val.json" (
        set ANNOTATIONS=%DATASET_PATH%\annotations\instances_val.json
    ) else (
        REM Ищем любой JSON в annotations
        for %%f in ("%DATASET_PATH%\annotations\*.json") do set ANNOTATIONS=%%f
    )
) else if "%FORMAT%"=="yolo" (
    REM YOLO формат
    if exist "%DATASET_PATH%\images\val" (
        set IMAGES_DIR=%DATASET_PATH%\images\val
    ) else if exist "%DATASET_PATH%\images" (
        set IMAGES_DIR=%DATASET_PATH%\images
    ) else (
        set IMAGES_DIR=%DATASET_PATH%
    )
    
    if exist "%DATASET_PATH%\labels\val" (
        set ANNOTATIONS=%DATASET_PATH%\labels\val
    ) else if exist "%DATASET_PATH%\labels" (
        set ANNOTATIONS=%DATASET_PATH%\labels
    ) else (
        set ANNOTATIONS=
    )
) else (
    REM Только изображения
    set IMAGES_DIR=%DATASET_PATH%
    set ANNOTATIONS=
)

REM Проверка папки с изображениями
if not exist "%IMAGES_DIR%" (
    echo [ERROR] Папка с изображениями не найдена: %IMAGES_DIR%
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   ПАРАМЕТРЫ АНАЛИЗА
echo ================================================================================
echo   Checkpoint:   output_dino\checkpoint.pth
echo   Формат:       %FORMAT%
echo   Изображения:  %IMAGES_DIR%
if not "%ANNOTATIONS%"=="" (
    echo   Аннотации:    %ANNOTATIONS%
) else (
    echo   Аннотации:    [не указаны - ROC AUC не будет рассчитан]
)
echo   OCR:          Включен
echo ================================================================================
echo.

echo.
echo Нажмите любую клавишу для запуска анализа...
pause >nul

REM Запуск анализа
echo.
echo [RUN] Запуск анализа...
echo.

if "%ANNOTATIONS%"=="" (
    python run_analysis.py --checkpoint output_dino/checkpoint.pth --images_dir "%IMAGES_DIR%" --use_ocr
) else (
    python run_analysis.py --checkpoint output_dino/checkpoint.pth --images_dir "%IMAGES_DIR%" --annotations "%ANNOTATIONS%" --format %FORMAT% --use_ocr
)

set PYTHON_EXIT_CODE=%errorlevel%

echo.
echo ================================================================================
if %PYTHON_EXIT_CODE% EQU 0 (
    echo   АНАЛИЗ ЗАВЕРШЁН УСПЕШНО
    echo   Результаты сохранены в папке: results
    echo ================================================================================
    echo.
    echo   Созданные файлы:
    echo     - summary_by_species.csv      Таблица 1: количество особей
    echo     - detailed_detections.csv     Таблица 2: анализ по фреймам
    echo     - analysis_results.json       ROC AUC + полные данные
) else (
    echo   ОШИБКА ПРИ ВЫПОЛНЕНИИ АНАЛИЗА
    echo   Код ошибки: %PYTHON_EXIT_CODE%
    echo ================================================================================
    echo.
    echo   Проверьте:
    echo     1. Установлены ли все зависимости
    echo     2. Правильный ли путь к датасету
    echo     3. Есть ли checkpoint в output_dino/
)
echo ================================================================================
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
