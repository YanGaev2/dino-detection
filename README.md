# DINO для детекции животных заповедника

Детектор животных на основе модели DINO с backbone Swin Transformer V2.

## Результаты

| Метрика | Значение |
|---------|----------|
| **Mean ROC AUC** | 0.9843 |
| **Overall Recall** | 0.9940 |
| Эпох обучения | 32 |

---

## Быстрый старт

### 1. Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Скачивание checkpoint

Скачайте checkpoint модели и поместите в папку:
```
output_dino/checkpoint.pth
```

### 3. Запуск анализа

**Windows (BAT файл):**
```bash
run.bat my_images
```

**С аннотациями (для ROC AUC):**
```bash
run.bat my_images annotations.json
```

**Ручной запуск:**
```bash
python run_analysis.py --checkpoint output_dino/checkpoint.pth --images_dir my_images --use_ocr
```

---

## Выходные файлы

После анализа в папке `results/` появятся:

| Файл | Описание |
|------|----------|
| `summary_by_species.csv` | **Таблица 1:** Количество особей по видам |
| `detailed_detections.csv` | **Таблица 2:** Анализ по фреймам (дата, время, температура, камера) |
| `analysis_results.json` | ROC AUC + полные данные в JSON |

### Пример summary_by_species.csv

```
Вид (EN),Вид (RU),Количество особей
deer,Олень,75
moose,Лось,91
bear,Медведь,9
wolf,Волк,4
TOTAL,ИТОГО,250
```

### Пример detailed_detections.csv

```
file_name,class_name,class_name_ru,confidence,date,time,temperature,camera_id,bbox
img24_6.jpg,deer,Олень,0.9710,2024-05-15,14:32:45,-5°C,CAM01,[...]
frame_001.png,moose,Лось,0.9532,2024-05-15,08:15:22,-3°C,CAM02,[...]
```

---

## Параметры

```bash
python run_analysis.py \
    --checkpoint output_dino/checkpoint.pth \
    --images_dir my_images \
    --annotations annotations.json \
    --conf_threshold 0.3 \
    --use_ocr \
    --device cuda \
    --output_dir results
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--checkpoint` | `output_dino/checkpoint.pth` | Путь к весам модели |
| `--images_dir` | - | Папка с изображениями (обязательный) |
| `--annotations` | - | Файл аннотаций COCO для ROC AUC (опционально) |
| `--conf_threshold` | 0.3 | Порог уверенности |
| `--use_ocr` | True | Извлечение метаданных через OCR |
| `--no_ocr` | - | Отключить OCR |
| `--device` | cuda | Устройство (cuda/cpu) |
| `--output_dir` | `results` | Папка для результатов |

---

## Классы животных

| ID | EN | RU |
|----|----|----|
| 1 | mountain_hare | Заяц-беляк |
| 2 | badger | Барсук |
| 3 | raccoon_dog | Енотовидная собака |
| 4 | moose | Лось |
| 5 | bear | Медведь |
| 6 | wolf | Волк |
| 7 | lynx | Рысь |
| 8 | boar | Кабан |
| 9 | fox | Лиса |
| 10 | capercaillie | Глухарь |
| 11 | black_grouse | Тетерев |
| 12 | crane | Журавль |
| 13 | jay | Сойка |
| 14 | mistle_thrush | Дрозд-деряба |
| 15 | white_wagtail | Белая трясогузка |
| 16 | deer | Олень |

---

## Формат аннотаций (для ROC AUC)

Для расчёта ROC AUC нужен файл аннотаций в формате COCO:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "mountain_hare"},
    {"id": 2, "name": "badger"}
  ]
}
```

---

## Структура проекта

```
dino-detection/
├── DINO/                    # Модель DINO
├── output_dino/
│   └── checkpoint.pth       # Веса модели (скачать отдельно)
├── run_analysis.py          # Главный скрипт анализа
├── run.bat                  # BAT для запуска
├── train_dino_zapovednik.py # Скрипт обучения
├── requirements.txt         # Зависимости
└── README.md
```

---

## Требования

- Python 3.10-3.12
- CUDA 12.x
- GPU с 16+ GB VRAM (RTX 3090/4090)
- Windows 10/11

---

## Решение проблем

### CUDA out of memory
```bash
python run_analysis.py --device cpu ...
```

### DLL load failed
Добавьте путь к CUDA в PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

### Checkpoint не найден
Скачайте checkpoint и поместите в `output_dino/checkpoint.pth`

---

## Автор

HSE Zapovednik Project
