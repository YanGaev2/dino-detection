# DINO для детекции животных заповедника

Детектор животных на основе модели DINO с backbone Swin Transformer V2.

## Результаты

| Метрика | Значение |
|---------|----------|
| **Mean ROC AUC** | 0.9843 |
| **Overall Recall** | 0.9940 |
| Эпох обучения | 32 |

### Метрики по классам

| Класс | ROC AUC | AP |
|-------|---------|-----|
| bear | 1.0000 | 1.0000 |
| wolf | 1.0000 | 1.0000 |
| lynx | 1.0000 | 1.0000 |
| boar | 1.0000 | 1.0000 |
| fox | 1.0000 | 1.0000 |
| capercaillie | 1.0000 | 1.0000 |
| deer | 1.0000 | 1.0000 |
| moose | 0.9927 | 0.8970 |
| mountain_hare | 0.9820 | 0.9157 |
| raccoon_dog | 0.9593 | 0.7507 |

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

### 3. Запуск инференса (с OCR)

**Windows (BAT файл):**
```bash
run_inference.bat my_images
```

**Ручной запуск:**
```bash
python run_inference.py --checkpoint output_dino/checkpoint.pth --images_dir my_images --use_ocr
```

**Выходные файлы:**
- `summary_by_species.csv` — количество особей по видам
- `detailed_detections.csv` — анализ по фреймам (дата, время, температура, камера)

### 4. Оценка модели (ROC AUC)

```bash
run_evaluation.bat my_dataset
```

или:

```bash
python evaluate_model.py --checkpoint output_dino/checkpoint.pth --coco_path my_dataset
```

---

## Формат датасета

Датасет должен быть в формате COCO:

```
my_dataset/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### Формат instances_val2017.json

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
    {"id": 2, "name": "badger"},
    ...
  ]
}
```

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

## Выходные файлы

После оценки в папке `evaluation_results/` появятся:

| Файл | Описание |
|------|----------|
| `metrics_by_class.csv` | Таблица метрик по классам (GT, TP, FP, Precision, Recall, ROC AUC, AP) |
| `detections.csv` | Топ-50 детекций с наивысшей уверенностью |
| `evaluation_results.json` | Все результаты в JSON |

---

## Параметры

```bash
python evaluate_model.py \
    --checkpoint output_dino/checkpoint.pth \
    --coco_path my_dataset \
    --iou_threshold 0.5 \
    --conf_threshold 0.3 \
    --device cuda \
    --output_dir evaluation_results
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--checkpoint` | `output_dino/checkpoint.pth` | Путь к весам модели |
| `--coco_path` | - | Путь к датасету (обязательный) |
| `--iou_threshold` | 0.5 | Порог IoU для matching |
| `--conf_threshold` | 0.3 | Порог уверенности |
| `--device` | cuda | Устройство (cuda/cpu) |
| `--output_dir` | `evaluation_results` | Папка для результатов |

---

## Обучение

```bash
python train_dino_zapovednik.py --epochs 50
```

---

## Требования

- Python 3.10-3.12
- CUDA 12.x
- GPU с 16+ GB VRAM (RTX 3090/4090)
- Windows 10/11

---

## Структура проекта

```
hse-zapovednik/
├── DINO/                    # Модель DINO
│   ├── config/              # Конфигурации
│   ├── models/              # Архитектура модели
│   └── ...
├── output_dino/             # Checkpoints
│   └── checkpoint.pth
├── coco_dataset_final/      # Обучающий датасет
├── evaluate_model.py        # Скрипт оценки
├── train_dino_zapovednik.py # Скрипт обучения
├── run_evaluation.bat       # BAT для быстрого запуска
├── requirements.txt         # Зависимости
└── README.md               # Этот файл
```

---

## Решение проблем

### CUDA out of memory
```bash
python evaluate_model.py --device cpu ...
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

