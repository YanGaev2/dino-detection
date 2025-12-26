"""
Инференс DINO с извлечением метаданных через OCR
Выходные таблицы:
1. summary_by_species.csv - количество особей по видам
2. detailed_detections.csv - детальный анализ по фреймам (дата, время, температура, камера)

Использование:
    python run_inference.py --checkpoint output_dino/checkpoint.pth --images_dir my_images
"""
import os
import sys
from pathlib import Path

# Windows DLL fix
if sys.platform == 'win32':
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    if cuda_path.exists():
        os.add_dll_directory(str(cuda_path))

import json
import csv
import re
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')

# Добавляем DINO в path
sys.path.insert(0, str(Path("DINO").absolute()))

# EasyOCR (lazy import)
ocr_reader = None


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader"""
    global ocr_reader
    if ocr_reader is None:
        import easyocr
        print("[OCR] Инициализация EasyOCR...")
        ocr_reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available(), verbose=False)
        print("[OCR] EasyOCR готов")
    return ocr_reader


def extract_metadata_ocr(image_path):
    """
    Извлечение метаданных из водяного знака изображения через OCR
    Возвращает: date, time, temperature, camera_id
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Обрезаем нижние 12% изображения (там обычно водяной знак)
        crop_top = int(height * 0.88)
        watermark_region = img.crop((0, crop_top, width, height))
        
        # OCR
        reader = get_ocr_reader()
        watermark_np = np.array(watermark_region)
        results = reader.readtext(watermark_np, detail=0)
        
        text = ' '.join(results).upper()
        
        # Парсинг даты (форматы: DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD)
        date = None
        date_patterns = [
            r'(\d{2})[./](\d{2})[./](20\d{2})',  # DD.MM.YYYY or DD/MM/YYYY
            r'(20\d{2})-(\d{2})-(\d{2})',         # YYYY-MM-DD
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups[0]) == 4:  # YYYY-MM-DD
                    date = f"{groups[0]}-{groups[1]}-{groups[2]}"
                else:  # DD.MM.YYYY
                    date = f"{groups[2]}-{groups[1]}-{groups[0]}"
                break
        
        # Парсинг времени (HH:MM:SS или HH:MM)
        time_val = None
        time_match = re.search(r'(\d{1,2})\s*[:;]\s*(\d{2})(?:\s*[:;]\s*(\d{2}))?', text)
        if time_match:
            h, m, s = time_match.groups()
            s = s or '00'
            time_val = f"{int(h):02d}:{m}:{s}"
        
        # Парсинг температуры (-XX°C, +XX°C, XXC)
        temperature = None
        temp_match = re.search(r'([+-]?\s*\d{1,2})\s*[°]?\s*[CС]', text)
        if temp_match:
            temp_str = temp_match.group(1).replace(' ', '')
            temperature = f"{int(temp_str)}°C"
        
        # Парсинг ID камеры (CAM XX, CAMERA XX, ID: XX)
        camera_id = None
        cam_patterns = [
            r'CAM\w*\s*[:#]?\s*(\w+)',
            r'ID\s*[:#]?\s*(\w+)',
            r'CAMERA\s*[:#]?\s*(\w+)',
        ]
        for pattern in cam_patterns:
            match = re.search(pattern, text)
            if match:
                camera_id = match.group(1)
                break
        
        return {
            'date': date,
            'time': time_val,
            'temperature': temperature,
            'camera_id': camera_id,
            'ocr_raw': text[:100] if text else None
        }
        
    except Exception as e:
        print(f"[OCR] Ошибка при обработке {image_path}: {e}")
        return {'date': None, 'time': None, 'temperature': None, 'camera_id': None, 'ocr_raw': None}


def load_model(checkpoint_path, device='cuda'):
    """Загрузка модели DINO"""
    from models.dino import build_dino
    from util.slconfig import SLConfig
    
    print(f"[MODEL] Загрузка конфигурации...")
    config_path = "DINO/config/DINO/DINO_4scale_swinv2_zapovednik.py"
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device
    
    print(f"[MODEL] Построение модели...")
    model, criterion, postprocessors = build_dino(cfg)
    model.to(device)
    
    print(f"[MODEL] Загрузка весов из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"[MODEL] Модель загружена (эпоха: {epoch})")
    
    return model, postprocessors


def run_inference_single(model, postprocessors, image_path, device='cuda', conf_threshold=0.3):
    """Инференс на одном изображении"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    orig_size = torch.tensor([[orig_h, orig_w]], device=device)
    results = postprocessors['bbox'](outputs, orig_size)[0]
    
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()
    
    detections = []
    for score, label, box in zip(scores, labels, boxes):
        if score >= conf_threshold:
            x1, y1, x2, y2 = box
            detections.append({
                'score': float(score),
                'class_id': int(label),
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            })
    
    return detections


# Маппинг классов
CLASS_NAMES = {
    1: 'mountain_hare',      # Заяц-беляк
    2: 'badger',             # Барсук
    3: 'raccoon_dog',        # Енотовидная собака
    4: 'moose',              # Лось
    5: 'bear',               # Медведь
    6: 'wolf',               # Волк
    7: 'lynx',               # Рысь
    8: 'boar',               # Кабан
    9: 'fox',                # Лиса
    10: 'capercaillie',      # Глухарь
    11: 'black_grouse',      # Тетерев
    12: 'crane',             # Журавль
    13: 'jay',               # Сойка
    14: 'mistle_thrush',     # Дрозд-деряба
    15: 'white_wagtail',     # Белая трясогузка
    16: 'deer',              # Олень
}

CLASS_NAMES_RU = {
    1: 'Заяц-беляк',
    2: 'Барсук',
    3: 'Енотовидная собака',
    4: 'Лось',
    5: 'Медведь',
    6: 'Волк',
    7: 'Рысь',
    8: 'Кабан',
    9: 'Лиса',
    10: 'Глухарь',
    11: 'Тетерев',
    12: 'Журавль',
    13: 'Сойка',
    14: 'Дрозд-деряба',
    15: 'Белая трясогузка',
    16: 'Олень',
}


def run_full_inference(args):
    """Основная функция инференса с OCR"""
    
    print("=" * 80)
    print("  ИНФЕРЕНС DINO С ИЗВЛЕЧЕНИЕМ МЕТАДАННЫХ (OCR)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {args.images_dir}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"OCR: {'Включен' if args.use_ocr else 'Выключен'}")
    print("=" * 80)
    
    # Загрузка модели
    print("\n[1/3] Загрузка модели...")
    model, postprocessors = load_model(args.checkpoint, args.device)
    
    # Поиск изображений
    print("\n[2/3] Поиск изображений...")
    images_dir = Path(args.images_dir)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    print(f"[INFO] Найдено {len(image_files)} изображений")
    
    if not image_files:
        print("[ERROR] Изображения не найдены!")
        return
    
    # Инференс
    print("\n[3/3] Инференс...")
    
    all_detections = []
    species_count = defaultdict(int)
    
    for i, image_path in enumerate(image_files):
        # Детекция
        detections = run_inference_single(model, postprocessors, str(image_path), 
                                          args.device, args.conf_threshold)
        
        # OCR метаданные
        metadata = {'date': None, 'time': None, 'temperature': None, 'camera_id': None}
        if args.use_ocr and detections:
            metadata = extract_metadata_ocr(str(image_path))
        
        # Записываем детекции
        for det in detections:
            class_id = det['class_id']
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            class_name_ru = CLASS_NAMES_RU.get(class_id, class_name)
            
            species_count[class_name] += 1
            
            all_detections.append({
                'file_name': image_path.name,
                'class_id': class_id,
                'class_name': class_name,
                'class_name_ru': class_name_ru,
                'confidence': det['score'],
                'bbox': det['bbox'],
                'date': metadata.get('date'),
                'time': metadata.get('time'),
                'temperature': metadata.get('temperature'),
                'camera_id': metadata.get('camera_id'),
            })
        
        if (i + 1) % 50 == 0:
            print(f"[INFO] Обработано {i+1}/{len(image_files)} изображений...")
    
    print(f"[INFO] Всего детекций: {len(all_detections)}")
    
    # Создание выходной директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # ТАБЛИЦА 1: Количество особей по видам (summary_by_species.csv)
    print("\n" + "=" * 80)
    print("  ТАБЛИЦА 1: КОЛИЧЕСТВО ОСОБЕЙ ПО ВИДАМ")
    print("=" * 80)
    
    summary_file = output_dir / "summary_by_species.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Вид (EN)', 'Вид (RU)', 'Количество особей'])
        
        print(f"{'Вид':<25} {'Количество':>15}")
        print("-" * 40)
        
        for class_id in sorted(CLASS_NAMES.keys()):
            class_name = CLASS_NAMES[class_id]
            class_name_ru = CLASS_NAMES_RU[class_id]
            count = species_count.get(class_name, 0)
            
            if count > 0:
                writer.writerow([class_name, class_name_ru, count])
                print(f"{class_name_ru:<25} {count:>15}")
        
        # Итого
        total = sum(species_count.values())
        writer.writerow(['TOTAL', 'ИТОГО', total])
        print("-" * 40)
        print(f"{'ИТОГО':<25} {total:>15}")
    
    print(f"\n[SAVED] {summary_file}")
    
    # ТАБЛИЦА 2: Детальный анализ по фреймам (detailed_detections.csv)
    print("\n" + "=" * 80)
    print("  ТАБЛИЦА 2: АНАЛИЗ ПО ФРЕЙМАМ")
    print("=" * 80)
    
    detailed_file = output_dir / "detailed_detections.csv"
    with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_name', 'class_name', 'class_name_ru', 'confidence', 
                      'date', 'time', 'temperature', 'camera_id', 'bbox']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for det in all_detections:
            writer.writerow({
                'file_name': det['file_name'],
                'class_name': det['class_name'],
                'class_name_ru': det['class_name_ru'],
                'confidence': f"{det['confidence']:.4f}",
                'date': det['date'] or '',
                'time': det['time'] or '',
                'temperature': det['temperature'] or '',
                'camera_id': det['camera_id'] or '',
                'bbox': str(det['bbox']),
            })
    
    print(f"[SAVED] {detailed_file}")
    
    # Показываем первые 10 строк таблицы 2
    print(f"\n{'Файл':<30} {'Вид':<15} {'Дата':<12} {'Время':<10} {'Темп':<8} {'Камера':<8}")
    print("-" * 90)
    for det in all_detections[:10]:
        print(f"{det['file_name'][:28]:<30} {det['class_name_ru'][:13]:<15} "
              f"{det['date'] or '-':<12} {det['time'] or '-':<10} "
              f"{det['temperature'] or '-':<8} {det['camera_id'] or '-':<8}")
    if len(all_detections) > 10:
        print(f"... и ещё {len(all_detections) - 10} записей")
    
    # JSON с результатами
    json_file = output_dir / "inference_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'images_dir': str(args.images_dir),
            'date': datetime.now().isoformat(),
            'total_images': len(image_files),
            'total_detections': len(all_detections),
            'species_count': dict(species_count),
            'detections': all_detections
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVED] {json_file}")
    
    print("\n" + "=" * 80)
    print("  ГОТОВО!")
    print("=" * 80)
    print(f"Результаты сохранены в: {output_dir}/")
    print(f"  - summary_by_species.csv (Таблица 1: количество особей)")
    print(f"  - detailed_detections.csv (Таблица 2: анализ по фреймам)")
    print(f"  - inference_results.json (полные данные)")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Инференс DINO с OCR метаданными")
    parser.add_argument("--checkpoint", type=str, default="output_dino/checkpoint.pth",
                       help="Путь к checkpoint модели")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Папка с изображениями для анализа")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                       help="Порог уверенности для детекций (default: 0.3)")
    parser.add_argument("--use_ocr", action="store_true", default=True,
                       help="Использовать OCR для извлечения метаданных (default: True)")
    parser.add_argument("--no_ocr", action="store_true",
                       help="Отключить OCR")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Устройство: cuda или cpu")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Папка для сохранения результатов")
    
    args = parser.parse_args()
    
    if args.no_ocr:
        args.use_ocr = False
    
    # Проверки
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint не найден: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.images_dir).exists():
        print(f"[ERROR] Папка с изображениями не найдена: {args.images_dir}")
        sys.exit(1)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA недоступна, переключение на CPU")
        args.device = "cpu"
    
    run_full_inference(args)


if __name__ == "__main__":
    main()

