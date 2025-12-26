"""
Скрипт оценки модели DINO для детекции животных
Выводит: ROC AUC, таблицу по классам, таблицу детекций

Использование:
    python evaluate_model.py --checkpoint output_dino/checkpoint.pth --coco_path coco_dataset_final

Автор: HSE Zapovednik Project
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
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Добавляем DINO в path
sys.path.insert(0, str(Path("DINO").absolute()))


def print_header(title):
    """Печать заголовка"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def load_model(checkpoint_path, device='cuda'):
    """Загрузка модели DINO"""
    from models.dino import build_dino
    from util.slconfig import SLConfig
    
    print(f"[INFO] Загрузка конфигурации...")
    config_path = "DINO/config/DINO/DINO_4scale_swinv2_zapovednik.py"
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device
    
    print(f"[INFO] Построение модели...")
    model, criterion, postprocessors = build_dino(cfg)
    model.to(device)
    
    print(f"[INFO] Загрузка весов из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"[INFO] Модель загружена (эпоха: {epoch})")
    
    return model, postprocessors, cfg


def run_inference(model, postprocessors, image_path, device='cuda', conf_threshold=0.3):
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
        x1, y1, x2, y2 = box
        detections.append({
            'score': float(score),
            'class_id': int(label),
            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        })
    
    return detections


def match_detections(detections, gt_annotations, iou_threshold=0.5):
    """Сопоставление детекций с ground truth"""
    results = []
    
    gt_by_class = defaultdict(list)
    for ann in gt_annotations:
        gt_by_class[ann['category_id']].append(ann['bbox'])
    
    matched_gt = defaultdict(set)
    sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    for det in sorted_dets:
        class_id = det['class_id']
        score = det['score']
        det_box = det['bbox']
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_by_class.get(class_id, [])):
            if gt_idx in matched_gt[class_id]:
                continue
            
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
        
        if is_tp:
            matched_gt[class_id].add(best_gt_idx)
        
        results.append({
            'score': score,
            'is_tp': is_tp,
            'class_id': class_id,
            'iou': best_iou
        })
    
    return results


def evaluate_model(args):
    """Основная функция оценки"""
    
    print_header("ОЦЕНКА МОДЕЛИ DINO ДЛЯ ДЕТЕКЦИИ ЖИВОТНЫХ")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.coco_path}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Confidence threshold: {args.conf_threshold}")
    
    # 1. Загрузка модели
    print_header("ШАГ 1: ЗАГРУЗКА МОДЕЛИ")
    model, postprocessors, cfg = load_model(args.checkpoint, args.device)
    
    # 2. Загрузка аннотаций
    print_header("ШАГ 2: ЗАГРУЗКА ДАННЫХ")
    ann_file = Path(args.coco_path) / "annotations" / "instances_val2017.json"
    
    if not ann_file.exists():
        print(f"[ERROR] Файл аннотаций не найден: {ann_file}")
        sys.exit(1)
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    class_names = {c['id']: c['name'] for c in coco_data['categories']}
    print(f"[INFO] Загружено {len(coco_data['images'])} изображений")
    print(f"[INFO] Классов: {len(class_names)}")
    print(f"[INFO] Классы: {', '.join(class_names.values())}")
    
    # Группировка аннотаций
    anns_by_image = defaultdict(list)
    gt_count_by_class = defaultdict(int)
    
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
        gt_count_by_class[ann['category_id']] += 1
    
    # 3. Инференс
    print_header("ШАГ 3: ИНФЕРЕНС")
    images_dir = Path(args.coco_path) / "val2017"
    
    all_results = []
    all_detections = []  # Для детальной таблицы
    processed = 0
    
    for i, img_info in enumerate(coco_data['images']):
        image_path = images_dir / img_info['file_name']
        
        if not image_path.exists():
            print(f"[WARN] Изображение не найдено: {image_path}")
            continue
        
        detections = run_inference(model, postprocessors, str(image_path), args.device)
        gt_anns = anns_by_image.get(img_info['id'], [])
        matches = match_detections(detections, gt_anns, args.iou_threshold)
        all_results.extend(matches)
        
        # Сохраняем детекции выше порога для таблицы
        for det in detections:
            if det['score'] >= args.conf_threshold:
                all_detections.append({
                    'image': img_info['file_name'],
                    'class_id': det['class_id'],
                    'class_name': class_names.get(det['class_id'], 'unknown'),
                    'confidence': det['score'],
                    'bbox': det['bbox']
                })
        
        processed += 1
        if processed % 50 == 0:
            print(f"[INFO] Обработано {processed}/{len(coco_data['images'])} изображений...")
    
    print(f"[INFO] Всего детекций: {len(all_results)}")
    print(f"[INFO] Детекций выше порога {args.conf_threshold}: {len(all_detections)}")
    
    # 4. Расчёт метрик
    print_header("ШАГ 4: РАСЧЁТ МЕТРИК")
    
    # Группировка по классам
    by_class = defaultdict(lambda: {'scores': [], 'labels': []})
    tp_by_class = defaultdict(int)
    fp_by_class = defaultdict(int)
    
    for r in all_results:
        class_id = r['class_id']
        by_class[class_id]['scores'].append(r['score'])
        by_class[class_id]['labels'].append(1 if r['is_tp'] else 0)
        if r['is_tp']:
            tp_by_class[class_id] += 1
        else:
            fp_by_class[class_id] += 1
    
    # Таблица 1: Метрики по классам
    print("\n" + "=" * 90)
    print("ТАБЛИЦА 1: МЕТРИКИ ПО КЛАССАМ")
    print("=" * 90)
    print(f"{'Класс':<20} {'GT':>6} {'TP':>6} {'FP':>6} {'Precision':>10} {'Recall':>10} {'ROC AUC':>10} {'AP':>10}")
    print("-" * 90)
    
    class_metrics = []
    all_aucs = []
    
    for class_id in sorted(class_names.keys()):
        class_name = class_names[class_id]
        gt_count = gt_count_by_class.get(class_id, 0)
        tp = tp_by_class.get(class_id, 0)
        fp = fp_by_class.get(class_id, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / gt_count if gt_count > 0 else 0
        
        scores = np.array(by_class[class_id]['scores']) if class_id in by_class else np.array([])
        labels = np.array(by_class[class_id]['labels']) if class_id in by_class else np.array([])
        
        auc_str = "N/A"
        ap_str = "N/A"
        
        if len(scores) > 0 and labels.sum() > 0 and (1 - labels).sum() > 0:
            try:
                auc = roc_auc_score(labels, scores)
                ap = average_precision_score(labels, scores)
                auc_str = f"{auc:.4f}"
                ap_str = f"{ap:.4f}"
                all_aucs.append(auc)
            except:
                pass
        
        print(f"{class_name:<20} {gt_count:>6} {tp:>6} {fp:>6} {precision:>10.4f} {recall:>10.4f} {auc_str:>10} {ap_str:>10}")
        
        class_metrics.append({
            'class_name': class_name,
            'gt_count': gt_count,
            'tp': tp,
            'fp': fp,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc_str,
            'ap': ap_str
        })
    
    print("-" * 90)
    
    mean_auc = np.mean(all_aucs) if all_aucs else 0
    total_tp = sum(tp_by_class.values())
    total_fp = sum(fp_by_class.values())
    total_gt = sum(gt_count_by_class.values())
    
    print(f"{'ИТОГО':<20} {total_gt:>6} {total_tp:>6} {total_fp:>6} {'':<10} {'':<10} {mean_auc:>10.4f}")
    
    # Таблица 2: Топ детекций
    print("\n" + "=" * 90)
    print("ТАБЛИЦА 2: ТОП-50 ДЕТЕКЦИЙ ПО УВЕРЕННОСТИ")
    print("=" * 90)
    print(f"{'#':<4} {'Изображение':<35} {'Класс':<20} {'Confidence':>12}")
    print("-" * 90)
    
    sorted_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)[:50]
    for i, det in enumerate(sorted_detections, 1):
        img_short = det['image'][:32] + "..." if len(det['image']) > 35 else det['image']
        print(f"{i:<4} {img_short:<35} {det['class_name']:<20} {det['confidence']:>12.4f}")
    
    # Сводка
    print_header("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"  • Обработано изображений: {processed}")
    print(f"  • Всего объектов в GT: {total_gt}")
    print(f"  • True Positives: {total_tp}")
    print(f"  • False Positives: {total_fp}")
    print(f"  • Mean ROC AUC: {mean_auc:.4f}")
    print(f"  • Overall Precision: {total_tp / (total_tp + total_fp):.4f}" if (total_tp + total_fp) > 0 else "  • Overall Precision: N/A")
    print(f"  • Overall Recall: {total_tp / total_gt:.4f}" if total_gt > 0 else "  • Overall Recall: N/A")
    
    # Сохранение результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # CSV 1: Метрики по классам
    csv_classes = output_dir / "metrics_by_class.csv"
    with open(csv_classes, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['class_name', 'gt_count', 'tp', 'fp', 'precision', 'recall', 'roc_auc', 'ap'])
        writer.writeheader()
        writer.writerows(class_metrics)
    
    # CSV 2: Детекции
    csv_detections = output_dir / "detections.csv"
    with open(csv_detections, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'class_name', 'confidence', 'bbox'])
        writer.writeheader()
        for det in sorted_detections:
            writer.writerow({
                'image': det['image'],
                'class_name': det['class_name'],
                'confidence': f"{det['confidence']:.4f}",
                'bbox': str(det['bbox'])
            })
    
    # JSON со всеми результатами
    json_results = output_dir / "evaluation_results.json"
    with open(json_results, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'dataset': args.coco_path,
            'date': datetime.now().isoformat(),
            'mean_roc_auc': mean_auc,
            'total_images': processed,
            'total_gt': total_gt,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'class_metrics': class_metrics
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Результаты сохранены в {output_dir}/")
    print(f"  - metrics_by_class.csv")
    print(f"  - detections.csv")
    print(f"  - evaluation_results.json")
    
    print("\n" + "=" * 80)
    print(f"  MEAN ROC AUC: {mean_auc:.4f}")
    print("=" * 80 + "\n")
    
    return mean_auc


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка модели DINO для детекции животных")
    parser.add_argument("--checkpoint", type=str, default="output_dino/checkpoint.pth",
                       help="Путь к checkpoint модели")
    parser.add_argument("--coco_path", type=str, required=True,
                       help="Путь к датасету в формате COCO (папка с val2017 и annotations)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                       help="Порог IoU для matching (default: 0.5)")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                       help="Порог уверенности для детекций (default: 0.3)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Устройство: cuda или cpu")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Папка для сохранения результатов")
    
    args = parser.parse_args()
    
    # Проверки
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint не найден: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.coco_path).exists():
        print(f"[ERROR] Датасет не найден: {args.coco_path}")
        sys.exit(1)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA недоступна, переключение на CPU")
        args.device = "cpu"
    
    evaluate_model(args)


if __name__ == "__main__":
    main()

