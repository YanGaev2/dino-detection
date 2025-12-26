"""
Скрипт для обучения DINO с Swin-V2-L на датасете Заповедник (16 классов)
RTX 5090 (32GB VRAM), CUDA 12.8, PyTorch 2.9.1
"""
import os
import sys
import subprocess
import signal
import logging
from pathlib import Path

# Windows DLL fix: добавляем пути к PyTorch и CUDA DLL перед импортом
if sys.platform == 'win32':
    # PyTorch DLLs
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    
    # CUDA DLLs
    cuda_paths = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
    ]
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            os.add_dll_directory(str(cuda_path))
            break

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_coco_dataset(coco_path):
    """Проверяет наличие COCO датасета"""
    coco_path = Path(coco_path)
    
    required = [
        coco_path / "train2017",
        coco_path / "annotations" / "instances_train2017.json",
    ]
    
    for path in required:
        if not path.exists():
            logger.error(f"Не найден: {path}")
            return False
        logger.info(f"[OK] {path}")
    
    return True


def check_dino():
    """Проверяет наличие DINO"""
    dino_path = Path("DINO")
    
    if not dino_path.exists():
        logger.error("DINO не найден! Клонируйте: git clone https://github.com/IDEA-Research/DINO.git")
        return False
    
    logger.info("[OK] DINO найден")
    return True


def compile_cuda_ops():
    """Компилирует CUDA операторы DINO"""
    ops_path = Path("DINO/models/dino/ops")
    
    if not ops_path.exists():
        logger.error(f"CUDA ops не найдены: {ops_path}")
        return False
    
    logger.info("Компиляция CUDA операторов...")
    
    # Проверяем, уже скомпилированы ли
    build_dir = ops_path / "build"
    if build_dir.exists():
        logger.info("CUDA ops уже скомпилированы, пропускаем")
        return True
    
    # Компиляция
    cmd = [sys.executable, "setup.py", "build", "install"]
    result = subprocess.run(cmd, cwd=str(ops_path), capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Ошибка компиляции CUDA ops: {result.stderr}")
        logger.info("Попробуйте запустить вручную:")
        logger.info(f"  cd {ops_path}")
        logger.info("  python setup.py build install")
        return False
    
    logger.info("[OK] CUDA ops скомпилированы")
    return True


def run_training(coco_path, output_dir, epochs=36, batch_size=2, resume=""):
    """Запускает обучение DINO напрямую (не через subprocess)"""
    
    config_path = "DINO/config/DINO/DINO_4scale_swinv2_zapovednik.py"
    
    if not Path(config_path).exists():
        logger.error(f"Config not found: {config_path}")
        return False
    
    # Добавляем DINO в PYTHONPATH
    dino_path = str(Path("DINO").absolute())
    if dino_path not in sys.path:
        sys.path.insert(0, dino_path)
    
    logger.info("=" * 60)
    logger.info("STARTING DINO TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Dataset: {coco_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    if resume:
        logger.info(f"Resume: {resume}")
    logger.info("Press Ctrl+C to stop with checkpoint save")
    logger.info("=" * 60)
    
    # Формируем аргументы для DINO main
    sys.argv = [
        "main.py",
        "--config_file", config_path,
        "--coco_path", str(coco_path),
        "--output_dir", str(output_dir),
        "--options", f"batch_size={batch_size}",
        "--options", f"epochs={epochs}",
    ]
    
    if resume and Path(resume).exists():
        sys.argv.extend(["--resume", str(resume)])
        logger.info(f"Resuming from: {resume}")
    
    # Импортируем и запускаем DINO напрямую
    # Это позволяет Ctrl+C работать корректно
    try:
        # Меняем рабочую директорию на DINO
        original_cwd = os.getcwd()
        
        # Импортируем main из DINO
        from main import main, get_args_parser
        
        parser = get_args_parser()
        args = parser.parse_args()
        
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        main(args)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        return True
        
    except KeyboardInterrupt:
        # Это не должно происходить - main.py сам обрабатывает Ctrl+C
        logger.info("")
        logger.info("=" * 60)
        logger.info("[INTERRUPT] Training stopped by user")
        logger.info("Check output_dino/ for checkpoints")
        logger.info("=" * 60)
        return False
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение DINO на датасете Заповедник")
    parser.add_argument("--coco_path", type=str, default="coco_dataset",
                       help="Путь к COCO датасету")
    parser.add_argument("--output_dir", type=str, default="output_dino",
                       help="Директория для сохранения результатов")
    parser.add_argument("--epochs", type=int, default=36,
                       help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (для RTX 5090 рекомендуется 2)")
    parser.add_argument("--skip_compile", action="store_true",
                       help="Пропустить компиляцию CUDA ops")
    parser.add_argument("--convert_dataset", action="store_true",
                       help="Конвертировать YOLO датасет в COCO")
    parser.add_argument("--resume", type=str, default="",
                       help="Путь к чекпоинту для продолжения обучения")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ПОДГОТОВКА К ОБУЧЕНИЮ DINO")
    logger.info("=" * 60)
    
    # Проверка DINO
    if not check_dino():
        return 1
    
    # Конвертация датасета если нужно
    if args.convert_dataset or not Path(args.coco_path).exists():
        logger.info("Конвертация датасета YOLO -> COCO...")
        from yolo_to_coco_converter import convert_yolo_to_coco
        result = convert_yolo_to_coco("dataset", args.coco_path)
        if result is None:
            logger.error("Ошибка конвертации датасета")
            return 1
    
    # Проверка датасета
    if not check_coco_dataset(args.coco_path):
        logger.error("Датасет не найден. Запустите с --convert_dataset")
        return 1
    
    # Компиляция CUDA ops
    if not args.skip_compile:
        if not compile_cuda_ops():
            logger.warning("CUDA ops не скомпилированы, продолжаем без них")
    
    # Создание output директории
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Запуск обучения
    success = run_training(
        coco_path=args.coco_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

