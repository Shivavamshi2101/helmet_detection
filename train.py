from pathlib import Path
import torch
from ultralytics import YOLO
from config import *

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This project is configured to train on RTX 4060 (device=0). "
            "Run check_gpu.py and install CUDA-enabled PyTorch."
        )

    # Prepare dataset first.
    from prepare_dataset import main as prepare_main
    if not (YOLO_ROOT / "data.yaml").exists():
        prepare_main()

    print("\nUsing GPU:", torch.cuda.get_device_name(DEVICE))
    print("CUDA:", torch.version.cuda)
    print("VRAM:", round(torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3, 2), "GB")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(YOLO_ROOT / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(ROOT / "runs"),
        name="helmet_detection",
        exist_ok=True,
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        verbose=True,
    )

    best = ROOT / "runs" / "helmet_detection" / "weights" / "best.pt"
    print("\nTraining finished.")
    print("Best model:", best)
    print("Use this model with: python detect.py --source input/your_image.jpg")

if __name__ == "__main__":
    main()
