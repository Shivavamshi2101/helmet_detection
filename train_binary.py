from ultralytics import YOLO
import torch


def main():

    print("=" * 60)
    print("HELMET / NO-HELMET TRAINING")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: CUDA unavailable")
        device = "cpu"

    # Load pretrained YOLO11 Small
    model = YOLO("yolo11s.pt")

    # Train
    model.train(
        data="dataset/helmet_binary/data.yaml",

        epochs=100,

        imgsz=640,

        batch=16,

        device=device,

        workers=4,

        project="runs",

        name="helmet_binary",

        pretrained=True,

        optimizer="auto",

        patience=20,

        cache=False,

        amp=True,

        plots=True,

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,

        translate=0.1,

        scale=0.5,

        fliplr=0.5,

        mosaic=1.0,

        mixup=0.1,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best model:"
    )

    print(
        "runs/helmet_binary/weights/best.pt"
    )


if __name__ == "__main__":
    main()