from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
YOLO_ROOT = DATASET_ROOT / "yolo"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# Dataset
KAGGLE_DATASET = "andrewmvd/hard-hat-detection"
CLASS_NAMES = ["Helmet", "Person", "Head"]

# Training settings - designed for a single RTX 4060 / 4060 Laptop GPU
MODEL_NAME = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH = 8
DEVICE = 0
WORKERS = 2
PATIENCE = 15
