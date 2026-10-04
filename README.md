# Helmet Detection — RTX 4060 Local Project

This project trains an Ultralytics YOLO object detector on:
`andrewmvd/hard-hat-detection`

The dataset contains 5,000 images with Pascal VOC XML bounding boxes and 3 classes:
`Helmet`, `Person`, `Head`.

## What this project does

1. Downloads the Kaggle dataset automatically with `kagglehub`.
2. Finds the dataset images/XML annotations automatically.
3. Converts Pascal VOC XML bounding boxes to YOLO format.
4. Creates an 80/20 train/validation split.
5. Trains a pretrained YOLO model using your NVIDIA RTX 4060 GPU (`device=0`).
6. Saves the best model as `runs/helmet_detection/weights/best.pt`.
7. Runs helmet-only detection on images.
8. Draws a bounding box around every detected helmet and writes:
   `Helmet | confidence | x1,y1,x2,y2 | area: N px²`
9. Can also run on a webcam or video.

## IMPORTANT: RTX 4060 PyTorch installation

Open PowerShell in this project folder.

Create an environment:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build. The exact command can change with the current PyTorch release, so use the official PyTorch selector:
https://pytorch.org/get-started/locally/

Then:
```powershell
pip install -r requirements.txt
```

Verify GPU:
```powershell
python check_gpu.py
```

You should see:
`CUDA available: True`
and
`NVIDIA GeForce RTX 4060 Laptop GPU` (or your RTX 4060 model).

## One-command full setup + training

```powershell
python train.py
```

The script automatically downloads and prepares the dataset, then trains.

Default settings are suitable for an RTX 4060 laptop:
- model: `yolo11n.pt`
- image size: 640
- epochs: 50
- batch: 8
- device: 0
- workers: 2 (Windows-friendly)
- AMP: enabled

If you have enough VRAM, increase `BATCH` in `config.py` from 8 to 12 or 16.

## Detect an image

Put an image into:
`input/`

Then:
```powershell
python detect.py --source input/my_image.jpg
```

Output:
`output/my_image_detected.jpg`

The detector only displays the `Helmet` class by default.

## Detect all three classes

```powershell
python detect.py --source input/my_image.jpg --all-classes
```

## Webcam

```powershell
python webcam.py
```

Press `Q` to exit.

## Video

```powershell
python detect.py --source input/video.mp4
```

## Area of each helmet box

For every detection:
`area = (x2 - x1) * (y2 - y1)`

The pixel area is printed on the image and in the terminal. This is the area of the rectangular detected bounding box in pixels², not the exact physical surface area of the helmet.

## Model output

After training, use:
`runs/helmet_detection/weights/best.pt`

The validation/training plots are also saved under:
`runs/helmet_detection/`

## Dataset source

Kaggle:
https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

The source dataset is reported as 5,000 images, Pascal VOC annotations, with Helmet/Person/Head classes.
