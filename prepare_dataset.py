import os
import shutil
from pathlib import Path

# ============================================================
# SOURCE DATASET
# ============================================================

SOURCE = Path("dataset/yolo")

# ============================================================
# NEW DATASET
# ============================================================

DEST = Path("dataset/helmet_binary")

TRAIN_IMAGES = DEST / "images" / "train"
VAL_IMAGES = DEST / "images" / "val"

TRAIN_LABELS = DEST / "labels" / "train"
VAL_LABELS = DEST / "labels" / "val"

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS MAPPING
# ============================================================

# Original:
#
# 0 = Helmet
# 1 = Person
# 2 = Head
#
# New:
#
# 0 = Helmet
# 1 = NoHelmet
#
# Person annotations are ignored.

def convert_label(source_label, destination_label):

    if not source_label.exists():
        return

    output_lines = []

    with open(source_label, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            # ------------------------------------------------
            # Helmet
            # ------------------------------------------------

            if class_id == 0:

                # Helmet stays class 0
                parts[0] = "0"

                output_lines.append(
                    " ".join(parts)
                )

            # ------------------------------------------------
            # Person
            # ------------------------------------------------

            elif class_id == 1:

                # Ignore Person
                continue

            # ------------------------------------------------
            # Head
            # ------------------------------------------------

            elif class_id == 2:

                # Head becomes NoHelmet
                parts[0] = "1"

                output_lines.append(
                    " ".join(parts)
                )

    with open(destination_label, "w") as f:

        f.write(
            "\n".join(output_lines)
        )


# ============================================================
# PROCESS TRAIN + VALIDATION
# ============================================================

for split in ["train", "val"]:

    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    if not source_images.exists():

        print(
            f"WARNING: {source_images} not found"
        )

        continue

    if not source_labels.exists():

        print(
            f"WARNING: {source_labels} not found"
        )

        continue

    destination_images = (
        TRAIN_IMAGES
        if split == "train"
        else VAL_IMAGES
    )

    destination_labels = (
        TRAIN_LABELS
        if split == "train"
        else VAL_LABELS
    )

    image_files = list(
        source_images.glob("*")
    )

    print()
    print("=" * 60)
    print(f"Processing {split}")
    print("=" * 60)

    converted = 0

    for image_path in image_files:

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:
            continue

        label_path = (
            source_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        # Copy image
        shutil.copy2(
            image_path,
            destination_images
            / image_path.name
        )

        # Convert label
        convert_label(
            label_path,
            destination_labels
            / label_path.name
        )

        converted += 1

    print(
        f"Converted {converted} images"
    )


# ============================================================
# CREATE DATA YAML
# ============================================================

yaml_path = DEST / "data.yaml"

yaml_content = f"""path: {DEST.resolve().as_posix()}
train: images/train
val: images/val

nc: 2

names:
- Helmet
- NoHelmet
"""

with open(yaml_path, "w") as f:

    f.write(yaml_content)

print()
print("=" * 60)
print("BINARY DATASET READY")
print("=" * 60)

print(f"Dataset: {DEST.resolve()}")
print(f"YAML:    {yaml_path.resolve()}")

print()
print("Classes:")
print("0 = Helmet")
print("1 = NoHelmet")