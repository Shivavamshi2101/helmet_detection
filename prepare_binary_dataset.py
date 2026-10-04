from pathlib import Path
import shutil

SOURCE = Path("dataset/yolo")
DEST = Path("dataset/helmet_binary")

# Create directories
for split in ["train", "val"]:
    (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

# Original classes:
# 0 = Helmet
# 1 = Person
# 2 = Head
#
# New classes:
# 0 = Helmet
# 1 = NoHelmet

stats = {
    "train_images": 0,
    "val_images": 0,
    "helmet_boxes": 0,
    "nohelmet_boxes": 0,
    "person_boxes_ignored": 0,
}

for split in ["train", "val"]:

    src_images = SOURCE / "images" / split
    src_labels = SOURCE / "labels" / split

    dst_images = DEST / "images" / split
    dst_labels = DEST / "labels" / split

    if not src_images.exists():
        print(f"ERROR: Missing {src_images}")
        continue

    print(f"\nProcessing {split}...")

    for image_path in src_images.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        label_path = src_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        new_labels = []

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                # Helmet -> Helmet
                if class_id == 0:

                    parts[0] = "0"
                    new_labels.append(" ".join(parts))

                    stats["helmet_boxes"] += 1

                # Person -> ignore
                elif class_id == 1:

                    stats["person_boxes_ignored"] += 1

                # Head -> NoHelmet
                elif class_id == 2:

                    parts[0] = "1"
                    new_labels.append(" ".join(parts))

                    stats["nohelmet_boxes"] += 1

        # Copy image
        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

        # Write new labels
        with open(
            dst_labels / label_path.name,
            "w"
        ) as f:

            f.write("\n".join(new_labels))

        stats[f"{split}_images"] += 1

    print(
        f"{split}: "
        f"{stats[f'{split}_images']} images"
    )


# Create data.yaml
yaml_path = DEST / "data.yaml"

yaml_text = f"""path: {DEST.resolve().as_posix()}
train: images/train
val: images/val

nc: 2

names:
- Helmet
- NoHelmet
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text)


print("\n" + "=" * 60)
print("BINARY DATASET CREATED")
print("=" * 60)

print(f"Train images       : {stats['train_images']}")
print(f"Validation images  : {stats['val_images']}")
print(f"Helmet boxes       : {stats['helmet_boxes']}")
print(f"NoHelmet boxes     : {stats['nohelmet_boxes']}")
print(f"Person boxes       : {stats['person_boxes_ignored']} ignored")

print()
print(f"Dataset: {DEST.resolve()}")
print(f"YAML   : {yaml_path.resolve()}")

print()
print("Classes:")
print("0 = Helmet")
print("1 = NoHelmet")