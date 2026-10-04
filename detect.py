import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from config import ROOT, OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="Helmet detection with bounding boxes and box area.")
    parser.add_argument("--source", required=True, help="Image/video path, folder, webcam index, or URL")
    parser.add_argument("--model", default=str(ROOT / "runs" / "helmet_detection" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--all-classes", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If source is an image, provide a clear output filename.
    source = args.source
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=0,
        save=False,
        verbose=True
    )

    for result in results:
        img = result.orig_img.copy()
        names = result.names
        helmet_count = 0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = str(names[cls_id])

                if not args.all_classes and name.lower() != "helmet":
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = width * height

                if name.lower() == "helmet":
                    helmet_count += 1

                # Draw box.
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Label with confidence and pixel area.
                label = f"{name} {conf:.2f} | Area: {area}px2"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                text_y = max(y1 - 10, th + 5)
                cv2.rectangle(
                    img,
                    (x1, text_y - th - baseline),
                    (x1 + tw + 6, text_y + 4),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    img, label, (x1 + 3, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
                )

                print(
                    f"{name}: confidence={conf:.3f}, "
                    f"box=({x1},{y1},{x2},{y2}), "
                    f"width={width}px, height={height}px, area={area}px²"
                )

        # Summary.
        cv2.putText(
            img, f"Helmets detected: {helmet_count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA
        )

        # Save image results.
        src_path = Path(source)
        if src_path.exists() and src_path.is_file():
            out = OUTPUT_DIR / f"{src_path.stem}_detected{src_path.suffix}"
        else:
            out = OUTPUT_DIR / "detection_result.jpg"

        cv2.imwrite(str(out), img)
        print(f"Saved result: {out}")

if __name__ == "__main__":
    main()
