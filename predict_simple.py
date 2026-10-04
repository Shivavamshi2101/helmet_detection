import cv2
from ultralytics import YOLO
import os
import sys


MODEL_PATH = "runs/detect/runs/helmet_binary/weights/best.pt"

# Confidence threshold
CONFIDENCE = 0.90


def detect_helmet(image_path):

    print("=" * 60)
    print("HELMET DETECTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Check image
    # ---------------------------------------------------------
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found:")
        print(image_path)
        return

    # ---------------------------------------------------------
    # Check model
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found:")
        print(MODEL_PATH)
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print("Device: NVIDIA RTX 4060")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = YOLO(MODEL_PATH)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    for result in results:

        img = result.orig_img.copy()

        helmet_count = 0
        nohelmet_count = 0
        person_count = 0

        print("\nDETECTIONS")
        print("-" * 60)

        if result.boxes is None or len(result.boxes) == 0:

            print("No objects detected.")

        else:

            for box in result.boxes:

                # Class ID
                cls_id = int(box.cls[0].item())

                # Class name
                class_name = str(result.names[cls_id]).lower()

                # Confidence
                confidence = float(box.conf[0].item())

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # -------------------------------------------------
                # Bounding box dimensions
                # -------------------------------------------------

                width = x2 - x1
                height = y2 - y1

                area = width * height

                # -------------------------------------------------
                # Helmet
                # -------------------------------------------------

                if class_name == "helmet":

                    helmet_count += 1

                    color = (0, 255, 0)

                    label = (
                        f"HELMET {confidence:.2f} "
                        f"| Area: {area}px2"
                    )

                    print("HELMET DETECTED")
                    print(f"Confidence : {confidence:.3f}")
                    print(
                        f"Bounding Box: "
                        f"[{x1}, {y1}, {x2}, {y2}]"
                    )
                    print(f"Width      : {width}px")
                    print(f"Height     : {height}px")
                    print(f"Area       : {area}px2")
                    print()

                # -------------------------------------------------
                # NoHelmet / Head
                # -------------------------------------------------

                elif class_name in ["nohelmet", "head"]:

                    nohelmet_count += 1

                    color = (0, 0, 255)

                    label = (
                        f"NO HELMET {confidence:.2f}"
                    )

                    print("NO HELMET DETECTED")
                    print(f"Confidence : {confidence:.3f}")
                    print(
                        f"Bounding Box: "
                        f"[{x1}, {y1}, {x2}, {y2}]"
                    )
                    print()

                # -------------------------------------------------
                # Person
                # -------------------------------------------------

                elif class_name == "person":

                    person_count += 1

                    color = (255, 0, 0)

                    label = (
                        f"PERSON {confidence:.2f}"
                    )

                # -------------------------------------------------
                # Unknown
                # -------------------------------------------------

                else:

                    color = (255, 255, 0)

                    label = (
                        f"{class_name.upper()} "
                        f"{confidence:.2f}"
                    )

                # -------------------------------------------------
                # Draw bounding box
                # -------------------------------------------------

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # -------------------------------------------------
                # Label background
                # -------------------------------------------------

                text_y = max(y1 - 10, 20)

                cv2.putText(
                    img,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"Helmets detected : {helmet_count}")
        print(f"NoHelmets detected: {nohelmet_count}")
        print(f"Persons detected : {person_count}")

        # ---------------------------------------------------------
        # Save output
        # ---------------------------------------------------------

        os.makedirs("output", exist_ok=True)

        filename = os.path.basename(image_path)

        name, ext = os.path.splitext(filename)

        output_path = os.path.join(
            "output",
            f"{name}_detected{ext}"
        )

        cv2.imwrite(
            output_path,
            img
        )

        print()
        print(f"Output saved to:")
        print(output_path)

        # ---------------------------------------------------------
        # Display
        # ---------------------------------------------------------

        cv2.imshow(
            "Helmet Detection",
            img
        )

        print("\nPress any key to close.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        image_path = sys.argv[1]

    else:

        image_path = input(
            "Enter image path: "
        ).strip()

    detect_helmet(image_path)