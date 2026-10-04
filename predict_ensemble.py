import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

MODEL_1_PATH = "runs/helmet_detection/weights/best.pt"
MODEL_2_PATH = "runs/detect/runs/helmet_binary/weights/best.pt"

CONFIDENCE = 0.90
IOU_THRESHOLD = 0.50

def detect_ensemble(image_path):
    print("=" * 60)
    print("HELMET DETECTION (ENSEMBLE)")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found:")
        print(image_path)
        return

    if not os.path.exists(MODEL_1_PATH) or not os.path.exists(MODEL_2_PATH):
        print("ERROR: One or both models not found. Please check paths:")
        print(MODEL_1_PATH)
        print(MODEL_2_PATH)
        return

    print("Loading models...")
    model1 = YOLO(MODEL_1_PATH)
    model2 = YOLO(MODEL_2_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image.")
        return

    print("Running predictions...")
    res1 = model1.predict(source=image_path, conf=CONFIDENCE, device=0, verbose=False)[0]
    res2 = model2.predict(source=image_path, conf=CONFIDENCE, device=0, verbose=False)[0]

    boxes_list = []
    scores_list = []
    class_names_list = []

    # Model 1 processing (0: Helmet, 1: Person, 2: Head)
    if res1.boxes is not None:
        for box in res1.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if cls_id == 0: name = "helmet"
            elif cls_id == 1: name = "person"
            elif cls_id == 2: name = "head"
            else: name = "unknown"
            
            boxes_list.append([x1, y1, x2 - x1, y2 - y1]) # format for NMSBoxes is [x, y, w, h]
            scores_list.append(conf)
            class_names_list.append(name)

    # Model 2 processing (0: Helmet, 1: NoHelmet/Head)
    if res2.boxes is not None:
        for box in res2.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if cls_id == 0: name = "helmet"
            elif cls_id == 1: name = "head"
            else: name = "unknown"
            
            boxes_list.append([x1, y1, x2 - x1, y2 - y1])
            scores_list.append(conf)
            class_names_list.append(name)

    # Apply NMS (Non-Maximum Suppression) across both models
    final_boxes = []
    if len(boxes_list) > 0:
        # cv2.dnn.NMSBoxes expects boxes in [x, y, w, h] format
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, CONFIDENCE, IOU_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append({
                    "box": boxes_list[i],
                    "score": scores_list[i],
                    "name": class_names_list[i]
                })

    print("\nDETECTIONS")
    print("-" * 60)
    
    helmet_count = 0
    head_count = 0
    person_count = 0

    if not final_boxes:
        print("No objects detected.")
    else:
        for det in final_boxes:
            x, y, w, h = det["box"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            name = det["name"]
            conf = det["score"]
            area = w * h

            if name == "helmet":
                helmet_count += 1
                color = (0, 255, 0)
                label = f"HELMET {conf:.2f} | Area: {area}px2"
                print(f"HELMET DETECTED (Conf: {conf:.3f})")
            elif name == "head":
                head_count += 1
                color = (0, 0, 255)
                label = f"NO HELMET / HEAD {conf:.2f}"
                print(f"HEAD DETECTED (Conf: {conf:.3f})")
            elif name == "person":
                person_count += 1
                color = (255, 0, 0)
                label = f"PERSON {conf:.2f}"
                print(f"PERSON DETECTED (Conf: {conf:.3f})")
            else:
                color = (255, 255, 0)
                label = f"{name.upper()} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_y = max(y1 - 10, 20)
            cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Helmets detected : {helmet_count}")
    print(f"Heads detected   : {head_count}")
    print(f"Persons detected : {person_count}")

    os.makedirs("output", exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join("output", f"{name}_ensemble{ext}")
    
    cv2.imwrite(output_path, img)
    print(f"\nOutput saved to:\n{output_path}")

    cv2.imshow("Ensemble Detection", img)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image path: ").strip()
        
    # Strip quotes if copied from Windows terminal
    if img_path.startswith('"') and img_path.endswith('"'):
        img_path = img_path[1:-1]
        
    detect_ensemble(img_path)
