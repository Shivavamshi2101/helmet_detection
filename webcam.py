import cv2
from pathlib import Path
from ultralytics import YOLO
from config import ROOT

MODEL = ROOT / "runs" / "helmet_detection" / "weights" / "best.pt"

def main():
    model = YOLO(str(MODEL))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=0.35, iou=0.5, device=0, verbose=False)
        r = results[0]
        helmets = 0

        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = r.names[cls_id]
            if name.lower() != "helmet":
                continue
            helmets += 1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2-x1) * (y2-y1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(frame, f"Helmet {conf:.2f} | {area}px2",
                        (x1, max(25,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        cv2.putText(frame, f"Helmets: {helmets}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.imshow("Helmet Detection - RTX 4060", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
