"""
detect_webcam.py

Real-time fire/smoke detection from a laptop webcam using a trained
YOLOv8 model.

Usage:
    python detection/detect_webcam.py
    python detection/detect_webcam.py --weights training/runs/fire_smoke_yolov8/weights/best.pt --conf 0.5

Press 'q' in the video window to quit.
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="training/runs/fire_smoke_yolov8/weights/best.pt")
    parser.add_argument("--source", default="0",
                         help="camera index (0 = default laptop camera) or a video file path")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source}")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera, stopping.")
            break

        results = model.predict(
            frame, conf=args.conf, device=args.device, imgsz=args.imgsz, verbose=False
        )
        result = results[0]
        annotated = result.plot()

        danger = len(result.boxes) > 0
        if danger:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(annotated, "DANGER: FIRE/SMOKE DETECTED", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Fire/Smoke Detection - Danger Glasses", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
