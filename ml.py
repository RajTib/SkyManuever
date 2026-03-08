import torch
import cv2
import os
from ultralytics import YOLO



class MLNode:
    def __init__(self, target_class_id=0, use_tensorrt=False, save_dir="target_detections"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt
        self.target_class_id = target_class_id

        self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)

        # --- MODEL LOADING ---
        if self.use_tensorrt:
            print("[ML] 🚀 HIGH PERFORMANCE MODE: Loading TensorRT YOLO Engine...")
            self.model = YOLO("best_L_mer.engine")
        else:
            print(f"[ML] 🐢 FALLBACK MODE: Loading PyTorch YOLO model to {self.device}...")
            self.model = YOLO("best_L_mer.pt")
            self.model.to(self.device)

    def process_frame(self, frame_np, gps_coords):
        """Runs inference and returns annotated frame."""
        results = self.model(frame_np, conf=0.85, verbose=False)[0]

        count =0

        annotated_frame = frame_np.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            annotated_frame = results.plot()

            for cls, conf in zip(results.boxes.cls.cpu().numpy(),
                                 results.boxes.conf.cpu().numpy()):

                if int(cls) == self.target_class_id and conf >= 0.85:
                    lat, lon, alt = gps_coords
                    print(f"[ML] 🎯 Target Spotted! Conf: {conf:.2f}")

                    filename = f"target_lat_{lat:.6f}_lon_{lon:.6f}_alt_{alt:.1f}_{count}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, annotated_frame)
                    count += 1

        return annotated_frame


# ==========================================
# INDEPENDENT WEBCAM TESTING BLOCK
# ==========================================

if __name__ == '__main__':
    detector = MLNode(target_class_id=0, use_tensorrt=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detector.process_frame(frame, (0, 0, 0))
        cv2.imshow("Test", annotated)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 