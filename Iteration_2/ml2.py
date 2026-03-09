import torch
import cv2
import os
import math
from ultralytics import YOLO


class MLNode:
    def __init__(self, target_class_id=0, use_tensorrt=False,
                 save_dir="target_detections", dedup_radius_m=10.0):
        """
        Args:
            target_class_id : YOLO class index to treat as the target.
            use_tensorrt    : Load .engine instead of .pt model.
            save_dir        : Where to save annotated target images.
            dedup_radius_m  : Minimum distance (metres) between two saved targets.
                              Any detection within this radius of an already-saved
                              target is silently dropped as a duplicate.
                              Default 10 m works well for typical drone altitudes.
        """
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt    = use_tensorrt
        self.target_class_id = target_class_id
        self.save_dir        = save_dir
        self.dedup_radius_m  = dedup_radius_m

        # Instance-level counter so filenames never collide across frames
        self.detection_count = 0

        # GPS deduplication registry: list of (lat, lon) of already-saved targets
        self.saved_target_gps = []

        # Always create the output directory — was commented out, causing silent
        # failures where imwrite succeeds but the file is written to /dev/null
        os.makedirs(self.save_dir, exist_ok=True)

        if self.use_tensorrt:
            print("[ML] HIGH PERFORMANCE MODE: Loading TensorRT YOLO Engine...")
            self.model = YOLO("best_L_mer.engine")
        else:
            print(f"[ML] FALLBACK MODE: Loading PyTorch YOLO model to {self.device}...")
            self.model = YOLO("best_L_mer.pt")
            self.model.to(self.device)

    # ==========================================
    # GPS DEDUPLICATION HELPERS
    # ==========================================
    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        """Returns the great-circle distance in metres between two GPS points."""
        R = 6_371_000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def _is_duplicate_target(self, lat, lon):
        """
        Returns True if there is already a saved target within dedup_radius_m
        of the given GPS coordinate.

        This prevents the system from saving hundreds of images of the same
        physical object as the drone circles overhead or flies past multiple times.
        """
        for saved_lat, saved_lon in self.saved_target_gps:
            if self._haversine_m(lat, lon, saved_lat, saved_lon) < self.dedup_radius_m:
                return True
        return False

    # ==========================================
    # MAIN PROCESSING
    # ==========================================
    def process_frame(self, frame_np, gps_coords):
        """
        Runs YOLO inference on one frame and saves annotated images for
        any NEW targets (i.e. not already seen within dedup_radius_m).

        Returns the annotated frame (or a clean copy if nothing detected).
        """
        results = self.model(frame_np, conf=0.85, verbose=False)[0]

        annotated_frame = frame_np.copy()

        if results.boxes is None or len(results.boxes) == 0:
            return annotated_frame

        # Draw all boxes regardless of class for the live feed
        annotated_frame = results.plot()

        lat, lon, alt = gps_coords

        for cls, conf in zip(results.boxes.cls.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()):

            if int(cls) != self.target_class_id or conf < 0.85:
                continue

            # --- GPS DEDUPLICATION FILTER ---
            if self._is_duplicate_target(lat, lon):
                print(f"[ML] Duplicate target at ({lat:.6f}, {lon:.6f}) "
                      f"within {self.dedup_radius_m}m of known target — skipped.")
                continue

            # --- NEW TARGET: save and register ---
            print(f"[ML] NEW Target! Conf: {conf:.2f}  "
                  f"GPS: ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")

            filename = (f"target_lat_{lat:.6f}_lon_{lon:.6f}"
                        f"_alt_{alt:.1f}_{self.detection_count}.jpg")
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, annotated_frame)

            # Register this GPS so future detections nearby are filtered out
            self.saved_target_gps.append((lat, lon))
            self.detection_count += 1

        return annotated_frame


# ==========================================
# INDEPENDENT WEBCAM TESTING BLOCK
# ==========================================
if __name__ == '__main__':
    detector = MLNode(target_class_id=0, use_tensorrt=False, dedup_radius_m=10.0)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detector.process_frame(frame, (0.0, 0.0, 0.0))
        cv2.imshow("Test", annotated)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()