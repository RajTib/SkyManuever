import torch
import cv2
import numpy as np
import os
import glob


class MapStitcher:
    def __init__(self, map_dir="map_tiles", output_file="final_map.jpg"):
        self.map_dir = map_dir
        self.output_file = output_file

        # Dynamic canvas — starts as None, grows as drone flies
        self.canvas = None
        self.canvas_w = 0
        self.canvas_h = 0

        # H_global starts as pure identity
        self.H_global = np.eye(3, dtype=np.float64)
        self.is_first_frame = True

    # ==========================================
    # CORE: DYNAMIC CANVAS EXPANSION + BLENDING
    # ==========================================
    def _place_frame(self, warped_frame):
        """
        Checks if the new warped frame fits inside the current canvas.
        If not, expands the canvas in whatever direction is needed.
        Then blends the new frame in.
        """
        if self.canvas is None:
            print("-> [SKIP] Canvas not initialized yet.")
            return
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        non_black = np.where(gray > 10)

        # If the frame is completely empty/black, skip it
        if len(non_black[0]) == 0:
            print("-> [SKIP] Warped frame is empty, ignoring.")
            return

        # Bounding box of the new frame's actual content
        y_min, y_max = int(non_black[0].min()), int(non_black[0].max())
        x_min, x_max = int(non_black[1].min()), int(non_black[1].max())

        # Calculate how much padding is needed in each direction
        pad_top    = max(0, -y_min)
        pad_left   = max(0, -x_min)
        pad_bottom = max(0, y_max - self.canvas_h + 1)
        pad_right  = max(0, x_max - self.canvas_w + 1)

        # If any expansion is needed, grow the canvas
        if any([pad_top, pad_left, pad_bottom, pad_right]):
            print(f"-> [EXPAND] Canvas growing by T:{pad_top} B:{pad_bottom} L:{pad_left} R:{pad_right}")
            self.canvas = cv2.copyMakeBorder(
                self.canvas,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            self.canvas_h, self.canvas_w = self.canvas.shape[:2]
            print(f"-> [CANVAS] New size: {self.canvas_w}x{self.canvas_h}")

            # Shift H_global to account for the canvas origin moving
            shift = np.array([
                [1, 0, pad_left],
                [0, 1, pad_top],
                [0, 0, 1]
            ], dtype=np.float64)
            self.H_global = shift @ self.H_global

            # Re-warp the frame with the updated H_global
            warped_frame = cv2.warpPerspective(
                warped_frame,
                shift,
                (self.canvas_w, self.canvas_h)
            )

        # Blend: only write pixels where the new frame has real content
        gray_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        mask_2d = gray_warped > 10
        mask = np.stack([mask_2d, mask_2d, mask_2d], axis=2)
        self.canvas[mask] = warped_frame[mask]

    # ==========================================
    # MAIN RUN LOOP
    # ==========================================
    def run(self):
        print("--- Starting Post-Flight Map Stitching ---")

        # Find and sort all .pt chunk files numerically
        chunk_files = sorted(
            glob.glob(os.path.join(self.map_dir, "*.pt")),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        if not chunk_files:
            print(f"[ERROR] No map data found in '{self.map_dir}'.")
            return

        total_frames_stitched = 0

        for chunk_file in chunk_files:
            print(f"\n[STITCH] Unpacking chunk: {chunk_file}...")
            buffer = torch.load(chunk_file, map_location='cpu', weights_only=False)

            for item in buffer:
                frame   = item['image_1']
                H_local = item['homography']
                inliers = item.get('inliers', '?')

                print(f"-> Processing frame | Inliers: {inliers}")

                # ---- FIRST FRAME: anchor to canvas center ----
                if self.is_first_frame:
                    print("-> Anchoring first frame.")
                    h, w = frame.shape[:2]

                    # Canvas starts exactly the size of the first frame
                    self.canvas_w = w
                    self.canvas_h = h

                    # Place first frame dead center
                    cx, cy = w // 2, h // 2
                    self.H_global = np.array([
                        [1.0, 0.0, cx],
                        [0.0, 1.0, cy],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)

                    self.canvas = cv2.warpPerspective(frame, self.H_global, (self.canvas_w, self.canvas_h))
                    self.canvas_h, self.canvas_w = self.canvas.shape[:2]
                    self.is_first_frame = False
                    total_frames_stitched += 1
                    continue

                # ---- SUBSEQUENT FRAMES: chain homographies ----

                # Invert H_local: map.py gives us Frame0->Frame1,
                # we need Frame1->Frame0 to warp backwards onto the canvas
                try:
                    H_inv = np.linalg.inv(H_local)
                except np.linalg.LinAlgError:
                    print("         -> [WARNING] Singular homography matrix, skipping frame.")
                    continue

                # Chain: accumulate all steps from canvas origin to this frame
                self.H_global = self.H_global @ H_inv

                # Warp the new frame to current canvas size
                warped_frame = cv2.warpPerspective(
                    frame,
                    self.H_global,
                    (self.canvas_w, self.canvas_h)
                )

                # Place it (expands canvas if needed)
                self._place_frame(warped_frame)
                total_frames_stitched += 1

        print(f"\n[STITCH] Successfully stitched {total_frames_stitched} frames.")
        print(f"[STITCH] Final canvas size: {self.canvas_w}x{self.canvas_h}")
        self._crop_and_save()

    # ==========================================
    # CROP + SAVE
    # ==========================================
    def _crop_and_save(self):
        if self.canvas is None:
            print("[STICH] [ERROR] Canvas is None — no frames were stitched. Check map_tiles folder.")
            return
        """Trims black borders and saves the final map."""
        print("\n[STITCH] Trimming black borders...")

        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped = self.canvas[y:y+h, x:x+w]
            print(f"[STITCH] Final map resolution: {w}x{h} pixels.")
            print(f"[STITCH] Saving to '{self.output_file}'...")
            cv2.imwrite(self.output_file, cropped)
            print("[STITCH] ✅ Map generation complete!")
        else:
            print("[STITCH] [ERROR] Canvas is completely empty. Nothing to save.")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    stitcher = MapStitcher()
    stitcher.run()
