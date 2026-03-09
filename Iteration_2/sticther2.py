import torch
import cv2
import numpy as np
import os
import gc
import glob
import math


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

        # ---- GPS-ASSISTED DRIFT CORRECTION ----
        # canvas_cx/cy: the canvas pixel that corresponds to the GPS origin
        # (i.e. where the first frame's centre sits). Updated as canvas expands.
        self.canvas_cx = 0
        self.canvas_cy = 0

        self.origin_gps = None          # (lat, lon) of the very first frame
        self.pixels_per_meter = None    # Estimated from data (PPM)
        self._ppm_samples = []          # Raw PPM estimates from early frames
        self.MIN_PPM_SAMPLES = 5        # Frames needed before we trust the PPM estimate
        self.MAX_DRIFT_PIXELS = 15.0    # Drift threshold before GPS correction fires (px)
        self.GPS_CORRECTION_STRENGTH = 0.7  # How aggressively we snap toward GPS truth (0-1)

    # ==========================================
    # GPS UTILITIES
    # ==========================================
    def _haversine_m(self, lat1, lon1, lat2, lon2):
        """Returns the distance in metres between two GPS coordinates."""
        R = 6_371_000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def _gps_to_canvas_pixels(self, lat, lon):
        """
        Convert an absolute GPS coordinate into its expected canvas pixel position
        using the stored pixels-per-metre scale and the origin anchor.
        Returns (px, py) or None if PPM has not been estimated yet.
        """
        if self.origin_gps is None or self.pixels_per_meter is None:
            return None

        R = 6_371_000.0
        dlat = math.radians(lat - self.origin_gps[0])
        dlon = math.radians(lon - self.origin_gps[1])

        north_m = R * dlat
        east_m  = R * dlon * math.cos(math.radians(self.origin_gps[0]))

        # Canvas X increases eastward; canvas Y increases southward (image convention)
        px = self.canvas_cx + east_m  * self.pixels_per_meter
        py = self.canvas_cy - north_m * self.pixels_per_meter
        return px, py

    def _update_ppm_estimate(self, gps_0, gps_1, H_local):
        """
        Accumulate one pixels-per-metre sample from the GPS displacement
        and the corresponding homography translation.  Once we have enough
        samples we lock in the median as our PPM scale.
        """
        if self.pixels_per_meter is not None:
            return  # Already locked in

        if gps_0 is None or gps_1 is None:
            return

        gps_dist_m = self._haversine_m(gps_0[0], gps_0[1], gps_1[0], gps_1[1])
        if gps_dist_m < 0.3:        # Drone barely moved — skip noisy sample
            return

        # Pixel displacement: how far the origin of frame(N-1) moved under H_local
        # H_local maps frame(N-1) -> frame(N), so [0,0,1] tells us where origin went
        origin_mapped = H_local @ np.array([0.0, 0.0, 1.0])
        px = origin_mapped[0] / origin_mapped[2]
        py = origin_mapped[1] / origin_mapped[2]
        pixel_dist = math.sqrt(px ** 2 + py ** 2)

        if pixel_dist < 2.0:        # Sub-pixel movement — skip
            return

        self._ppm_samples.append(pixel_dist / gps_dist_m)

        if len(self._ppm_samples) >= self.MIN_PPM_SAMPLES:
            self.pixels_per_meter = float(np.median(self._ppm_samples))
            print(f"-> [GPS] PPM scale locked: {self.pixels_per_meter:.2f} px/m "
                  f"(from {len(self._ppm_samples)} samples)")

    def _apply_gps_drift_correction(self, gps_1, frame_shape):
        """
        Compares where H_global places the current frame centre on the canvas
        against where GPS says it should be.  If drift exceeds MAX_DRIFT_PIXELS,
        applies a weighted translation correction to H_global — preserving
        the rotation and scale components from the homography chain.
        """
        if self.pixels_per_meter is None or gps_1 is None:
            return

        gps_result = self._gps_to_canvas_pixels(gps_1[0], gps_1[1])
        if gps_result is None:
            return
        gps_cx, gps_cy = gps_result

        # Where does H_global currently place the centre of this frame?
        h, w = frame_shape[:2]
        centre_h = self.H_global @ np.array([w / 2.0, h / 2.0, 1.0])
        homo_cx = centre_h[0] / centre_h[2]
        homo_cy = centre_h[1] / centre_h[2]

        drift_x   = gps_cx - homo_cx
        drift_y   = gps_cy - homo_cy
        drift_mag = math.sqrt(drift_x ** 2 + drift_y ** 2)

        if drift_mag > self.MAX_DRIFT_PIXELS:
            corr_x = self.GPS_CORRECTION_STRENGTH * drift_x
            corr_y = self.GPS_CORRECTION_STRENGTH * drift_y
            print(f"-> [GPS CORRECTION] Drift {drift_mag:.1f}px "
                  f"correcting by ({corr_x:.1f}, {corr_y:.1f})px")
            correction = np.array([
                [1.0, 0.0, corr_x],
                [0.0, 1.0, corr_y],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            self.H_global = correction @ self.H_global

    # ==========================================
    # HOMOGRAPHY SANITY CHECK
    # ==========================================
    @staticmethod
    def _is_valid_homography(H):
        """
        Reject degenerate or physically implausible homographies before
        they corrupt the accumulated warp chain.

        Checks:
          - Matrix is non-singular and orientation-preserving (det > 0)
          - Scale change between frames is not extreme (0.4x - 2.5x)
          - Perspective coefficients are small (expected for nadir drone camera)
        """
        if H is None:
            return False
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            return False
        scale = math.sqrt(abs(det))
        if not (0.4 < scale < 2.5):
            return False
        if abs(H[2, 0]) > 0.005 or abs(H[2, 1]) > 0.005:
            return False
        return True

    # ==========================================
    # CORE: DYNAMIC CANVAS EXPANSION + BLENDING
    # ==========================================
    def _place_frame(self, warped_frame):
        """
        Checks if the new warped frame fits inside the current canvas.
        Expands the canvas in whatever direction is needed, then blends
        the new frame in using a binary content mask.
        """
        if self.canvas is None:
            print("-> [SKIP] Canvas not initialized yet.")
            return

        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        non_black = np.where(gray > 10)

        if len(non_black[0]) == 0:
            print("-> [SKIP] Warped frame is empty, ignoring.")
            return

        y_min, y_max = int(non_black[0].min()), int(non_black[0].max())
        x_min, x_max = int(non_black[1].min()), int(non_black[1].max())

        pad_top    = max(0, -y_min)
        pad_left   = max(0, -x_min)
        pad_bottom = max(0, y_max - self.canvas_h + 1)
        pad_right  = max(0, x_max - self.canvas_w + 1)

        if any([pad_top, pad_left, pad_bottom, pad_right]):
            print(f"-> [EXPAND] Canvas growing by T:{pad_top} B:{pad_bottom} "
                  f"L:{pad_left} R:{pad_right}")
            self.canvas = cv2.copyMakeBorder(
                self.canvas,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            self.canvas_h, self.canvas_w = self.canvas.shape[:2]
            print(f"-> [CANVAS] New size: {self.canvas_w}x{self.canvas_h}")

            # Shift H_global to account for the canvas origin moving
            shift = np.array([
                [1.0, 0.0, float(pad_left)],
                [0.0, 1.0, float(pad_top)],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            self.H_global = shift @ self.H_global

            # GPS origin anchor moves with the canvas expansion
            self.canvas_cx += pad_left
            self.canvas_cy += pad_top

            # Re-warp the incoming frame with the updated transform
            warped_frame = cv2.warpPerspective(
                warped_frame, shift, (self.canvas_w, self.canvas_h)
            )

        # Blend: only overwrite pixels where the new frame has real content
        gray_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        mask_2d = gray_warped > 10
        mask = np.stack([mask_2d] * 3, axis=2)
        self.canvas[mask] = warped_frame[mask]

    # ==========================================
    # MAIN RUN LOOP
    # ==========================================
    def run(self):
        print("--- Starting Post-Flight Map Stitching ---")

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
                gps_0   = item.get('gps_0')   # GPS of previous frame  (lat, lon, alt)
                gps_1   = item.get('gps_1')   # GPS of this frame       (lat, lon, alt)

                print(f"-> Processing frame | Inliers: {inliers}")

                # ---- FIRST FRAME: anchor with identity transform ----
                if self.is_first_frame:
                    print("-> Anchoring first frame.")
                    h, w = frame.shape[:2]

                    # Canvas starts exactly the size of the first frame.
                    # Identity H_global -> frame maps 1-to-1 onto canvas with no clipping.
                    self.canvas_w  = w
                    self.canvas_h  = h
                    self.H_global  = np.eye(3, dtype=np.float64)
                    self.canvas    = frame.copy()

                    # GPS origin anchor sits at the centre of the first frame
                    self.canvas_cx = w // 2
                    self.canvas_cy = h // 2
                    if gps_1 is not None:
                        self.origin_gps = (gps_1[0], gps_1[1])
                        print(f"-> [GPS] Origin set: {self.origin_gps}")

                    self.is_first_frame = False
                    total_frames_stitched += 1
                    continue

                # ---- SANITY CHECK: reject bad homographies ----
                if not self._is_valid_homography(H_local):
                    print("-> [WARNING] Implausible homography, skipping frame.")
                    continue

                # ---- UPDATE PIXELS-PER-METRE SCALE (GPS + homography) ----
                self._update_ppm_estimate(gps_0, gps_1, H_local)

                # ---- CHAIN HOMOGRAPHIES ----
                # H_local: frame(N-1) -> frame(N)
                # H_inv  : frame(N)   -> frame(N-1)  (warps current frame onto canvas)
                try:
                    H_inv = np.linalg.inv(H_local)
                except np.linalg.LinAlgError:
                    print("-> [WARNING] Singular homography, skipping frame.")
                    continue

                self.H_global = self.H_global @ H_inv

                # ---- GPS DRIFT CORRECTION (translation only, rotation/scale kept) ----
                self._apply_gps_drift_correction(gps_1, frame.shape)

                # Warp the new frame to the current canvas size
                warped_frame = cv2.warpPerspective(
                    frame, self.H_global, (self.canvas_w, self.canvas_h)
                )

                self._place_frame(warped_frame)
                total_frames_stitched += 1

            # Free chunk memory after each .pt file is processed
            del buffer
            gc.collect()

        print(f"\n[STITCH] Successfully stitched {total_frames_stitched} frames.")
        print(f"[STITCH] Final canvas size: {self.canvas_w}x{self.canvas_h}")
        self._crop_and_save()

    # ==========================================
    # CROP + SAVE
    # ==========================================
    def _crop_and_save(self):
        """Trims black borders and saves the final map."""
        if self.canvas is None:
            print("[STITCH] [ERROR] Canvas is None — no frames were stitched. "
                  "Check map_tiles folder.")
            return

        print("\n[STITCH] Trimming black borders...")
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped = self.canvas[y:y + h, x:x + w]
            print(f"[STITCH] Final map resolution: {w}x{h} pixels.")
            print(f"[STITCH] Saving to '{self.output_file}'...")
            cv2.imwrite(self.output_file, cropped)
            print("[STITCH] Map generation complete!")
        else:
            print("[STITCH] [ERROR] Canvas is completely empty. Nothing to save.")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    stitcher = MapStitcher()
    stitcher.run()