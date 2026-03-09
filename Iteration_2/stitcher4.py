"""
MapStitcher v2 — Coverage-Mask Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changed from v1
────────────────────
① COVERAGE MASK  — a single-channel uint8 image (same size as canvas)
                   where 255 = already painted, 0 = empty.
                   Before placing any tile we compute what % of its
                   area is already covered.  If coverage >= OVERLAP_THRESHOLD
                   (default 70 % → means <30 % new content) the tile is
                   dropped.  This kills the blurring/ghosting caused by
                   re-painting the same ground over and over.

② NEW-CONTENT %  — logged per frame so you can tune the threshold live.

③ CANVAS EXPAND  — coverage mask expands in sync with canvas (was missing).

④ SAVE COVERAGE  — optionally saves coverage_map.png for debug.

Everything else (GPS drift correction, PPM estimation, homography chain,
dynamic canvas, crop-and-save) is identical to v1 — just re-integrated
cleanly.
"""

from __future__ import annotations

import gc
import glob
import math
import os

import cv2
import numpy as np
import torch


class MapStitcher:
    # ── Tunable knobs ────────────────────────────────────────────────────────
    OVERLAP_THRESHOLD       = 0.70   # Drop tile if >70 % of its area already painted
                                     # (= require at least 30 % new content)
    MIN_PPM_SAMPLES         = 5
    MAX_DRIFT_PIXELS        = 15.0
    GPS_CORRECTION_STRENGTH = 0.7

    def __init__(
        self,
        map_dir     : str = "map_tiles",
        output_file : str = "final_map.jpg",
        overlap_threshold : float | None = None,   # override at runtime if desired
    ):
        self.map_dir     = map_dir
        self.output_file = output_file

        if overlap_threshold is not None:
            self.OVERLAP_THRESHOLD = overlap_threshold

        # ── Canvas state ─────────────────────────────────────────────────────
        self.canvas   : np.ndarray | None = None   # BGR  uint8
        self.coverage : np.ndarray | None = None   # Gray uint8  (255=painted)
        self.canvas_w = 0
        self.canvas_h = 0

        # ── Homography chain ─────────────────────────────────────────────────
        self.H_global      = np.eye(3, dtype=np.float64)
        self.is_first_frame = True

        # ── GPS drift correction ─────────────────────────────────────────────
        self.canvas_cx       = 0
        self.canvas_cy       = 0
        self.origin_gps      = None
        self.pixels_per_meter = None
        self._ppm_samples    : list[float] = []

        # ── Stats ─────────────────────────────────────────────────────────────
        self._placed  = 0
        self._dropped = 0

    # =========================================================================
    # GPS UTILITIES  (unchanged from v1)
    # =========================================================================
    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6_371_000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def _gps_to_canvas_pixels(self, lat, lon):
        if self.origin_gps is None or self.pixels_per_meter is None:
            return None
        R = 6_371_000.0
        dlat   = math.radians(lat - self.origin_gps[0])
        dlon   = math.radians(lon - self.origin_gps[1])
        north  = R * dlat
        east   = R * dlon * math.cos(math.radians(self.origin_gps[0]))
        px     = self.canvas_cx + east  * self.pixels_per_meter
        py     = self.canvas_cy - north * self.pixels_per_meter
        return px, py

    def _update_ppm_estimate(self, gps_0, gps_1, H_local):
        if self.pixels_per_meter is not None or gps_0 is None or gps_1 is None:
            return
        gps_dist = self._haversine_m(gps_0[0], gps_0[1], gps_1[0], gps_1[1])
        if gps_dist < 0.3:
            return
        o = H_local @ np.array([0.0, 0.0, 1.0])
        px_dist = math.sqrt((o[0]/o[2])**2 + (o[1]/o[2])**2)
        if px_dist < 2.0:
            return
        self._ppm_samples.append(px_dist / gps_dist)
        if len(self._ppm_samples) >= self.MIN_PPM_SAMPLES:
            self.pixels_per_meter = float(np.median(self._ppm_samples))
            print(f"   [GPS] PPM locked: {self.pixels_per_meter:.2f} px/m")

    def _apply_gps_drift_correction(self, gps_1, frame_shape):
        if self.pixels_per_meter is None or gps_1 is None:
            return
        res = self._gps_to_canvas_pixels(gps_1[0], gps_1[1])
        if res is None:
            return
        gps_cx, gps_cy = res
        h, w = frame_shape[:2]
        c = self.H_global @ np.array([w/2.0, h/2.0, 1.0])
        homo_cx, homo_cy = c[0]/c[2], c[1]/c[2]
        dx = gps_cx - homo_cx
        dy = gps_cy - homo_cy
        mag = math.sqrt(dx**2 + dy**2)
        if mag > self.MAX_DRIFT_PIXELS:
            cx = self.GPS_CORRECTION_STRENGTH * dx
            cy = self.GPS_CORRECTION_STRENGTH * dy
            print(f"   [GPS CORRECTION] drift={mag:.1f}px  corr=({cx:.1f},{cy:.1f})")
            T = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
            self.H_global = T @ self.H_global

    # =========================================================================
    # HOMOGRAPHY SANITY  (unchanged from v1)
    # =========================================================================
    @staticmethod
    def _is_valid_homography(H) -> bool:
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

    # =========================================================================
    # ① NEW — COVERAGE MASK HELPERS
    # =========================================================================
    def _compute_tile_coverage(self, warped_frame: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Given a warped frame (canvas-sized BGR image), compute:
          - coverage_ratio : fraction of the tile's non-black pixels that
                             overlap with already-painted canvas area
          - tile_mask_2d   : boolean mask of non-black pixels in the tile

        Returns (coverage_ratio, tile_mask_2d).
        A coverage_ratio >= OVERLAP_THRESHOLD means "too much overlap → drop".
        """
        gray      = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        tile_mask = gray > 10                         # True where tile has content

        tile_pixel_count = int(tile_mask.sum())
        if tile_pixel_count == 0:
            return 1.0, tile_mask                     # empty tile → treat as 100 % overlap

        # How many of this tile's pixels land on already-covered canvas area?
        already_covered = (self.coverage > 0) & tile_mask
        overlap_count   = int(already_covered.sum())

        coverage_ratio  = overlap_count / tile_pixel_count
        return coverage_ratio, tile_mask

    def _expand_canvas(self, pad_top: int, pad_left: int,
                       pad_bottom: int, pad_right: int) -> None:
        """
        Expand both canvas (BGR) AND coverage mask by the requested padding.
        Shift H_global and the GPS origin anchor accordingly.
        """
        if not any([pad_top, pad_left, pad_bottom, pad_right]):
            return

        print(f"   [EXPAND] T:{pad_top} B:{pad_bottom} L:{pad_left} R:{pad_right}")

        self.canvas = cv2.copyMakeBorder(
            self.canvas, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        # ② Expand coverage mask in sync — this was missing in v1
        self.coverage = cv2.copyMakeBorder(
            self.coverage, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        self.canvas_h, self.canvas_w = self.canvas.shape[:2]
        print(f"   [CANVAS] {self.canvas_w}×{self.canvas_h}")

        shift = np.array([
            [1.0, 0.0, float(pad_left)],
            [0.0, 1.0, float(pad_top)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.H_global  = shift @ self.H_global
        self.canvas_cx += pad_left
        self.canvas_cy += pad_top

    # =========================================================================
    # CORE: PLACE FRAME  (v2 — with coverage gate)
    # =========================================================================
    def _place_frame(self, warped_frame: np.ndarray) -> bool:
        """
        Attempts to place warped_frame onto the canvas.

        Returns True  if the tile was placed.
        Returns False if it was dropped (too much overlap OR empty).

        Steps
        ─────
        1. Find bounding box of non-black pixels → compute canvas expansion.
        2. After any expansion, re-warp the frame.
        3. Compute overlap ratio against coverage mask.
        4. If overlap >= OVERLAP_THRESHOLD → drop.
        5. Otherwise → paste pixels + update coverage mask.
        """
        if self.canvas is None:
            return False

        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        non_black = np.where(gray > 10)
        if len(non_black[0]) == 0:
            print("   [SKIP] Warped frame empty.")
            return False

        y_min, y_max = int(non_black[0].min()), int(non_black[0].max())
        x_min, x_max = int(non_black[1].min()), int(non_black[1].max())

        pad_top    = max(0, -y_min)
        pad_left   = max(0, -x_min)
        pad_bottom = max(0, y_max - self.canvas_h + 1)
        pad_right  = max(0, x_max - self.canvas_w + 1)

        if any([pad_top, pad_left, pad_bottom, pad_right]):
            self._expand_canvas(pad_top, pad_left, pad_bottom, pad_right)
            # Re-warp with updated H_global after shift
            shift = np.array([
                [1.0, 0.0, float(pad_left)],
                [0.0, 1.0, float(pad_top)],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            warped_frame = cv2.warpPerspective(
                warped_frame, shift, (self.canvas_w, self.canvas_h)
            )

        # ── ③ COVERAGE CHECK ────────────────────────────────────────────────
        coverage_ratio, tile_mask_2d = self._compute_tile_coverage(warped_frame)
        new_content_pct = (1.0 - coverage_ratio) * 100.0

        if coverage_ratio >= self.OVERLAP_THRESHOLD:
            print(f"   [DROP]  New content: {new_content_pct:.1f}%  "
                  f"(threshold requires >{(1-self.OVERLAP_THRESHOLD)*100:.0f}%)")
            self._dropped += 1
            return False

        print(f"   [PLACE] New content: {new_content_pct:.1f}%  ✓")

        # ── ④ PASTE + UPDATE COVERAGE ────────────────────────────────────────
        mask_3d = np.stack([tile_mask_2d] * 3, axis=2)
        self.canvas[mask_3d]   = warped_frame[mask_3d]
        self.coverage[tile_mask_2d] = 255          # mark as painted

        self._placed += 1
        return True

    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================
    def run(self):
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  MapStitcher v2 — Post-Flight Stitching")
        print(f"  Overlap threshold : {self.OVERLAP_THRESHOLD*100:.0f}% coverage → drop")
        print(f"  New-content gate  : >{(1-self.OVERLAP_THRESHOLD)*100:.0f}% required to place")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        chunk_files = sorted(
            glob.glob(os.path.join(self.map_dir, "*.pt")),
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        if not chunk_files:
            print(f"[ERROR] No .pt files found in '{self.map_dir}'.")
            return

        for chunk_file in chunk_files:
            print(f"\n[CHUNK] {chunk_file}")
            buf = torch.load(chunk_file, map_location="cpu", weights_only=False)

            for item in buf:
                frame   = item["image_1"]
                H_local = item["homography"]
                inliers = item.get("inliers", "?")
                gps_0   = item.get("gps_0")
                gps_1   = item.get("gps_1")

                print(f"-> frame | inliers={inliers}", end="  ")

                # ── FIRST FRAME ──────────────────────────────────────────────
                if self.is_first_frame:
                    h, w = frame.shape[:2]
                    self.canvas_w  = w
                    self.canvas_h  = h
                    self.H_global  = np.eye(3, dtype=np.float64)
                    self.canvas    = frame.copy()

                    # Initialise coverage mask: first frame is fully painted
                    first_gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.coverage  = (first_gray > 10).astype(np.uint8) * 255

                    self.canvas_cx = w // 2
                    self.canvas_cy = h // 2
                    if gps_1 is not None:
                        self.origin_gps = (gps_1[0], gps_1[1])
                        print(f"\n   [GPS] Origin: {self.origin_gps}")

                    self.is_first_frame = False
                    self._placed += 1
                    print("→ anchored (first frame)")
                    continue

                # ── SANITY CHECK ─────────────────────────────────────────────
                if not self._is_valid_homography(H_local):
                    print("→ SKIP (bad homography)")
                    continue

                # ── GPS SCALE ────────────────────────────────────────────────
                self._update_ppm_estimate(gps_0, gps_1, H_local)

                # ── CHAIN ────────────────────────────────────────────────────
                try:
                    H_inv = np.linalg.inv(H_local)
                except np.linalg.LinAlgError:
                    print("→ SKIP (singular H)")
                    continue

                self.H_global = self.H_global @ H_inv

                # ── GPS DRIFT CORRECTION ─────────────────────────────────────
                self._apply_gps_drift_correction(gps_1, frame.shape)

                # ── WARP + PLACE (with coverage gate) ────────────────────────
                warped = cv2.warpPerspective(
                    frame, self.H_global, (self.canvas_w, self.canvas_h)
                )
                self._place_frame(warped)

            del buf
            gc.collect()

        print(f"\n{'━'*43}")
        print(f"  Frames placed  : {self._placed}")
        print(f"  Frames dropped : {self._dropped}  (overlap>{self.OVERLAP_THRESHOLD*100:.0f}%)")
        print(f"  Canvas size    : {self.canvas_w}×{self.canvas_h}")
        print(f"{'━'*43}\n")
        self._crop_and_save()

    # =========================================================================
    # CROP + SAVE
    # =========================================================================
    def _crop_and_save(self):
        if self.canvas is None:
            print("[ERROR] Canvas is None — nothing to save.")
            return

        print("[SAVE] Trimming black borders...")
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[ERROR] Canvas is empty.")
            return

        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cropped = self.canvas[y:y+h, x:x+w]
        cv2.imwrite(self.output_file, cropped)
        print(f"[SAVE] Map saved → '{self.output_file}'  ({w}×{h} px)")

        # Optional: save coverage map for debugging
        cov_path = self.output_file.replace(".jpg", "_coverage.png")
        cov_crop = self.coverage[y:y+h, x:x+w]
        cv2.imwrite(cov_path, cov_crop)
        print(f"[SAVE] Coverage mask → '{cov_path}'")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    stitcher = MapStitcher(
        map_dir    = "map_tiles",
        output_file= "final_map.jpg",
        # Override threshold here if needed:
        # overlap_threshold=0.5   # 50% gate
    )
    stitcher.run()
