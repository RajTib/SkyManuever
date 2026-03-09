import torch
import cv2
import numpy as np
import sys
import os
import gc
import math
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd  # removes batch dimension cleanly


class MapNode:
    def __init__(self, use_tensorrt=False, save_dir="map_tiles", mem_limit_mb=500):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt

        print("[MAP] Loading PyTorch Models...")
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher   = LightGlue(features='superpoint').eval().to(self.device)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.mem_limit_bytes  = mem_limit_mb * 1024 * 1024
        self.tile_buffer      = []
        self.tile_counter     = 0
        self.last_frame_feats = None
        self.last_gps         = None
        self.frame_shape      = None  # stored on first frame for translation check

    # ==========================================
    # HOMOGRAPHY SANITY CHECK (EXPANDED)
    # ==========================================
    @staticmethod
    def _is_valid_homography(H, frame_shape):
        if H is None:
            return False, "H is None"

        # 1. Orientation check — no reflections
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            return False, f"Negative det={det:.4f} (reflection/flip)"

        # 2. Scale check — reject zoom-ins/outs beyond reason
        scale = math.sqrt(abs(det))
        if not (0.5 < scale < 2.0):
            return False, f"Scale={scale:.3f} out of range (0.5–2.0)"

        # 3. Perspective check — camera must be roughly nadir
        if abs(H[2, 0]) > 0.003 or abs(H[2, 1]) > 0.003:
            return False, f"Perspective too large: h20={H[2,0]:.5f}, h21={H[2,1]:.5f}"

        # 4. Translation sanity — drone shouldn't jump more than 60% of frame
        h, w = frame_shape[:2]
        tx, ty = H[0, 2], H[1, 2]
        if abs(tx) > 0.6 * w or abs(ty) > 0.6 * h:
            return False, f"Translation too large: tx={tx:.1f}, ty={ty:.1f}"

        # 5. Shear/rotation sanity — reject extreme rotations between frames
        angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))
        if abs(angle) > 30:
            return False, f"Rotation too large: {angle:.1f}°"

        return True, "OK"

    # ==========================================
    # MAIN PROCESSING
    # ==========================================
    def process_frame(self, frame_np, gps_coords):
        if self.frame_shape is None:
            self.frame_shape = frame_np.shape

        image_rgb    = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            feats = self.extractor.extract(image_tensor)

        if self.last_frame_feats is not None:
            with torch.inference_mode():
                matches01 = self.matcher({
                    'image0': self.last_frame_feats,
                    'image1': feats
                })

            # ✅ Correct LightGlue output parsing
            feats0_clean = rbd(self.last_frame_feats)
            feats1_clean = rbd(feats)
            matches_clean = rbd(matches01)

            m_indices = matches_clean['matches']  # shape: [N, 2]

            if len(m_indices) > 20:
                kpts0 = feats0_clean['keypoints'].cpu().numpy()
                kpts1 = feats1_clean['keypoints'].cpu().numpy()

                m_kpts0 = kpts0[m_indices[:, 0]]
                m_kpts1 = kpts1[m_indices[:, 1]]

                H, inliers = cv2.findHomography(
                    m_kpts0, m_kpts1,
                    cv2.USAC_MAGSAC, 3.0,          # tighter threshold
                    confidence=0.9999,
                    maxIters=10000
                )

                valid, reason = self._is_valid_homography(H, self.frame_shape)

                if valid:
                    inlier_count = int(inliers.sum())
                    inlier_ratio = inlier_count / len(m_indices)

                    # ✅ Extra: reject if too few inliers relative to matches
                    if inlier_ratio < 0.25:
                        print(f"[MAP] Low inlier ratio {inlier_ratio:.2f}, frame dropped.")
                    else:
                        # Store compressed image path, not raw array
                        frame_path = os.path.join(
                            self.save_dir, f"frame_{self.tile_counter}_{len(self.tile_buffer)}.jpg"
                        )
                        cv2.imwrite(frame_path, frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])

                        self.tile_buffer.append({
                            'gps_0':      self.last_gps,
                            'gps_1':      gps_coords,
                            'homography': H,
                            'inliers':    inlier_count,
                            'inlier_ratio': inlier_ratio,
                            'image_path': frame_path   # ✅ Path only, not raw array
                        })
                        print(f"[MAP] Frame accepted | matches={len(m_indices)} "
                              f"inliers={inlier_count} ratio={inlier_ratio:.2f}")
                else:
                    print(f"[MAP] Homography rejected: {reason}")
            else:
                print(f"[MAP] Too few matches ({len(m_indices)}), frame skipped.")

        # ✅ Detach features — don't keep computation graph
        self.last_frame_feats = {k: v.detach() for k, v in feats.items()}
        self.last_gps = gps_coords
        self._check_memory_and_flush()

    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    def _check_memory_and_flush(self):
        # ✅ Don't count image bytes since we store paths now
        current_mem_bytes = len(self.tile_buffer) * 512  # rough metadata estimate
        if current_mem_bytes >= self.mem_limit_bytes:
            self._flush_to_ssd()

    def _flush_to_ssd(self):
        if not self.tile_buffer:
            return
        tile_name = os.path.join(self.save_dir, f"tile_chunk_{self.tile_counter}.pt")
        print(f"[MAP] Saving chunk {self.tile_counter} "
              f"({len(self.tile_buffer)} frames) -> {tile_name}")
        torch.save(self.tile_buffer, tile_name)
        self.tile_buffer.clear()
        self.tile_counter += 1
        gc.collect()
        torch.cuda.empty_cache()
        print("[MAP] Buffer flushed.")

    def finalize(self):
        if self.tile_buffer:
            print(f"[MAP] Finalize: flushing {len(self.tile_buffer)} remaining frames.")
            self._flush_to_ssd()
        else:
            print("[MAP] Finalize: buffer empty.")


# ==========================================
# TEST
# ==========================================
if __name__ == '__main__':
    mapper = MapNode(use_tensorrt=False, mem_limit_mb=20)
    print("\nStarting pipeline test...")

    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_gps   = (12.9716, 77.5946, 100.0)

    mapper.process_frame(dummy_frame, dummy_gps)
    mapper.process_frame(dummy_frame, dummy_gps)  # Need 2 frames to trigger matching
    mapper.finalize()
    print("Test Complete.")
