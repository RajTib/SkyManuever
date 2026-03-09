import torch
import cv2
import numpy as np
import sys
import os
import gc
import math

from lightglue import LightGlue, SuperPoint


# ==========================================
# TENSORRT WRAPPER PLACEHOLDERS
# ==========================================
class TRT_SuperPoint:
    def __init__(self, engine_path):
        print(f"[TRT] Loading Engine: {engine_path} (Placeholder)")

    def extract(self, image_tensor):
        print("[TRT] Executing SuperPoint Engine... (Placeholder)")
        return None


class TRT_LightGlue:
    def __init__(self, engine_path):
        print(f"[TRT] Loading Engine: {engine_path} (Placeholder)")

    def __call__(self, data):
        print("[TRT] Executing LightGlue Engine... (Placeholder)")
        return {'matches': [torch.empty((0, 2), dtype=torch.int64)]}


# ==========================================
# THE UNIFIED MAP NODE
# ==========================================
class MapNode:
    def __init__(self, use_tensorrt=False, save_dir="map_tiles", mem_limit_mb=500):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt

        if self.use_tensorrt:
            print("[MAP] HIGH PERFORMANCE MODE: Loading TensorRT Engines...")
            self.extractor = TRT_SuperPoint("superpoint.engine")
            self.matcher   = TRT_LightGlue("lightglue.engine")
        else:
            print("[MAP] FALLBACK MODE: Loading standard PyTorch Models...")
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher   = LightGlue(features='superpoint').eval().to(self.device)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.mem_limit_bytes  = mem_limit_mb * 1024 * 1024
        self.tile_buffer      = []
        self.tile_counter     = 0
        self.last_frame_feats = None
        self.last_gps         = None

    # ==========================================
    # HOMOGRAPHY SANITY CHECK
    # ==========================================
    @staticmethod
    def _is_valid_homography(H):
        """
        Reject degenerate or physically implausible homographies BEFORE they
        are stored.  A bad H here corrupts every future frame in the stitcher
        because homographies are chained multiplicatively.

        Checks:
          - Orientation-preserving (det > 0, no reflection/flip)
          - Scale change between frames is within 0.4x - 2.5x
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
    # MAIN PROCESSING
    # ==========================================
    def process_frame(self, frame_np, gps_coords):
        """
        Processes one BGR image frame paired with its GPS coordinates.

        On a good homography match the buffer entry always includes both
        gps_0 (previous frame GPS) and gps_1 (this frame GPS) so the
        post-flight stitcher can use GPS-assisted drift correction.
        """
        image_rgb    = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if not self.use_tensorrt:
            with torch.inference_mode():
                feats = self.extractor.extract(image_tensor)

            if self.last_frame_feats is not None:
                with torch.inference_mode():
                    matches01 = self.matcher(
                        {'image0': self.last_frame_feats, 'image1': feats}
                    )

                matches = matches01['matches'][0]

                if len(matches) > 15:
                    kpts0 = self.last_frame_feats['keypoints'][0]
                    kpts1 = feats['keypoints'][0]

                    m_kpts0 = kpts0[matches[..., 0]].cpu().numpy()
                    m_kpts1 = kpts1[matches[..., 1]].cpu().numpy()

                    H, inliers = cv2.findHomography(
                        m_kpts0, m_kpts1, cv2.USAC_MAGSAC, 5.0
                    )

                    if self._is_valid_homography(H):
                        inlier_count = int(inliers.sum())
                        self.tile_buffer.append({
                            'gps_0':      self.last_gps,  # GPS of previous frame
                            'gps_1':      gps_coords,      # GPS of this frame
                            'homography': H,
                            'inliers':    inlier_count,
                            'image_1':    frame_np         # Current frame (BGR)
                        })
                    else:
                        print("[MAP] Homography rejected (degenerate), frame dropped.")

            self.last_frame_feats = feats
            self.last_gps         = gps_coords
            self._check_memory_and_flush()

        else:
            self.extractor.extract(image_tensor)

    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    def _check_memory_and_flush(self):
        current_mem_bytes = sum(
            sys.getsizeof(item['image_1']) + item['image_1'].nbytes
            for item in self.tile_buffer
        )
        if current_mem_bytes >= self.mem_limit_bytes:
            print(f"\n[MAP] Memory threshold reached: "
                  f"{current_mem_bytes / 1e6:.1f} MB.")
            self._flush_to_ssd()

    def _flush_to_ssd(self):
        if not self.tile_buffer:
            return
        tile_name = os.path.join(
            self.save_dir, f"tile_chunk_{self.tile_counter}.pt"
        )
        print(f"[MAP] Saving chunk {self.tile_counter} "
              f"({len(self.tile_buffer)} frames) -> {tile_name}")
        torch.save(self.tile_buffer, tile_name)
        self.tile_buffer.clear()
        self.tile_counter += 1
        gc.collect()
        torch.cuda.empty_cache()
        print("[MAP] RAM cleared. Resuming mapping.")

    def finalize(self):
        """
        Flush any frames still in the buffer to disk.

        MUST be called when the drone lands / worker shuts down.
        Without this, every flight will silently lose the last batch of
        frames (those sitting below the memory threshold when killed).
        """
        if self.tile_buffer:
            print(f"[MAP] Finalize: flushing {len(self.tile_buffer)} remaining "
                  f"frames to disk.")
            self._flush_to_ssd()
        else:
            print("[MAP] Finalize: buffer empty, nothing to flush.")


# ==========================================
# INDEPENDENT TESTING BLOCK
# ==========================================
if __name__ == '__main__':
    mapper = MapNode(use_tensorrt=False, mem_limit_mb=20)

    print("\nStarting pipeline test...")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_gps   = (12.9716, 77.5946, 100.0)

    mapper.process_frame(dummy_frame, dummy_gps)
    mapper.finalize()
    print("Test Complete.")
