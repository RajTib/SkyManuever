import torch
import cv2
import numpy as np
import sys
import os
import gc

# Standard PyTorch Imports
from lightglue import LightGlue, SuperPoint

# ==========================================
# TENSORRT WRAPPER PLACEHOLDERS
# ==========================================
class TRT_SuperPoint:
    def __init__(self, engine_path):
        print(f"[TRT] Loading Engine: {engine_path} (Placeholder)")
        # We will write the PyCUDA/TensorRT memory bindings here later

    def extract(self, image_tensor):
        print("[TRT] Executing SuperPoint Engine... (Placeholder)")
        return None

class TRT_LightGlue:
    def __init__(self, engine_path):
        print(f"[TRT] Loading Engine: {engine_path} (Placeholder)")
        # We will write the PyCUDA/TensorRT memory bindings here later

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

        # --- THE UNIFIED INTERFACE TOGGLE ---
        if self.use_tensorrt:
            print("[MAP] 🚀 HIGH PERFORMANCE MODE: Loading TensorRT Engines...")
            self.extractor = TRT_SuperPoint("superpoint.engine")
            self.matcher = TRT_LightGlue("lightglue.engine")
        else:
            print("[MAP] 🐢 FALLBACK MODE: Loading standard PyTorch Models...")
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.mem_limit_bytes = mem_limit_mb * 1024 * 1024
        self.tile_buffer = []
        self.tile_counter = 0

        self.last_frame_feats = None
        self.last_gps = None

    def process_frame(self, frame_np, gps_coords):
        """Processes a BGR image and GPS tag regardless of the AI backend."""
        image_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # The rest of the pipeline doesn't care if it's PyTorch or TensorRT!
        if not self.use_tensorrt:
            with torch.inference_mode():
                feats = self.extractor.extract(image_tensor)

            if self.last_frame_feats is not None:
                with torch.inference_mode():
                    matches01 = self.matcher({'image0': self.last_frame_feats, 'image1': feats})

                matches = matches01['matches'][0]

                if len(matches) > 15:
                    kpts0 = self.last_frame_feats['keypoints'][0]
                    kpts1 = feats['keypoints'][0]

                    m_kpts0 = kpts0[matches[..., 0]].cpu().numpy()
                    m_kpts1 = kpts1[matches[..., 1]].cpu().numpy()

                    H, inliers = cv2.findHomography(m_kpts0, m_kpts1, cv2.USAC_MAGSAC, 5.0)

                    if H is not None:
                        inlier_count = int(inliers.sum())
                        self.tile_buffer.append({
                            'gps_0': self.last_gps,
                            'gps_1': gps_coords,
                            'homography': H,
                            'inliers': inlier_count,
                            'image_1': frame_np
                        })

            self.last_frame_feats = feats
            self.last_gps = gps_coords
            self._check_memory_and_flush()
        else:
            # Placeholder for TRT execution flow
            self.extractor.extract(image_tensor)

    def _check_memory_and_flush(self):
        current_mem_bytes = sum([sys.getsizeof(item['image_1']) + item['image_1'].nbytes for item in self.tile_buffer])
        if current_mem_bytes >= self.mem_limit_bytes:
            print(f"\n[MAP] Memory threshold reached: {current_mem_bytes / 1e6:.1f} MB.")
            self._flush_to_ssd()

    def _flush_to_ssd(self):
        tile_name = os.path.join(self.save_dir, f"tile_chunk_{self.tile_counter}.pt")
        print(f"[MAP] Saving map chunk to {tile_name}...")
        torch.save(self.tile_buffer, tile_name)
        self.tile_buffer.clear()
        self.tile_counter += 1
        gc.collect()
        torch.cuda.empty_cache()
        print("[MAP] RAM cleared. Resuming mapping.")

# ==========================================
# INDEPENDENT TESTING BLOCK
# ==========================================
if __name__ == '__main__':
    # You can now toggle the script right here!
    # Change to True to test the TRT placeholders, False for real PyTorch
    mapper = MapNode(use_tensorrt=False, mem_limit_mb=20)

    print("\nStarting pipeline test...")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_gps = (12.9716, 77.5946, 100.0)

    mapper.process_frame(dummy_frame, dummy_gps)
    print("Test Complete.")
