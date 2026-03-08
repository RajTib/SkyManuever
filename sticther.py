import torch
import cv2
import numpy as np
import os
import glob
weights_only=False
class MapStitcher:
    def __init__(self, map_dir="map_tiles", output_file="final_map.jpg"):
        self.map_dir = map_dir
        self.output_file = output_file
        
        # Create a massive blank canvas (10000x10000) to ensure we don't map out of bounds
        self.canvas_size = (10000, 10000)
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        # Start drawing right in the middle of the massive canvas
        self.H_global = np.array([
            [1.0, 0.0, self.canvas_size[0] // 2],
            [0.0, 1.0, self.canvas_size[1] // 2],
            [0.0, 0.0, 1.0]
        ])
        
        self.is_first_frame = True

    def run(self):
        print(f"--- Starting Post-Flight Map Stitching ---")
        
        # 1. Find all .pt chunks and sort them numerically so we process them in exact chronological order
        chunk_files = sorted(glob.glob(os.path.join(self.map_dir, "*.pt")), 
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        if not chunk_files:
            print(f"[ERROR] No map data found in '{self.map_dir}'.")
            return

        # 2. Iterate through each flight chunk
        total_frames_stitched = 0
        for chunk_file in chunk_files:
            print(f"[STITCH] Unpacking RAM chunk: {chunk_file}...")
            
            # Load the PyTorch buffer from the SSD
            buffer = torch.load(chunk_file, map_location='cpu', weights_only=False)
            
            for item in buffer:
                frame = item['image_1']
                H_local = item['homography']
                
                if self.is_first_frame:
                    # Paint the very first frame onto the center of the canvas
                    print("         -> Anchoring first frame to canvas center.")
                    self.canvas = cv2.warpPerspective(frame, self.H_global, self.canvas_size)
                    self.is_first_frame = False
                    total_frames_stitched += 1
                    continue
                
                # 3. The Math: Invert the Homography
                # map.py calculates H from Frame0 -> Frame1. 
                # To stitch, we must warp Frame1 backwards onto Frame0.
                try:
                    H_inv = np.linalg.inv(H_local)
                except np.linalg.LinAlgError:
                    print("         -> [WARNING] Bad matrix inversion, skipping frame.")
                    continue
                    
                # 4. Chain the Math: Multiply current global position by the new step
                self.H_global = np.matmul(self.H_global, H_inv)
                
                # 5. Warp the new frame onto the global canvas geometry
                warped_frame = cv2.warpPerspective(frame, self.H_global, self.canvas_size)
                
                # 6. Smart Blending
                # Create a mask of the new warped frame (where is it not black?)
                mask = (warped_frame > 0)
                
                # Only overwrite pixels on the canvas that don't exist yet, 
                # or optionally blend them. Here we overwrite for maximum sharpness.
                self.canvas[mask] = warped_frame[mask]
                total_frames_stitched += 1

        print(f"\n[STITCH] Successfully chained {total_frames_stitched} frames.")
        self._crop_and_save()

    def _crop_and_save(self):
        """Finds the bounding box of the actual map and crops out the useless black space."""
        print("[STITCH] Trimming excess black canvas...")
        
        # Convert to grayscale to find non-black pixels
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find all contours (shapes) of the colored areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the absolute minimum bounding rectangle that fits all map pixels
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped_canvas = self.canvas[y:y+h, x:x+w]
            
            print(f"[STITCH] Final map resolution: {w}x{h} pixels.")
            print(f"[STITCH] Saving to {self.output_file}...")
            cv2.imwrite(self.output_file, cropped_canvas)
            print("[STITCH] ✅ Map generation complete!")
        else:
            print("[STITCH] Error: Canvas is completely empty.")


if __name__ == "__main__":
    stitcher = MapStitcher()
    stitcher.run()
