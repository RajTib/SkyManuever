import cv2
import time
import multiprocessing as mp

# ==========================================
# WORKER PROCESS 1: THE MAPPER (LightGlue)
# ==========================================
def map_worker(queue):
    print("[WORKER] Map Node Process Started.")
    # CRITICAL: Import happens inside the worker to prevent CUDA context crashes
    from map import MapNode

    # Initialize in High-Performance TensorRT Mode
    mapper = MapNode(use_tensorrt=False)

    while True:
        data = queue.get()
        if data is None:  # The "Poison Pill" for safe shutdown
            break

        frame, gps = data
        mapper.process_frame(frame, gps)

    print("[WORKER] Map Node Process Terminated.")

# ==========================================
# WORKER PROCESS 2: THE ML (YOLO Target)
# ==========================================
def ml_worker(queue):
    print("[WORKER] ML Targeting Process Started.")
    from ml import MLNode

    # Initialize in High-Performance TensorRT Mode
    detector = MLNode(target_class_id=0, use_tensorrt=False)

    while True:
        data = queue.get()
        if data is None:
            break

        frame, gps = data
        detector.process_frame(frame, gps)

    print("[WORKER] ML Node Process Terminated.")

# ==========================================
# THE ORCHESTRATOR (MAIN PROCESS)
# ==========================================
def main():
    # 1. Force Python to use 'spawn' to safely allocate separate GPU memory
    mp.set_start_method('spawn', force=True)

    print("--- Booting Drone Architecture ---")

    # 2. Create the drop-frame queues (maxsize=3 physically limits RAM consumption)
    map_queue = mp.Queue(maxsize=3)
    ml_queue = mp.Queue(maxsize=3)

    # 3. Start the independent AI worker processes
    p_map = mp.Process(target=map_worker, args=(map_queue,))
    p_ml = mp.Process(target=ml_worker, args=(ml_queue,))

    p_map.start()
    p_ml.start()

    # 4. Connect Telemetry (Spawns the background UDP Ring Buffer thread)
    from pixhawk import TelemetryNode
    telemetry = TelemetryNode(connection_string='udpin:0.0.0.0:14550')

    # 5. Open the Camera
    cap = cv2.VideoCapture(0)

    print("[MAIN] Camera Loop Active. Press 'q' in the video window to safely land/quit.")

    try:
        while True:
            # A. Grab the absolute newest camera frame
            ret, frame = cap.read()

            # B. HARD SYNC: Capture the exact microsecond the frame hit the Jetson
            frame_time = time.time()

            if not ret:
                print("[MAIN] Camera error! Dropping frame.")
                continue

            # C. Ask the Pixhawk node to mathematically interpolate the exact GPS
            # coordinates for that specific microsecond
            current_gps, current_imu = telemetry.get_synchronized_data(frame_time)

            # D. Send perfectly synchronized data to the ML Node (If not busy)
            if not ml_queue.full():
                ml_queue.put_nowait((frame, current_gps))

            # E. Send perfectly synchronized data to the Map Node (If not busy)
            if not map_queue.full():
                map_queue.put_nowait((frame, current_gps))

            # F. Display the raw, lag-free feed for the pilot/debug
            #cv2.imshow("Drone Main Camera", frame)

            # Quit safely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[MAIN] 'q' pressed. Initiating safe shutdown...")
                break

    except KeyboardInterrupt:
         print("\n[MAIN] Ctrl+C detected. Initiating safe shutdown...")

    finally:
        # 6. Shut down everything gracefully
        cap.release()
        cv2.destroyAllWindows()

        # Send poison pills to kill the worker processes
        map_queue.put(None)
        ml_queue.put(None)

        # Wait for them to finish their last frame and clear memory
        p_map.join()
        p_ml.join()
        print("--- Drone Architecture Shutdown Complete ---")

if __name__ == '__main__':
    main()
