import cv2
import time
import threading
import tkinter as tk
import multiprocessing as mp


# ==========================================
# WORKER PROCESS 1: THE MAPPER (LightGlue)
# ==========================================
def map_worker(queue):
    print("[WORKER] Map Node Process Started.")
    # CRITICAL: import inside the worker to prevent CUDA context crashes on spawn
    from map2 import MapNode

    mapper = MapNode(use_tensorrt=False)

    while True:
        data = queue.get()
        if data is None:            # Poison pill — time to shut down
            break
        frame, gps = data
        mapper.process_frame(frame, gps)

    # Flush any frames still in RAM before the process exits.
    # Without this, every flight silently loses the last batch of frames
    # (those sitting below the memory threshold when the worker is killed).
    mapper.finalize()
    print("[WORKER] Map Node Process Terminated.")


# ==========================================
# WORKER PROCESS 2: THE ML (YOLO Targeting)
# ==========================================
def ml_worker(queue):
    print("[WORKER] ML Targeting Process Started.")
    from ml2 import MLNode

    detector = MLNode(target_class_id=0, use_tensorrt=False)

    while True:
        data = queue.get()
        if data is None:
            break
        frame, gps = data
        detector.process_frame(frame, gps)

    print("[WORKER] ML Node Process Terminated.")


# ==========================================
# QUIT BUTTON  (runs in its own thread)
# ==========================================
def _launch_quit_button(shutdown_event: threading.Event):
    """
    Opens a small always-on-top tkinter window with a single red QUIT button.

    Pressing it (or closing the window) sets the shared shutdown_event, which
    tells the main camera loop to stop, flush all in-RAM data, and exit cleanly.

    Runs in a daemon thread so it never blocks program startup.
    """
    root = tk.Tk()
    root.title("Drone Control")
    root.resizable(False, False)
    root.attributes("-topmost", True)   # Always visible above the camera feed

    def _on_quit():
        print("\n[QUIT BUTTON] Shutdown requested — saving all in-RAM data...")
        shutdown_event.set()
        root.destroy()

    # Closing the window via the X button is treated the same as pressing QUIT
    root.protocol("WM_DELETE_WINDOW", _on_quit)

    btn = tk.Button(
        root,
        text="  QUIT & SAVE  ",
        command=_on_quit,
        bg="#cc0000",
        fg="white",
        font=("Helvetica", 14, "bold"),
        padx=20,
        pady=12,
        relief=tk.FLAT,
        cursor="hand2",
    )
    btn.pack(padx=30, pady=20)

    # Poll the shutdown_event every 200 ms so the window closes automatically
    # if shutdown was triggered by 'q' or Ctrl+C rather than the button itself.
    def _poll():
        if shutdown_event.is_set():
            root.destroy()
            return
        root.after(200, _poll)

    root.after(200, _poll)
    root.mainloop()


# ==========================================
# THE ORCHESTRATOR (MAIN PROCESS)
# ==========================================
def main():
    # Force 'spawn' so each worker gets a clean, separate GPU memory context.
    # 'fork' is unsafe with CUDA and will cause hangs or corrupted GPU state.
    mp.set_start_method('spawn', force=True)

    print("--- Booting Drone Architecture ---")

    # ---- Shared shutdown flag ----
    # A threading.Event is the single source of truth for "time to stop".
    # It can be set by: the QUIT button, 'q' keypress, Ctrl+C, or any error.
    # Every path converges here, so nothing can be skipped.
    shutdown_event = threading.Event()

    # maxsize=3 physically caps RAM by dropping frames when workers are busy.
    # This is intentional — we prefer dropping frames over running out of memory.
    map_queue = mp.Queue(maxsize=3)
    ml_queue  = mp.Queue(maxsize=3)

    p_map = mp.Process(target=map_worker, args=(map_queue,))
    p_ml  = mp.Process(target=ml_worker,  args=(ml_queue,))

    p_map.start()
    p_ml.start()

    # Connect to Pixhawk (spawns the background UDP ring-buffer thread)
    from pixhawk2 import TelemetryNode
    telemetry = TelemetryNode(connection_string='udpin:0.0.0.0:14550')

    cap = cv2.VideoCapture(0)

    print("[MAIN] Camera loop active.")
    print("[MAIN] Press 'q' in the camera window  OR  click QUIT & SAVE  OR  Ctrl+C to stop.")

    try:
        while not shutdown_event.is_set():
            ret, frame = cap.read()

            # Timestamp IMMEDIATELY after cap.read() returns so the GPS
            # interpolation in pixhawk.py gets the most accurate sync.
            frame_time = time.time()

            if not ret:
                print("[MAIN] Camera error — dropping frame.")
                continue

            current_gps, current_imu = telemetry.get_synchronized_data(frame_time)

            # put_nowait + full() check = non-blocking drop-frame behaviour.
            # Workers are never starved and the main loop never blocks.
            if not ml_queue.full():
                ml_queue.put_nowait((frame, current_gps))

            if not map_queue.full():
                map_queue.put_nowait((frame, current_gps))

            time.sleep(0.001)   # tiny yield so the loop doesn't peg the CPU at 100%

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected — initiating safe shutdown...")
        shutdown_event.set()

    finally:
        print("[MAIN] Saving all in-RAM data before exit...")
        cap.release()
        cv2.destroyAllWindows()

        # Send poison pills to each worker so they run finalize() and exit.
        # put(timeout=5) avoids a deadlock if the queue is still full.
        for q, p in [(map_queue, p_map), (ml_queue, p_ml)]:
            try:
                q.put(None, timeout=5)
            except Exception:
                print(f"[MAIN] Worker {p.name} unresponsive — terminating forcefully.")
                p.terminate()

        # Wait for workers to finish their current frame and flush to disk.
        # 30 s is generous; the map worker needs time to torch.save() the last chunk.
        p_map.join(timeout=30)
        p_ml.join(timeout=30)

        if p_map.is_alive():
            print("[MAIN] Map worker did not exit in time — killing.")
            p_map.kill()
        if p_ml.is_alive():
            print("[MAIN] ML worker did not exit in time — killing.")
            p_ml.kill()

        print("--- Drone Architecture Shutdown Complete. All data saved. ---")


if __name__ == '__main__':
    main()