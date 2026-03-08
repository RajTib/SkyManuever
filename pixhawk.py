import time
import threading
from collections import deque
from pymavlink import mavutil

class TelemetryNode:
    def __init__(self, connection_string='udpin:0.0.0.0:14550', buffer_size=50):
        print(f"[PIXHAWK] 📡 Waiting for heartbeat on {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        print(f"[PIXHAWK] ✅ Heartbeat detected! Connected to System {self.master.target_system}")

        # Request 10Hz telemetry streams
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )

        # Thread-safe Ring Buffers to hold historical data (Timestamp, Data_Tuple)
        # buffer_size=50 at 10Hz means we store the last 5 seconds of flight data
        self.gps_buffer = deque(maxlen=buffer_size)
        self.imu_buffer = deque(maxlen=buffer_size)

        # Thread locking to prevent read/write collisions between main.py and the background listener
        self.lock = threading.Lock()

        # Start the background listener thread
        self.listener_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.listener_thread.start()

    def _read_loop(self):
        """Background thread that constantly ingests and timestamps MAVLink packets."""
        while True:
            # blocking=True is safe here because it's running in its own isolated thread
            msg = self.master.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True)
            if not msg:
                continue

            current_time = time.time() # The Jetson's Master Clock timestamp

            with self.lock:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    # Convert to standard degrees and meters
                    lat, lon, alt = msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1000.0
                    self.gps_buffer.append((current_time, (lat, lon, alt)))

                elif msg.get_type() == 'ATTITUDE':
                    self.imu_buffer.append((current_time, (msg.roll, msg.pitch, msg.yaw)))

    def _interpolate(self, t_target, t0, data0, t1, data1):
        """Applies linear interpolation between two data points."""
        # Calculate the ratio of where the target time falls between t0 and t1
        ratio = (t_target - t0) / (t1 - t0)

        # Interpolate each value in the tuple (e.g., Lat, Lon, Alt)
        interpolated_data = tuple(
            v0 + ratio * (v1 - v0) for v0, v1 in zip(data0, data1)
        )
        return interpolated_data

    def get_synchronized_data(self, frame_timestamp):
        """
        Searches the buffers and calculates the exact telemetry at the frame's timestamp.
        Returns: (sync_gps, sync_imu)
        """
        with self.lock:
            # Create shallow copies so the thread doesn't modify them while we iterate
            gps_history = list(self.gps_buffer)
            imu_history = list(self.imu_buffer)

        sync_gps = self._find_and_interpolate(frame_timestamp, gps_history, default=(0.0, 0.0, 0.0))
        sync_imu = self._find_and_interpolate(frame_timestamp, imu_history, default=(0.0, 0.0, 0.0))

        return sync_gps, sync_imu

    def _find_and_interpolate(self, target_t, history, default):
        """Helper to find the surrounding timestamps and run the math."""
        if len(history) == 0:
            return default

        if len(history) == 1:
            return history[0][1] # Only one point exists, return it

        # If the camera frame is newer than our newest telemetry, we just return the absolute newest.
        # (Extrapolating into the future is mathematically dangerous for drones).
        if target_t >= history[-1][0]:
            return history[-1][1]

        # If the camera frame is extremely old (buffer rolled over), return the oldest.
        if target_t <= history[0][0]:
            return history[0][1]

        # Search for the two packets that surround the camera frame
        for i in range(len(history) - 1):
            t0, data0 = history[i]
            t1, data1 = history[i+1]

            if t0 <= target_t <= t1:
                return self._interpolate(target_t, t0, data0, t1, data1)

        return history[-1][1] # Fallback


# ==========================================
# INDEPENDENT TESTING BLOCK
# ==========================================
if __name__ == '__main__':
    telemetry = TelemetryNode()
    print("--- Starting Synchronized Telemetry Test ---")

    # Let the buffer fill up for a second
    time.sleep(1)

    try:
        while True:
            # Simulate a camera firing right now
            fake_camera_fire_time = time.time()

            # Ask the node for the exact coordinates at that specific microsecond
            sync_gps, sync_imu = telemetry.get_synchronized_data(fake_camera_fire_time)

            print(f"Sync GPS: Lat {sync_gps[0]:.6f}, Lon {sync_gps[1]:.6f} | Sync Roll: {sync_imu[0]:.3f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[PIXHAWK] Test stopped.")
