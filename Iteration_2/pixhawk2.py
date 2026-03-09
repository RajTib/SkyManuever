import time
import math
import threading
from collections import deque
from pymavlink import mavutil


class TelemetryNode:
    def __init__(self, connection_string='udpin:0.0.0.0:14550',
                 buffer_size=50, heartbeat_timeout=15):
        """
        Args:
            connection_string  : MAVLink connection string.
            buffer_size        : Ring buffer depth.  At 10 Hz this is 5 seconds
                                 of flight history.
            heartbeat_timeout  : Seconds to wait for the first heartbeat before
                                 raising a ConnectionError.  Without this the
                                 process would block forever if Pixhawk is off.
        """
        print(f"[PIXHAWK] Waiting for heartbeat on {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string)

        # Fail fast: if no heartbeat arrives within the timeout window,
        # raise immediately rather than hanging the whole drone boot sequence.
        heartbeat = self.master.wait_heartbeat(timeout=heartbeat_timeout)
        if heartbeat is None:
            raise ConnectionError(
                f"[PIXHAWK] No heartbeat received within {heartbeat_timeout}s. "
                f"Check Pixhawk power and connection string."
            )
        print(f"[PIXHAWK] Heartbeat detected! "
              f"Connected to System {self.master.target_system}")

        # Request 10 Hz telemetry streams
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )

        # Thread-safe ring buffers: (timestamp, data_tuple)
        # buffer_size=50 @ 10 Hz = last 5 seconds of flight data
        self.gps_buffer = deque(maxlen=buffer_size)
        self.imu_buffer = deque(maxlen=buffer_size)

        # Staleness tracking: warn if the buffer stops updating
        self._last_gps_time = None
        self._last_imu_time = None
        self.STALE_WARN_SECS = 2.0   # Warn if no new packet for this long

        self.lock = threading.Lock()

        self.listener_thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self.listener_thread.start()

    # ==========================================
    # BACKGROUND LISTENER
    # ==========================================
    def _read_loop(self):
        """Background thread: ingests and timestamps MAVLink packets."""
        while True:
            msg = self.master.recv_match(
                type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True, timeout=5.0
            )
            # timeout=5.0 so the thread unblocks occasionally even if the
            # Pixhawk goes silent — allows future reconnection logic to be added.
            if not msg:
                continue

            current_time = time.time()

            with self.lock:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                    alt = msg.alt / 1000.0
                    self.gps_buffer.append((current_time, (lat, lon, alt)))
                    self._last_gps_time = current_time

                elif msg.get_type() == 'ATTITUDE':
                    self.imu_buffer.append(
                        (current_time, (msg.roll, msg.pitch, msg.yaw))
                    )
                    self._last_imu_time = current_time

    # ==========================================
    # INTERPOLATION
    # ==========================================
    def _interpolate(self, t_target, t0, data0, t1, data1, is_attitude=False):
        """
        Linear interpolation between two telemetry samples.

        For attitude data (roll, pitch, yaw) the yaw component is treated
        with circular arithmetic to handle the +pi / -pi wrap-around.
        Without this, interpolating yaw=+3.1 rad to yaw=-3.1 rad would give
        yaw=0 (pointing south) instead of the correct ~+/-pi (pointing north).
        """
        ratio = (t_target - t0) / (t1 - t0)
        result = []

        for i, (v0, v1) in enumerate(zip(data0, data1)):
            # Yaw is always the 3rd element (index 2) of attitude tuples
            if is_attitude and i == 2:
                # Circular interpolation: always take the shortest arc
                diff = (v1 - v0 + math.pi) % (2 * math.pi) - math.pi
                interp = v0 + ratio * diff
                # Normalise back to [-pi, +pi]
                interp = (interp + math.pi) % (2 * math.pi) - math.pi
                result.append(interp)
            else:
                result.append(v0 + ratio * (v1 - v0))

        return tuple(result)

    def _find_and_interpolate(self, target_t, history, default, is_attitude=False):
        """Find the surrounding samples and interpolate."""
        if len(history) == 0:
            return default

        if len(history) == 1:
            return history[0][1]

        # Camera frame is newer than our newest telemetry —
        # return the absolute newest (extrapolating into the future is unsafe)
        if target_t >= history[-1][0]:
            return history[-1][1]

        # Camera frame pre-dates our oldest telemetry (buffer rolled over)
        if target_t <= history[0][0]:
            return history[0][1]

        for i in range(len(history) - 1):
            t0, data0 = history[i]
            t1, data1 = history[i + 1]
            if t0 <= target_t <= t1:
                return self._interpolate(
                    target_t, t0, data0, t1, data1, is_attitude=is_attitude
                )

        return history[-1][1]   # Should never reach here

    # ==========================================
    # PUBLIC API
    # ==========================================
    def get_synchronized_data(self, frame_timestamp):
        """
        Returns (sync_gps, sync_imu) interpolated to the exact moment
        the camera frame was captured.

        Also emits a warning if telemetry has gone stale, so the main
        loop knows it is working with old data rather than failing silently.
        """
        now = time.time()

        with self.lock:
            gps_history = list(self.gps_buffer)
            imu_history = list(self.imu_buffer)
            last_gps    = self._last_gps_time
            last_imu    = self._last_imu_time

        # Stale-data warning — don't silently feed old GPS to mapping/ML
        if last_gps is not None and (now - last_gps) > self.STALE_WARN_SECS:
            print(f"[PIXHAWK] WARNING: GPS telemetry stale "
                  f"({now - last_gps:.1f}s since last packet). "
                  f"Check MAVLink link.")
        if last_imu is not None and (now - last_imu) > self.STALE_WARN_SECS:
            print(f"[PIXHAWK] WARNING: IMU telemetry stale "
                  f"({now - last_imu:.1f}s since last packet).")

        sync_gps = self._find_and_interpolate(
            frame_timestamp, gps_history,
            default=(0.0, 0.0, 0.0), is_attitude=False
        )
        sync_imu = self._find_and_interpolate(
            frame_timestamp, imu_history,
            default=(0.0, 0.0, 0.0), is_attitude=True   # enable yaw wrap fix
        )

        return sync_gps, sync_imu


# ==========================================
# INDEPENDENT TESTING BLOCK
# ==========================================
if __name__ == '__main__':
    telemetry = TelemetryNode()
    print("--- Starting Synchronized Telemetry Test ---")

    time.sleep(1)   # Let the ring buffer fill

    try:
        while True:
            fake_camera_fire_time = time.time()
            sync_gps, sync_imu = telemetry.get_synchronized_data(
                fake_camera_fire_time
            )
            print(f"Sync GPS: Lat {sync_gps[0]:.6f}, Lon {sync_gps[1]:.6f}  "
                  f"| Sync Yaw: {math.degrees(sync_imu[2]):.1f} deg")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[PIXHAWK] Test stopped.")