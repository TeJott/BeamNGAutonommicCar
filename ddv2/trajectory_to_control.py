import math


class TrajectoryController:
    """
    Converts DDV2 trajectory predictions into BeamNG vehicle control commands.

    Uses a pure pursuit controller: looks ahead to the first waypoint of the
    selected trajectory and computes steering angle to reach it, plus
    throttle/brake to match desired speed.
    """

    def __init__(self, dt=0.1, wheelbase=2.8):
        self.dt = dt
        self.wheelbase = wheelbase
        self.speed_kp = 0.6
        self.speed_ki = 0.1
        self.steering_kp = 0.5
        self._speed_integral = 0.0
        self._max_integral = 10.0
        self._prev_steering = 0.0
        self._steering_ema = 0.0
        self._ema_alpha = 0.4

    def select_trajectory(self, trajectories, scores, vehicle_state=None):
        """
        Select best trajectory from K proposals.

        For initial integration: picks the highest-scoring trajectory.
        Future: add consistency check with current vehicle state.

        Args:
            trajectories: (K, T, 3) array
            scores: (K,) array
        """
        if scores.ndim > 1:
            scores = scores.flatten()
        best_idx = int(scores.argmax())
        return trajectories[best_idx]  # (T, 3)

    def trajectory_to_control(self, trajectory, current_speed_ms):
        """
        Convert DDV2 trajectory to steering/throttle/brake.

        DDV2 outputs 8 waypoints over 4s (0.5s interval) in ego frame.
        Steering uses the first waypoint (0.5s) for immediate response.
        Speed uses waypoint at ~2s (index 3) for stable control — using
        the closest waypoint causes constant braking at highway speeds.

        Args:
            trajectory: (T, 3) array of (x, y, yaw) waypoints
            current_speed_ms: current vehicle speed in m/s

        Returns:
            steering: float [-1, 1]
            throttle: float [0, 1]
            brake: float [0, 1]
        """
        T = len(trajectory)

        # Steering: use closest waypoint (0.5s) for immediate response
        steer_x, steer_y, _ = trajectory[0]
        steer_dist = math.sqrt(steer_x**2 + steer_y**2)
        steer_dist = max(0.5, steer_dist)

        alpha = math.atan2(steer_y, steer_x)
        steering_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha), steer_dist
        )
        raw_steering = max(-1.0, min(1.0, steering_angle * self.steering_kp / (math.pi / 4)))

        # EMA smoothing to prevent wild oscillations
        self._steering_ema = self._ema_alpha * raw_steering + (1 - self._ema_alpha) * self._steering_ema
        steering = max(-1.0, min(1.0, self._steering_ema))

        # Rate limit: max 0.3 change per call
        max_delta = 0.3
        steering = self._prev_steering + max(-max_delta, min(max_delta, steering - self._prev_steering))
        self._prev_steering = steering

        # Speed: use waypoint at ~2s (index 3) with matching dt
        speed_idx = min(3, T - 1)
        speed_x, speed_y, _ = trajectory[speed_idx]
        speed_dist = math.sqrt(speed_x**2 + speed_y**2)
        speed_dt = 0.5 * (speed_idx + 1)  # time to reach this waypoint
        speed_dist = max(2.0, speed_dist)  # minimum 2m ahead

        desired_speed = speed_dist / speed_dt
        desired_speed = max(2.0, min(30.0, desired_speed))

        speed_error = desired_speed - current_speed_ms
        self._speed_integral += speed_error * self.dt
        self._speed_integral = max(-self._max_integral, min(self._max_integral, self._speed_integral))

        throttle = max(0.0, min(1.0, self.speed_kp * speed_error + self.speed_ki * self._speed_integral))

        brake = 0.0
        if speed_error < -2.0:
            brake = min(0.5, abs(speed_error) * 0.05)

        return steering, throttle, brake

    def interpolate_control(self, trajectory, step, total_skip, current_speed_ms):
        """
        Interpolate controls for intermediate frames when DDV2 runs every N frames.

        Args:
            trajectory: selected trajectory (T, 3)
            step: current sub-frame index (0 to total_skip-1)
            total_skip: total frames between DDV2 inferences
            current_speed_ms: REAL vehicle speed in m/s (not derived from waypoints)

        Returns:
            interpolated steering, throttle, brake
        """
        alpha = step / total_skip
        idx = min(int(alpha * (len(trajectory) - 1)), len(trajectory) - 2)
        fraction = alpha * (len(trajectory) - 1) - idx

        wp1 = trajectory[idx]
        wp2 = trajectory[min(idx + 1, len(trajectory) - 1)]

        interp_x = wp1[0] + fraction * (wp2[0] - wp1[0])
        interp_y = wp1[1] + fraction * (wp2[1] - wp1[1])
        interp_yaw = wp1[2] + fraction * (wp2[2] - wp1[2])

        interp_traj = [(interp_x, interp_y, interp_yaw)] + list(trajectory[idx + 1:])

        return self.trajectory_to_control(interp_traj, current_speed_ms)

    def reset(self):
        self._speed_integral = 0.0
        self._prev_steering = 0.0
        self._steering_ema = 0.0
