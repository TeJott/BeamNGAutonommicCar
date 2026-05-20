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
        self.steering_kp = 1.0
        self._speed_integral = 0.0
        self._max_integral = 10.0

    def select_trajectory(self, trajectories, scores, vehicle_state=None):
        """
        Select best trajectory from K proposals.

        For initial integration: picks the highest-scoring trajectory.
        Future: add consistency check with current vehicle state.
        """
        if scores.ndim > 1:
            scores = scores.flatten()
        best_idx = int(scores.argmax())
        return trajectories[0, best_idx]

    def trajectory_to_control(self, trajectory, current_speed_ms):
        """
        Convert first trajectory waypoint to steering/throttle/brake.

        Args:
            trajectory: (T, 3) array of (x, y, yaw) waypoints
            current_speed_ms: current vehicle speed in m/s

        Returns:
            steering: float [-1, 1]
            throttle: float [0, 1]
            brake: float [0, 1]
        """
        lookahead_x, lookahead_y, target_yaw = trajectory[0]

        lookahead_dist = math.sqrt(lookahead_x**2 + lookahead_y**2)
        lookahead_dist = max(0.5, lookahead_dist)

        alpha = math.atan2(lookahead_y, lookahead_x)
        steering_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha), lookahead_dist
        )
        steering = max(-1.0, min(1.0, steering_angle * self.steering_kp / (math.pi / 4)))

        desired_speed = lookahead_dist / self.dt
        desired_speed = max(3.0, min(30.0, desired_speed))

        speed_error = desired_speed - current_speed_ms
        self._speed_integral += speed_error * self.dt
        self._speed_integral = max(-self._max_integral, min(self._max_integral, self._speed_integral))

        throttle = max(0.0, min(1.0, self.speed_kp * speed_error + self.speed_ki * self._speed_integral))

        brake = 0.0
        if speed_error < -2.0:
            brake = min(0.5, abs(speed_error) * 0.05)

        return steering, throttle, brake

    def interpolate_control(self, trajectory, step, total_skip):
        """
        Interpolate controls for intermediate frames when DDV2 runs every N frames.

        Args:
            trajectory: selected trajectory (T, 3)
            step: current sub-frame index (0 to total_skip-1)
            total_skip: total frames between DDV2 inferences

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
        current_speed = math.sqrt(interp_x**2 + interp_y**2) / self.dt

        return self.trajectory_to_control(interp_traj, current_speed)

    def reset(self):
        self._speed_integral = 0.0
