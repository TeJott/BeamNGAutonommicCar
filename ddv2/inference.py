import numpy as np

from ddv2.camera_adapter import BeamNGCameraAdapter
from ddv2.feature_builder import build_lidar_bev, build_ego_status
from ddv2.model_loader import DDV2Model
from ddv2.trajectory_to_control import TrajectoryController


class DDV2Inference:
    """
    Complete DDV2 inference pipeline for BeamNG autonomous driving.

    Ties together camera adaptation, feature building, model inference,
    and trajectory-to-control conversion. Designed for frame-skipped operation.

    Usage:
        ddv2 = DDV2Inference()
        if ddv2.load():
            for frame in range(1000):
                images = get_camera_images()
                if frame % ddv2.frame_skip == 0:
                    steering, throttle, brake = ddv2.step(
                        images, speed_ms, driving_command='follow_lane'
                    )
                else:
                    steering, throttle, brake = ddv2.interpolate(frame)
    """

    def __init__(self, checkpoint_path=None, device=None, frame_skip=3, wheelbase=2.8):
        self.adapter = BeamNGCameraAdapter()
        self.model = DDV2Model(checkpoint_path=checkpoint_path, device=device)
        self.traj_ctrl = TrajectoryController(wheelbase=wheelbase)
        self.frame_skip = frame_skip
        self.current_trajectory = None

    def load(self):
        return self.model.load()

    def is_available(self):
        return self.model.is_available()

    def step(self, images_dict, speed_ms, acceleration=0.0,
             radar_vehicles=None, driving_command='follow_lane'):
        """
        Run full inference step. Call every frame_skip frames.

        Args:
            images_dict: 6-camera images from BeamNG {'F': ndarray, ...}
            speed_ms: forward speed in m/s
            acceleration: longitudinal acceleration in m/s^2
            radar_vehicles: optional list of radar detections for BEV
            driving_command: 'follow_lane', 'turn_left', 'turn_right', or 'keep_straight'

        Returns:
            steering, throttle, brake
        """
        composite = self.adapter.build_composite(images_dict)
        lidar_bev = build_lidar_bev(radar_vehicles)
        ego_status = build_ego_status(speed_ms, acceleration, driving_command=driving_command)

        features = {
            'camera_feature': composite[np.newaxis, ...],
            'lidar_feature': lidar_bev[np.newaxis, ...],
            'status_feature': ego_status[np.newaxis, ...],
        }

        trajectories, scores = self.model.predict(features)
        self.current_trajectory = self.traj_ctrl.select_trajectory(trajectories, scores)

        return self.traj_ctrl.trajectory_to_control(self.current_trajectory, speed_ms)

    def interpolate(self, sub_frame):
        """Interpolate controls between DDV2 inference steps."""
        if self.current_trajectory is None:
            return 0.0, 0.0, 0.0
        return self.traj_ctrl.interpolate_control(
            self.current_trajectory, sub_frame % self.frame_skip, self.frame_skip
        )

    def reset(self):
        self.current_trajectory = None
        self.traj_ctrl.reset()
