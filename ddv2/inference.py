import numpy as np

from ddv2.camera_adapter import BeamNGCameraAdapter
from ddv2.model_loader import DDV2Model
from ddv2.trajectory_to_control import TrajectoryController


class DDV2Inference:
    """
    Complete DDV2 inference pipeline for BeamNG autonomous driving.

    Ties together camera adaptation, model inference, and trajectory-to-control
    conversion. Designed for frame-skipped operation (e.g., run every 3rd frame)
    to accommodate ~100-200ms inference latency.

    Usage:
        ddv2 = DDV2Inference(checkpoint_path='./checkpoints/ddv2_model.pth')
        if ddv2.load():
            for frame in range(1000):
                images = get_camera_images()
                if frame % ddv2.frame_skip == 0:
                    steering, throttle, brake = ddv2.step(images, speed_ms)
                else:
                    steering, throttle, brake = ddv2.interpolate(frame)
                vehicle.control(steering=steering, throttle=throttle, brake=brake)
    """

    def __init__(self, checkpoint_path=None, device=None, frame_skip=3,
                 wheelbase=2.8, target_resolution=(224, 400)):
        self.adapter = BeamNGCameraAdapter(target_resolution=target_resolution)
        self.model = DDV2Model(checkpoint_path=checkpoint_path, device=device)
        self.traj_ctrl = TrajectoryController(wheelbase=wheelbase)
        self.frame_skip = frame_skip
        self.current_trajectory = None
        self.step_in_frame = 0

    def load(self):
        return self.model.load()

    def is_available(self):
        return self.model.is_available()

    def step(self, images_dict, speed_ms):
        """
        Run full inference step. Call every frame_skip frames.
        """
        images_tensor, intrinsics, extrinsics = self.adapter.adapt(images_dict)
        images_tensor = images_tensor[np.newaxis]  # add batch dim

        trajectories, scores = self.model.predict(images_tensor, intrinsics, extrinsics)
        self.current_trajectory = self.traj_ctrl.select_trajectory(trajectories, scores)
        self.step_in_frame = 0

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
        self.step_in_frame = 0
        self.traj_ctrl.reset()
