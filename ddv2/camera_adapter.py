import cv2
import numpy as np


class BeamNGCameraAdapter:
    """
    Adapts BeamNG 6-camera setup to DDV2's expected NAVSIM-format input.

    DDV2 expects:
    - Multi-camera input (typically 6 cameras)
    - Resolution at least 224x224 (ResNet-34 backbone minimum)
    - RGB channel order
    - Normalized pixel values [0, 1]
    - Camera intrinsics and extrinsics matrices

    BeamNG provides:
    - 300x200 BGR images via shared memory
    - Camera positions and directions as (x, y, z) vectors
    - No explicit intrinsics (must estimate from FOV)
    """

    CAMERA_ORDER = ['F', 'FR', 'FL', 'B', 'BR', 'BL']

    def __init__(self, target_resolution=(224, 400)):
        self.target_h, self.target_w = target_resolution
        self.default_fov_y = 70.0

    def adapt(self, images_dict, camera_specs=None):
        """
        Convert BeamNG camera images to DDV2 tensor format.

        Args:
            images_dict: {'F': ndarray(200,300,3), 'FL': ..., ...}
            camera_specs: Optional dict of camera positions/directions

        Returns:
            images_tensor: numpy array of shape (6, 3, H, W) normalized [0,1]
            intrinsics: dummy intrinsics matrix (6, 3, 3)
            extrinsics: dummy extrinsics matrix (6, 4, 4)
        """
        batch = []
        for cam_name in self.CAMERA_ORDER:
            img = images_dict.get(cam_name)
            if img is None:
                img = np.zeros((200, 300, 3), dtype=np.uint8)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_norm, (2, 0, 1))
            batch.append(img_chw)

        images_tensor = np.stack(batch, axis=0)

        intrinsics = self._estimate_intrinsics(self.target_h, self.target_w)
        intrinsics_batch = np.stack([intrinsics] * 6, axis=0)

        extrinsics = self._estimate_extrinsics(camera_specs)
        extrinsics_batch = np.stack([extrinsics] * 6, axis=0) if extrinsics is not None \
            else np.eye(4)[np.newaxis].repeat(6, axis=0)

        return images_tensor, intrinsics_batch, extrinsics_batch

    def _estimate_intrinsics(self, h, w):
        fov_y_rad = np.radians(self.default_fov_y)
        fy = (h / 2.0) / np.tan(fov_y_rad / 2.0)
        fx = fy
        cx = w / 2.0
        cy = h / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        return K

    def _estimate_extrinsics(self, camera_specs):
        if camera_specs is None:
            return None

        extrinsics = {}
        for name, spec in camera_specs.items():
            pos = np.array(spec['pos'], dtype=np.float32)
            forward = np.array(spec['dir'], dtype=np.float32)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            up = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)

            Rt = np.eye(4, dtype=np.float32)
            Rt[0, :3] = right
            Rt[1, :3] = forward
            Rt[2, :3] = up
            Rt[:3, 3] = pos
            extrinsics[name] = Rt

        return extrinsics
