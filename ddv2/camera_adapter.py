import numpy as np
import cv2


class BeamNGCameraAdapter:
    """
    Adapts BeamNG 6-camera setup to DDV2's 360-degree surround format.

    DDV2 expects a single 1024x256 panorama from all 6 cameras, ordered
    around the vehicle: FL -> F -> FR -> BR -> B -> BL.

    Each camera receives ~170px width. The model was trained on 3-camera
    NAVSIM panoramas but handles 6 cameras in the same resolution.
    """

    CAMERA_ORDER = ['FL', 'F', 'FR', 'BR', 'B', 'BL']
    PANORAMA_WIDTH = 1024
    PANORAMA_HEIGHT = 256

    def __init__(self, target_w=1024, target_h=256):
        self.target_w = target_w
        self.target_h = target_h

    def build_composite(self, images_dict):
        """
        Build 6-camera composite panorama for DDV2.

        Args:
            images_dict: {'F': ndarray(H,W,3), 'FL': ..., 'FR': ...,
                          'B': ..., 'BR': ..., 'BL': ...}
                         BGR uint8, native BeamNG resolution.

        Returns:
            composite: ndarray (3, target_h, target_w) float32, CHW, RGB, [0,1]
        """
        per_cam_w = self.target_w // 6
        extra = self.target_w - per_cam_w * 6

        strips = []
        for i, cam_name in enumerate(self.CAMERA_ORDER):
            img = images_dict.get(cam_name)
            if img is None:
                img = np.zeros((200, 300, 3), dtype=np.uint8)

            w = per_cam_w + (1 if i < extra else 0)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (w, self.target_h))
            img_norm = img_resized.astype(np.float32) / 255.0
            strips.append(img_norm)

        composite = np.concatenate(strips, axis=1)
        composite = np.transpose(composite, (2, 0, 1))
        return composite.astype(np.float32)

    def adapt(self, images_dict, camera_specs=None):
        """
        Build DDV2 input composite from BeamNG camera images.

        Returns:
            composite: (3, target_h, target_w) CHW float32, RGB, [0,1]
        """
        return self.build_composite(images_dict)
