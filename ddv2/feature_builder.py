"""Feature builder: converts BeamNG raw data into DDV2 model inputs."""
import numpy as np
import cv2


def build_sixcam_composite(images_dict, target_w=1024, target_h=256):
    """
    Stitch 6 BeamNG cameras horizontally into a 1024x256 composite.

    Camera order (360° around vehicle): FL -> F -> FR -> BR -> B -> BL
    Each camera gets ~170px width. Overlap blending at edges.

    Args:
        images_dict: {'F': ndarray(H,W,3), 'FL': ..., 'FR': ..., 'B': ..., 'BR': ..., 'BL': ...}
                     Images in BGR uint8 format (BeamNG raw output).
        target_w: output width (default 1024)
        target_h: output height (default 256)

    Returns:
        composite: ndarray (3, target_h, target_w) float32, normalized [0, 1], CHW, RGB
    """
    camera_order = ['FL', 'F', 'FR', 'BR', 'B', 'BL']
    per_cam_w = target_w // 6  # ~170px
    extra = target_w - per_cam_w * 6  # distribute remainder

    strips = []
    for i, cam_name in enumerate(camera_order):
        img = images_dict.get(cam_name)
        if img is None:
            img = np.zeros((200, 300, 3), dtype=np.uint8)

        w = per_cam_w + (1 if i < extra else 0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, target_h))
        img_norm = img_resized.astype(np.float32) / 255.0
        strips.append(img_norm)

    composite = np.concatenate(strips, axis=1)  # (target_h, target_w, 3)
    composite = np.transpose(composite, (2, 0, 1))  # (3, target_h, target_w)
    return composite.astype(np.float32)


def build_lidar_bev(radar_vehicles=None, grid_size=256, grid_range=(-32, 32)):
    """
    Build synthetic LiDAR BEV from IdealRadar vehicle detections.

    Without real LiDAR, radar vehicle positions are projected onto a 2D grid
    as Gaussian blobs. If no radar data is available, returns all-zeros BEV.

    Args:
        radar_vehicles: list of dicts with 'x', 'y' positions in meters (ego frame)
        grid_size: pixels per side (default 256)
        grid_range: (min, max) meters (default (-32, 32))

    Returns:
        bev: ndarray (1, grid_size, grid_size) float32
    """
    bev = np.zeros((grid_size, grid_size), dtype=np.float32)

    if radar_vehicles:
        meters_per_pixel = (grid_range[1] - grid_range[0]) / grid_size
        for rv in radar_vehicles:
            x_m, y_m = rv.get('x', 0), rv.get('y', 0)
            px = int((x_m - grid_range[0]) / meters_per_pixel)
            py = int((y_m - grid_range[0]) / meters_per_pixel)
            if 0 <= px < grid_size and 0 <= py < grid_size:
                # Gaussian blob with 2px sigma
                y_grid, x_grid = np.ogrid[:grid_size, :grid_size]
                gaussian = np.exp(-((x_grid - px)**2 + (y_grid - py)**2) / (2 * 2**2))
                bev = np.maximum(bev, gaussian.astype(np.float32))

    return bev[np.newaxis, ...]


def encode_driving_command(cmd_str):
    """
    Encode driving command string to 4-dim one-hot vector.

    Args:
        cmd_str: 'follow_lane', 'turn_left', 'turn_right', or 'keep_straight'

    Returns:
        cmd_vec: ndarray (4,) float32
    """
    commands = ['follow_lane', 'turn_left', 'turn_right', 'keep_straight']
    vec = np.zeros(4, dtype=np.float32)
    if cmd_str in commands:
        vec[commands.index(cmd_str)] = 1.0
    else:
        vec[0] = 1.0  # default: follow_lane
    return vec


def build_ego_status(speed_ms, acceleration=0.0, steering=0.0, driving_command='follow_lane'):
    """
    Build 8-dim ego status vector for DDV2.

    Format: [vx, vy, ax, ay, cmd_follow, cmd_left, cmd_right, cmd_straight]

    Args:
        speed_ms: forward speed in m/s (positive)
        acceleration: longitudinal acceleration in m/s^2
        steering: steering wheel angle (unused, placeholder)
        driving_command: navigation command string

    Returns:
        status: ndarray (8,) float32
    """
    lateral_speed = 0.0
    lateral_accel = 0.0

    cmd_vec = encode_driving_command(driving_command)

    status = np.array([
        speed_ms,        # vx - forward velocity
        lateral_speed,   # vy - lateral velocity
        acceleration,    # ax - longitudinal acceleration
        lateral_accel,   # ay - lateral acceleration
        cmd_vec[0],      # follow_lane
        cmd_vec[1],      # turn_left
        cmd_vec[2],      # turn_right
        cmd_vec[3],      # keep_straight
    ], dtype=np.float32)

    return status


def build_features(images_dict, speed_ms, acceleration=0.0, radar_vehicles=None,
                   driving_command='follow_lane'):
    """
    Build complete feature dict for one DDV2 inference step.

    Args:
        images_dict: {'F': ndarray, ...} — 6 camera images
        speed_ms: forward speed
        acceleration: longitudinal acceleration
        radar_vehicles: optional IdealRadar detections
        driving_command: navigation command string

    Returns:
        features: dict with 'camera_feature', 'lidar_feature', 'status_feature' tensors
                  Each has batch dim = 1, ready for model inference.
    """
    camera_feat = build_sixcam_composite(images_dict)
    lidar_feat = build_lidar_bev(radar_vehicles)
    status_feat = build_ego_status(speed_ms, acceleration, driving_command=driving_command)

    return {
        'camera_feature': camera_feat[np.newaxis, ...],   # (1, 3, 256, 1024)
        'lidar_feature': lidar_feat[np.newaxis, ...],     # (1, 1, 256, 256)
        'status_feature': status_feat[np.newaxis, ...],   # (1, 8)
    }
