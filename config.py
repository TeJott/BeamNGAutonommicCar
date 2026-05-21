BEAMNG_HOME = 'G:\\Gry\\BeamNG.tech.v0.38.5.0'
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 64256

VEHICLE_NAME = 'ego_vehicle'
VEHICLE_MODEL = 'etki'
MAP_NAME = 'west_coast_usa'
SCENARIO_NAME = 'multicamera_autonomous_drive'

SPAWN_POS = (-1045.877197, -518.1680298, 101.8693466)
SPAWN_YAW_DEG = -135.678

CAM_RESOLUTION = (300, 200)
CAM_UPDATE_TIME = 0.01
CAM_HEIGHT = 1.3

CAM_SPECS = {
    'B':  {'pos': (0, 1.6, CAM_HEIGHT),   'dir': (0, 1, 0)},
    'BR': {'pos': (-0.8, 1.2, CAM_HEIGHT), 'dir': (-1, 1, 0)},
    'BL': {'pos': (0.8, 1.2, CAM_HEIGHT),  'dir': (1, 1, 0)},
    'F':  {'pos': (0, -1.6, CAM_HEIGHT),  'dir': (0, -1, 0)},
    'FR': {'pos': (-0.8, -1.2, CAM_HEIGHT), 'dir': (-1, -1, 0)},
    'FL': {'pos': (0.8, -1.2, CAM_HEIGHT),  'dir': (1, -1, 0)},
}

LANE_DETECTION = {
    'roi_top_ratio': 0.55,
    'clahe_clip': 2.0,
    'clahe_grid': (8, 8),
    'canny_low_factor': 0.5,
    'canny_high_factor': 1.2,
    'canny_low_min': 20,
    'canny_high_min': 50,
    'hough_threshold': 30,
    'hough_min_line': 20,
    'hough_max_gap': 40,
    'slope_min_abs': 0.1,
    'slope_threshold': 0.3,
    'assumed_lane_width_px': 80,
    'ema_alpha': 0.3,
    'steering_kp': 0.007,
    'confidence_min_kp': 0.3,
}

SPEED_CONTROL = {
    'max_speed_kmh': 100.0,
    'min_speed_kmh': 25.0,
    'default_speed_kmh': 90.0,
    'curvature_factor': 50.0,
    'kp': 0.015,
    'ki': 0.003,
    'integral_max': 20.0,
    'brake_threshold_kmh': 5.0,
    'brake_kp': 0.015,
    'brake_max': 0.6,
}

DISPLAY = {
    'window_name': '360 Camera View',
    'window_width': 1150,
    'window_height': 780,
    'frame_interval': 50,
}

NAVIGATION = {
    'enabled': True,
    'auto_distance': 500.0,
    'maneuver_lookahead': 3,
    'junction_angle_threshold': 30.0,
}

DDV2 = {
    'frame_skip': 3,
    'checkpoint_path': 'ddv2/checkpoints/diffusiondrivev2_sel.ckpt',
}
