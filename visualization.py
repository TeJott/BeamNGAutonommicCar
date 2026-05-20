import cv2
import numpy as np


def build_camera_grid(images_dict):
    try:
        top_row = cv2.hconcat([images_dict['FL'], images_dict['F'], images_dict['FR']])
        bottom_row = cv2.hconcat([images_dict['BL'], images_dict['B'], images_dict['BR']])
        camera_view = cv2.vconcat([top_row, bottom_row])
    except Exception:
        camera_view = np.zeros((400, 900, 3), dtype=np.uint8)
        cv2.putText(camera_view, "CAMERA MERGE ERROR", (250, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return camera_view


def draw_hud(camera_view, speed_kmh, target_speed, steering, throttle, brake, frame):
    titles = ['FRONT LEFT', 'FRONT (LANE TRACK)', 'FRONT RIGHT',
              'BACK LEFT', 'BACK', 'BACK RIGHT']
    h, w = 200, 300
    for i, title in enumerate(titles):
        x = (i % 3) * w + 10
        y = (i // 3) * h + 25
        cv2.putText(camera_view, title, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(camera_view, f"SPEED: {int(speed_kmh)} km/h", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(camera_view, f"TARGET: {int(target_speed)} km/h", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(camera_view, f"STEER: {steering:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
    cv2.putText(camera_view, f"THR: {throttle:.2f} BRK: {brake:.2f}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(camera_view, f"FRAME: {frame}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return camera_view
