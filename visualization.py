import cv2
import numpy as np

# Minimal canvas — speed > beauty
CANVAS_W = 960
CANVAS_H = 540

STATUS_H = 24
GRID_Y = STATUS_H
GRID_H = CANVAS_H - STATUS_H
GRID_COLS = 3
GRID_ROWS = 2
CELL_W = CANVAS_W // GRID_COLS
CELL_H = GRID_H // GRID_ROWS

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = 0.4
FONT_MD = 0.5

MODE_COLORS = {
    'TRAFFIC_AI': (0, 180, 80),
    'LKA+ACC': (80, 160, 220),
    'CUSTOM_CV': (200, 130, 0),
    'DDV2': (140, 60, 200),
    'DDV2+NAV': (140, 60, 200),
    'EMERGENCY': (200, 30, 30),
    'UNKNOWN': (100, 100, 100),
}

NAV_COLORS = {
    'turn_left': (230, 200, 0),
    'turn_right': (80, 200, 255),
    'follow_lane': (80, 220, 80),
    'keep_straight': (80, 220, 80),
}

NAV_ARROWS = {
    'turn_left': '<-- TURN LEFT',
    'turn_right': 'TURN RIGHT -->',
    'follow_lane': '^^ STRAIGHT ^^',
    'keep_straight': '^^ STRAIGHT ^^',
}

# Camera grid layout position — maps cam name to (col, row)
CAM_POS = {
    'FL': (0, 0), 'F': (1, 0), 'FR': (2, 0),
    'BL': (0, 1), 'B': (1, 1), 'BR': (2, 1),
}
CAM_LABELS = {
    'FL': 'FRONT-L', 'F': 'FRONT', 'FR': 'FRONT-R',
    'BL': 'BACK-L', 'B': 'BACK', 'BR': 'BACK-R',
}


def _make_blank(h, w, color=(20, 20, 25)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# ==============================================================================
# STATUS BAR
# ==============================================================================
def draw_status_bar(img, mode_label, speed_kmh, target_speed, steering, throttle,
                     brake, confidence, nav_command='', nav_distance=0.0):
    color = MODE_COLORS.get(mode_label, MODE_COLORS['UNKNOWN'])
    img[0:STATUS_H, 0:CANVAS_W] = color

    text = f"{mode_label} | {int(speed_kmh)} km/h"
    cv2.putText(img, text, (6, 17), FONT, FONT_MD, (255, 255, 255), 1)

    if target_speed > 0:
        cv2.putText(img, f"TGT:{int(target_speed)}", (220, 17), FONT, FONT_SM, (255, 255, 200), 1)

    cv2.putText(img, f"S:{steering:+.2f} T:{throttle:.2f} B:{brake:.2f}",
                (340, 17), FONT, FONT_SM, (220, 220, 220), 1)

    if nav_command:
        color_nav = NAV_COLORS.get(nav_command, (200, 200, 200))
        arrow = NAV_ARROWS.get(nav_command, '?')
        nav_text = f"{arrow} {nav_distance:.0f}m"
        (tw, _), _ = cv2.getTextSize(nav_text, FONT, FONT_SM, 1)
        cv2.putText(img, nav_text, (CANVAS_W - tw - 8, 17), FONT, FONT_SM, color_nav, 1)


# ==============================================================================
# CAMERA GRID (fast — just nearest-neighbor resize, no overlays)
# ==============================================================================
def draw_camera_grid(img, images_dict):
    for cam_name, (col, row) in CAM_POS.items():
        x0 = col * CELL_W
        y0 = GRID_Y + row * CELL_H
        feed = images_dict.get(cam_name)
        if feed is not None:
            cv2.resize(feed, (CELL_W, CELL_H), dst=img[y0:y0 + CELL_H, x0:x0 + CELL_W],
                       interpolation=cv2.INTER_NEAREST)
        else:
            img[y0:y0 + CELL_H, x0:x0 + CELL_W] = (40, 40, 50)
            cv2.putText(img, "NO SIG", (x0 + CELL_W // 2 - 30, y0 + CELL_H // 2),
                        FONT, FONT_SM, (0, 0, 255), 1)

        # Thin border
        cv2.rectangle(img, (x0, y0), (x0 + CELL_W - 1, y0 + CELL_H - 1), (60, 60, 70), 1)
        label = CAM_LABELS.get(cam_name, cam_name)
        cv2.putText(img, label, (x0 + 4, y0 + 14), FONT, FONT_SM, (200, 200, 200), 1)


# ==============================================================================
# LANE OVERLAY (only on front camera cell, only if lane data available)
# ==============================================================================
def draw_lane_overlay(img, lane_data):
    if lane_data is None:
        return

    col, row = CAM_POS['F']
    x0 = col * CELL_W
    y0 = GRID_Y + row * CELL_H

    roi_top = lane_data['roi_top']
    scale_y = CELL_H / lane_data['img_height']
    scale_x = CELL_W / lane_data['img_width']
    y_off = y0 + int(roi_top * scale_y)

    # Lane fill
    if lane_data.get('left_mean') is not None and lane_data.get('right_mean') is not None:
        lx = int(lane_data['left_mean'] * scale_x) + x0
        rx = int(lane_data['right_mean'] * scale_x) + x0
        pts = np.array([[lx, y_off], [lx, y0 + CELL_H], [rx, y0 + CELL_H], [rx, y_off]], dtype=np.int32)
        roi = img[y0:y0 + CELL_H, x0:x0 + CELL_W].copy()
        pts_roi = pts - [x0, y0]
        cv2.fillPoly(roi, [pts_roi], (30, 200, 30))
        cv2.addWeighted(roi, 0.2, img[y0:y0 + CELL_H, x0:x0 + CELL_W], 0.8, 0,
                        img[y0:y0 + CELL_H, x0:x0 + CELL_W])

    # Lane center line
    cx = x0 + int(lane_data['lane_center'] * scale_x)
    cv2.line(img, (cx, y0 + CELL_H), (cx, y_off), (80, 200, 255), 2)


# ==============================================================================
# MAIN RENDER
# ==============================================================================
def render_display(images_dict, lane_data, radar_boxes, radar_vehicles,
                   mode_label, confidence, level,
                   speed_kmh, target_speed, steering, throttle, brake,
                   frame, nav_command='', nav_distance=0.0):
    canvas = _make_blank(CANVAS_H, CANVAS_W)

    draw_camera_grid(canvas, images_dict)
    draw_lane_overlay(canvas, lane_data)
    draw_status_bar(canvas, mode_label, speed_kmh, target_speed, steering,
                    throttle, brake, confidence, nav_command, nav_distance)

    return canvas
