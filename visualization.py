import cv2
import math
import numpy as np

CANVAS_W = 1100
CANVAS_H = 720

MODE_BAR_H = 30
MAIN_Y = MODE_BAR_H
MAIN_H = 440
FRONT_W = 660
RIGHT_X = FRONT_W
RIGHT_W = CANVAS_W - FRONT_W
BIRD_H = 220
SPEEDO_Y = MAIN_Y + BIRD_H
SPEEDO_H = 220
SMALL_Y = MAIN_Y + MAIN_H
SMALL_H = 160
SMALL_W = CANVAS_W // 5
TELEM_Y = SMALL_Y + SMALL_H
TELEM_H = 40
TRAJ_Y = TELEM_Y + TELEM_H
TRAJ_H = 50

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = 0.4
FONT_MD = 0.55
FONT_LG = 0.7

MODE_COLORS = {
    'TRAFFIC_AI': (0, 180, 80),
    'LKA+ACC': (80, 160, 220),
    'CUSTOM_CV': (200, 130, 0),
    'DDV2': (140, 60, 200),
    'EMERGENCY': (200, 30, 30),
    'UNKNOWN': (100, 100, 100),
}

COLOR_LEFT_LANE = (80, 220, 80)
COLOR_RIGHT_LANE = (220, 130, 50)
COLOR_LANE_CENTER = (80, 200, 255)
COLOR_VEHICLE_CENTER = (255, 50, 50)
COLOR_LANE_FILL = (30, 200, 30)
COLOR_BBOX = (50, 50, 255)
COLOR_CANVAS_BG = (25, 25, 30)
COLOR_PANEL_BG = (40, 40, 50)
COLOR_TELEM_BG = (30, 30, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_GAUGE_BG = (35, 35, 42)
COLOR_NEEDLE = (255, 80, 80)
COLOR_TARGET_NEEDLE = (255, 200, 50)
COLOR_GREEN = (0, 220, 100)
COLOR_RED = (220, 40, 40)
COLOR_YELLOW = (230, 200, 0)
COLOR_THROTTLE = (80, 220, 80)
COLOR_BRAKE = (40, 100, 255)
COLOR_STEER_BAR = (220, 180, 40)


def _make_blank(h, w, color=COLOR_CANVAS_BG):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# ==============================================================================
# MODE BAR
# ==============================================================================
def draw_mode_bar(img, mode_label, confidence=0.0, level=0):
    color = MODE_COLORS.get(mode_label, MODE_COLORS['UNKNOWN'])
    overlay = img[0:MODE_BAR_H, 0:CANVAS_W]
    overlay[:] = color

    text = f"MODE: {mode_label}"
    cv2.putText(img, text, (12, 22), FONT, FONT_MD, COLOR_WHITE, 1)
    cv2.putText(img, f"CONF: {confidence:.1%}", (300, 22), FONT, FONT_SM, COLOR_WHITE, 1)
    cv2.putText(img, f"LEVEL: {level}", (500, 22), FONT, FONT_SM, COLOR_WHITE, 1)

    keys = " [1]TrafficAI  [2]CustomCV  [3]LKA+ACC  [4]DDV2  [0]STOP  [q]QUIT"
    cv2.putText(img, keys, (620, 22), FONT, FONT_SM, (200, 200, 200), 1)


# ==============================================================================
# FRONT CAMERA WITH LANE OVERLAYS + VEHICLE BOUNDING BOXES
# ==============================================================================
def draw_front_camera(img, front_bgr, lane_data, radar_boxes):
    h, w = MAIN_H, FRONT_W
    if front_bgr is not None:
        resized = cv2.resize(front_bgr, (w, h))
        img[MAIN_Y:MAIN_Y + h, 0:w] = resized
    else:
        img[MAIN_Y:MAIN_Y + h, 0:w] = COLOR_PANEL_BG
        cv2.putText(img, "NO SIGNAL", (w // 3, MAIN_Y + h // 2),
                    FONT, FONT_LG, COLOR_RED, 2)

    if lane_data is None:
        return

    roi_top = lane_data['roi_top']
    roi_scale_y = h / lane_data['img_height']
    roi_scale_x = w / lane_data['img_width']
    y_offset = MAIN_Y + int(roi_top * roi_scale_y)

    left_x = lane_data.get('left_x', [])
    right_x = lane_data.get('right_x', [])
    lane_center = lane_data['lane_center']
    img_center_px = lane_data['img_width'] / 2.0

    # Translucent lane fill between left and right lanes
    if lane_data.get('left_mean') and lane_data.get('right_mean'):
        left_screen = int(lane_data['left_mean'] * roi_scale_x)
        right_screen = int(lane_data['right_mean'] * roi_scale_x)
        pts = np.array([
            [left_screen, y_offset],
            [left_screen, MAIN_Y + h],
            [right_screen, MAIN_Y + h],
            [right_screen, y_offset],
        ], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (30, 200, 30))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Lane points
    for lx in left_x:
        sx = int(lx * roi_scale_x)
        sy = int(y_offset + (h - (y_offset - MAIN_Y)) * 0.6)
        cv2.circle(img, (sx, sy), 2, COLOR_LEFT_LANE, -1)
    for rx in right_x:
        sx = int(rx * roi_scale_x)
        sy = int(y_offset + (h - (y_offset - MAIN_Y)) * 0.6)
        cv2.circle(img, (sx, sy), 2, COLOR_RIGHT_LANE, -1)

    # Lane center line
    cx = int(lane_center * roi_scale_x)
    cv2.line(img, (cx, MAIN_Y + h - 5), (cx, y_offset), COLOR_LANE_CENTER, 2)

    # Vehicle vertical center (red dashed)
    vcx = int(img_center_px * roi_scale_x)
    for dy in range(y_offset + 10, MAIN_Y + h, 15):
        cv2.line(img, (vcx, dy), (vcx, min(dy + 8, MAIN_Y + h)), COLOR_VEHICLE_CENTER, 1)

    # Detected vehicles from radar
    for (bx, by, bw, bh, dist) in radar_boxes:
        screen_bx = int(bx * w / 300)
        screen_by = int(by * h / 200) + MAIN_Y
        screen_bw = int(bw * w / 300)
        screen_bh = int(bh * h / 200)
        cv2.rectangle(img, (screen_bx, screen_by),
                      (screen_bx + screen_bw, screen_by + screen_bh), COLOR_BBOX, 2)
        cv2.putText(img, f"{dist:.0f}m", (screen_bx, screen_by - 5),
                    FONT, FONT_SM, COLOR_BBOX, 1)

    # Label
    cv2.putText(img, "FRONT CAMERA", (8, MAIN_Y + 20), FONT, FONT_SM, COLOR_WHITE, 1)
    cv2.putText(img, f"curv: {lane_data['curvature']:.3f}", (8, MAIN_Y + 40),
                FONT, FONT_SM, COLOR_TEXT, 1)


# ==============================================================================
# BIRD'S EYE MINIMAP
# ==============================================================================
def draw_birds_eye(img, lane_data, radar_vehicles, speed_kmh):
    x0, y0 = RIGHT_X, MAIN_Y
    bw, bh = RIGHT_W, BIRD_H
    sub = img[y0:y0 + bh, x0:x0 + bw]
    sub[:] = COLOR_GAUGE_BG

    cx_bird = bw // 2
    cy_bird = bh - 30
    scale = 2.5

    # Road surface
    road_left = cx_bird - 40
    road_right = cx_bird + 40
    cv2.rectangle(sub, (road_left, 0), (road_right, bh), (60, 60, 70), -1)

    # Lane markings (dashed center line)
    for dy in range(0, bh, 18):
        cv2.line(sub, (cx_bird, dy), (cx_bird, min(dy + 10, bh)), COLOR_WHITE, 1)

    # Curved lane lines from curvature data
    if lane_data and lane_data.get('left_mean') and lane_data.get('right_mean'):
        curv = lane_data['curvature']
        for dy in range(0, bh, 3):
            offset = curv * (dy / bh) * (dy / bh) * 120
            lx = road_left + int(offset)
            rx = road_right + int(offset)
            if 0 <= lx < bw and 0 <= rx < bw:
                c = (80, 220, 80) if dy % 12 < 6 else (60, 180, 60)
                sub[max(0, bh - dy - 1), max(0, lx)] = c
                sub[max(0, bh - dy - 1), min(bw - 1, rx)] = (220, 130, 50)

    # Ego vehicle
    car_pts = np.array([
        [cx_bird - 10, cy_bird - 18],
        [cx_bird + 10, cy_bird - 18],
        [cx_bird + 10, cy_bird + 4],
        [cx_bird + 6, cy_bird + 8],
        [cx_bird - 6, cy_bird + 8],
        [cx_bird - 10, cy_bird + 4],
    ], dtype=np.int32)
    cv2.fillPoly(sub, [car_pts], (255, 220, 40))
    cv2.polylines(sub, [car_pts], True, (200, 160, 20), 1)

    # Other vehicles from radar
    for v in radar_vehicles:
        dist = abs(v.get('rel_dist', 0))
        if dist < 1 or dist > 80:
            continue
        vy = cy_bird - int(dist * scale)
        vx = cx_bird + int(v.get('rel_pos_y', 0) * scale * 10)
        vx = max(5, min(bw - 5, vx))
        if 10 < vy < bh - 10:
            cv2.rectangle(sub, (vx - 5, vy - 8), (vx + 5, vy + 3), COLOR_RED, -1)

    # Rays from vehicle forward
    for angle in [-0.3, 0, 0.3]:
        end_y = cy_bird - int(bh * 0.7)
        end_x = cx_bird + int(angle * bh * 0.7)
        cv2.line(sub, (cx_bird, cy_bird), (end_x, max(0, end_y)), (100, 100, 120), 1)

    # Border and label
    cv2.rectangle(sub, (0, 0), (bw - 1, bh - 1), (80, 80, 90), 1)
    cv2.putText(img, "BIRD'S EYE", (x0 + 6, y0 + 16), FONT, FONT_SM, COLOR_WHITE, 1)
    cv2.putText(img, f"{int(speed_kmh)} km/h", (x0 + bw - 80, y0 + 16),
                FONT, FONT_SM, COLOR_GREEN, 1)


# ==============================================================================
# SPEEDOMETER GAUGE
# ==============================================================================
def draw_speedometer(img, speed_kmh, target_speed):
    x0, y0 = RIGHT_X, SPEEDO_Y
    sw, sh = RIGHT_W, SPEEDO_H
    sub = img[y0:y0 + sh, x0:x0 + sw]
    sub[:] = COLOR_GAUGE_BG

    cx = sw // 2
    cy = sh - 30
    radius = min(sw, sh) - 35
    max_speed = 140

    # Arc background
    for angle_deg in range(180, 0, -2):
        speed_val = (180 - angle_deg) / 180.0 * max_speed
        ang = math.radians(angle_deg)
        ex = int(cx + radius * math.cos(ang))
        ey = int(cy - radius * math.sin(ang))

        if speed_val <= target_speed:
            color = (50, 180, 50) if speed_val <= 100 else (180, 130, 0)
        else:
            color = (60, 60, 60)

        cv2.circle(sub, (ex, ey), 2, color, -1)

    # Tick marks
    for speed_val in [0, 20, 40, 60, 80, 100, 120, 140]:
        angle_deg = 180 - (speed_val / max_speed) * 180
        ang = math.radians(angle_deg)
        inner_r = radius - 12
        outer_r = radius + 2
        ix = int(cx + inner_r * math.cos(ang))
        iy = int(cy - inner_r * math.sin(ang))
        ox = int(cx + outer_r * math.cos(ang))
        oy = int(cy - outer_r * math.sin(ang))
        cv2.line(sub, (ix, iy), (ox, oy), COLOR_WHITE, 2)
        tx = int(cx + (radius - 22) * math.cos(ang))
        ty = int(cy - (radius - 22) * math.sin(ang))
        cv2.putText(sub, str(speed_val), (tx - 12, ty + 4), FONT, FONT_SM, COLOR_WHITE, 1)

    # Needle (current speed)
    speed_clamped = min(speed_kmh, max_speed)
    needle_angle = math.radians(180 - (speed_clamped / max_speed) * 180)
    nx = int(cx + (radius - 30) * math.cos(needle_angle))
    ny = int(cy - (radius - 30) * math.sin(needle_angle))
    cv2.line(sub, (cx, cy), (nx, ny), COLOR_NEEDLE, 3)
    cv2.circle(sub, (cx, cy), 6, COLOR_NEEDLE, -1)

    # Target speed marker
    if target_speed > 0:
        target_clamped = min(target_speed, max_speed)
        tang = math.radians(180 - (target_clamped / max_speed) * 180)
        tx = int(cx + radius * math.cos(tang))
        ty = int(cy - radius * math.sin(tang))
        cv2.drawMarker(sub, (tx, ty), COLOR_TARGET_NEEDLE, cv2.MARKER_TRIANGLE_UP, 8, 2)

    # Digital readout
    cv2.putText(sub, f"{int(speed_kmh)}", (cx - 30, cy + 20),
                FONT, FONT_LG, COLOR_GREEN, 2)
    cv2.putText(sub, "km/h", (cx + 10, cy + 22), FONT, FONT_SM, COLOR_TEXT, 1)
    cv2.putText(sub, f"TGT {int(target_speed)}", (cx - 25, sh - 8),
                FONT, FONT_SM, COLOR_TARGET_NEEDLE, 1)

    cv2.rectangle(sub, (0, 0), (sw - 1, sh - 1), (80, 80, 90), 1)
    cv2.putText(img, "SPEED", (x0 + 6, y0 + 16), FONT, FONT_SM, COLOR_WHITE, 1)


# ==============================================================================
# SMALL CAMERA FEEDS
# ==============================================================================
def draw_small_cameras(img, images_dict):
    cams = ['FL', 'FR', 'B', 'BL', 'BR']
    labels = ['FRONT-LEFT', 'FRONT-RIGHT', 'BACK', 'BACK-LEFT', 'BACK-RIGHT']
    for i, (key, label) in enumerate(zip(cams, labels)):
        x0 = i * SMALL_W
        y0 = SMALL_Y
        feed = images_dict.get(key)
        if feed is not None:
            resized = cv2.resize(feed, (SMALL_W, SMALL_H))
            img[y0:y0 + SMALL_H, x0:x0 + SMALL_W] = resized
        else:
            img[y0:y0 + SMALL_H, x0:x0 + SMALL_W] = COLOR_PANEL_BG

        cv2.rectangle(img, (x0, y0), (x0 + SMALL_W - 1, y0 + SMALL_H - 1), (80, 80, 90), 1)
        cv2.putText(img, label, (x0 + 4, y0 + 14), FONT, FONT_SM, COLOR_WHITE, 1)


# ==============================================================================
# TELEMETRY BAR
# ==============================================================================
def draw_telemetry(img, throttle, brake, steering, frame, speed_kmh, target_speed):
    y0 = TELEM_Y
    h = TELEM_H

    img[y0:y0 + h, 0:CANVAS_W] = COLOR_TELEM_BG
    cv2.line(img, (0, y0), (CANVAS_W, y0), (60, 60, 70), 1)

    bar_y = y0 + 8
    bar_h = h - 16

    # Throttle bar
    cv2.putText(img, "THR", (8, y0 + h - 6), FONT, FONT_SM, COLOR_WHITE, 1)
    tw = 120
    tx = 45
    cv2.rectangle(img, (tx, bar_y), (tx + tw, bar_y + bar_h), (50, 50, 60), -1)
    fill_w = int(tw * throttle)
    if fill_w > 0:
        cv2.rectangle(img, (tx, bar_y), (tx + fill_w, bar_y + bar_h), COLOR_THROTTLE, -1)
    cv2.rectangle(img, (tx, bar_y), (tx + tw, bar_y + bar_h), (120, 120, 130), 1)
    cv2.putText(img, f"{throttle:.2f}", (tx + 2, bar_y + bar_h - 3), FONT, FONT_SM, COLOR_WHITE, 1)

    # Brake bar
    bx = tx + tw + 60
    cv2.putText(img, "BRK", (bx - 28, y0 + h - 6), FONT, FONT_SM, COLOR_WHITE, 1)
    cv2.rectangle(img, (bx, bar_y), (bx + tw, bar_y + bar_h), (50, 50, 60), -1)
    fill_w = int(tw * brake)
    if fill_w > 0:
        cv2.rectangle(img, (bx, bar_y), (bx + fill_w, bar_y + bar_h), COLOR_BRAKE, -1)
    cv2.rectangle(img, (bx, bar_y), (bx + tw, bar_y + bar_h), (120, 120, 130), 1)
    cv2.putText(img, f"{brake:.2f}", (bx + 2, bar_y + bar_h - 3), FONT, FONT_SM, COLOR_WHITE, 1)

    # Steering indicator
    sx = bx + tw + 60
    cv2.putText(img, "STEER", (sx - 32, y0 + h - 6), FONT, FONT_SM, COLOR_WHITE, 1)
    sw2 = 160
    mid_x = sx + sw2 // 2
    cv2.rectangle(img, (sx, bar_y), (sx + sw2, bar_y + bar_h), (50, 50, 60), -1)
    steer_x = int(mid_x + steering * sw2 // 2)
    cv2.line(img, (mid_x, bar_y + 2), (mid_x, bar_y + bar_h - 2), (150, 150, 150), 1)
    cv2.line(img, (steer_x, bar_y + 2), (steer_x, bar_y + bar_h - 2), COLOR_STEER_BAR, 3)
    cv2.rectangle(img, (sx, bar_y), (sx + sw2, bar_y + bar_h), (120, 120, 130), 1)
    cv2.putText(img, f"{steering:.2f}", (sx + 2, bar_y + bar_h - 3), FONT, FONT_SM, COLOR_WHITE, 1)

    # Speed and frame on the right
    rx = sx + sw2 + 40
    cv2.putText(img, f"SPD: {int(speed_kmh)} | TGT: {int(target_speed)} km/h",
                (rx, y0 + h - 6), FONT, FONT_MD, COLOR_GREEN, 1)
    cv2.putText(img, f"FRAME: {frame}", (rx + 280, y0 + h - 6), FONT, FONT_SM, COLOR_TEXT, 1)


# ==============================================================================
# TRAJECTORY / STEERING PREVIEW BAR
# ==============================================================================
def draw_trajectory_bar(img, steering, curvature):
    y0 = TRAJ_Y
    h = TRAJ_H
    img[y0:y0 + h, 0:CANVAS_W] = COLOR_TELEM_BG
    cv2.line(img, (0, y0), (CANVAS_W, y0), (60, 60, 70), 1)

    cx = CANVAS_W // 2
    cy = y0 + h // 2

    # Lane width approximation
    lane_w = 60

    # Draw predicted path as a curve
    pts = []
    for i in range(30):
        t = i / 30.0
        y = int(cy - t * h * 0.8)
        x_offset = int(steering * 200 * t * t)
        x = cx + x_offset

        curve_offset = curvature * 40 * t * t
        x += int(curve_offset)

        pts.append((x, y))

    if len(pts) > 1:
        for i in range(1, len(pts)):
            alpha = 1.0 - i / len(pts)
            color = (int(50 + 150 * alpha), int(180 * alpha), int(255 * alpha))
            cv2.line(img, pts[i - 1], pts[i], color, 2)

    # Lane guide lines
    for i in range(0, CANVAS_W, 15):
        ly = cy - 5
        cv2.line(img, (i, ly), (i + 8, ly), (80, 80, 80), 1)

    cv2.line(img, (cx, cy + 15), (cx, cy - 20), COLOR_WHITE, 1)

    cv2.putText(img, "TRAJECTORY", (8, y0 + h - 6), FONT, FONT_SM, COLOR_WHITE, 1)
    cv2.putText(img, f"steer: {steering:.2f}  curv: {curvature:.4f}",
                (CANVAS_W - 280, y0 + h - 6), FONT, FONT_SM, COLOR_TEXT, 1)


# ==============================================================================
# MAIN RENDER FUNCTION
# ==============================================================================
def render_display(images_dict, lane_data, radar_boxes, radar_vehicles,
                   mode_label, confidence, level,
                   speed_kmh, target_speed, steering, throttle, brake,
                   frame):
    canvas = _make_blank(CANVAS_H, CANVAS_W)

    front_bgr = images_dict.get('F')
    draw_front_camera(canvas, front_bgr, lane_data, radar_boxes)
    draw_birds_eye(canvas, lane_data, radar_vehicles, speed_kmh)
    draw_speedometer(canvas, speed_kmh, target_speed)
    draw_small_cameras(canvas, images_dict)
    draw_telemetry(canvas, throttle, brake, steering, frame, speed_kmh, target_speed)
    draw_trajectory_bar(canvas, steering, lane_data['curvature'] if lane_data else 0.0)
    draw_mode_bar(canvas, mode_label, confidence, level)

    return canvas
