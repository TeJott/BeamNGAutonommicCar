import cv2
import numpy as np
from config import LANE_DETECTION as CFG


def _detect_lane_points(img_bgr):
    h, w, _ = img_bgr.shape
    roi_top = int(h * CFG['roi_top_ratio'])
    roi = img_bgr[roi_top:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CFG['clahe_clip'], tileGridSize=CFG['clahe_grid'])
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lower_portion = blur[int(blur.shape[0] * 0.5):, :]
    mean_intensity = np.mean(lower_portion)
    low_thresh = max(CFG['canny_low_min'], int(mean_intensity * CFG['canny_low_factor']))
    high_thresh = max(CFG['canny_high_min'], int(mean_intensity * CFG['canny_high_factor']))
    edges = cv2.Canny(blur, low_thresh, high_thresh)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=CFG['hough_threshold'],
                            minLineLength=CFG['hough_min_line'], maxLineGap=CFG['hough_max_gap'])

    left_x = []
    right_x = []
    mid_y = int((h - roi_top) / 2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < CFG['slope_min_abs']:
                continue
            x_at_mid = int(x1 + (mid_y - y1) / slope)
            if slope < -CFG['slope_threshold'] and x_at_mid < w * 0.5:
                left_x.append(x_at_mid)
            elif slope > CFG['slope_threshold'] and x_at_mid > w * 0.5:
                right_x.append(x_at_mid)

    return left_x, right_x


def estimate_curvature(left_x, right_x, img_width):
    if not left_x or not right_x:
        return 0.0
    if len(left_x) < 2 or len(right_x) < 2:
        return 0.0
    lane_width_px = abs(np.mean(right_x) - np.mean(left_x))
    if lane_width_px < 10:
        return 0.0
    xm_per_pix = 3.7 / lane_width_px
    curv_est = abs(np.mean(right_x) - img_width / 2) - abs(np.mean(left_x) - img_width / 2)
    curv_est *= xm_per_pix * 0.1
    return max(0.0, abs(curv_est))


def get_lane_steering(img_bgr):
    if img_bgr is None:
        return 0.0, 0.0, 0.0

    h, w, _ = img_bgr.shape
    roi_top = int(h * CFG['roi_top_ratio'])
    left_x, right_x = _detect_lane_points(img_bgr)
    img_center = w / 2.0

    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2.0
        confidence = 1.0
    elif left_x:
        lane_center = np.mean(left_x) + CFG['assumed_lane_width_px']
        confidence = 0.55
    elif right_x:
        lane_center = np.mean(right_x) - CFG['assumed_lane_width_px']
        confidence = 0.55
    else:
        lane_center = img_center
        confidence = 0.0

    curvature = estimate_curvature(left_x, right_x, w)

    error = lane_center - img_center
    Kp = CFG['steering_kp'] * max(CFG['confidence_min_kp'], confidence)
    steering = error * Kp
    steering = max(-1.0, min(1.0, steering))

    cv2.line(img_bgr, (int(img_center), h-10), (int(img_center), roi_top), (0, 0, 255), 2)
    cv2.line(img_bgr, (int(lane_center), h-10), (int(lane_center), roi_top), (255, 0, 0), 2)

    return steering, confidence, curvature


def get_fused_lane_steering(images_dict):
    front_img = images_dict.get('F')
    if front_img is None:
        return 0.0, 0.0

    h, w, _ = front_img.shape
    left_x, right_x = _detect_lane_points(front_img)
    roi_top = int(h * CFG['roi_top_ratio'])

    if left_x and right_x:
        raw_center = (np.mean(left_x) + np.mean(right_x)) / 2.0
        confidence = 1.0
    elif left_x:
        raw_center = np.mean(left_x) + CFG['assumed_lane_width_px']
        confidence = 0.55
    elif right_x:
        raw_center = np.mean(right_x) - CFG['assumed_lane_width_px']
        confidence = 0.55
    else:
        raw_center = w / 2.0
        confidence = 0.0

    curvature = estimate_curvature(left_x, right_x, w)

    # EMA smoothing
    if not hasattr(get_fused_lane_steering, '_prev_center'):
        get_fused_lane_steering._prev_center = raw_center
    prev = get_fused_lane_steering._prev_center
    smoothed = 0.7 * prev + CFG['ema_alpha'] * raw_center
    get_fused_lane_steering._prev_center = smoothed

    error = smoothed - w / 2.0
    Kp = CFG['steering_kp'] * max(CFG['confidence_min_kp'], confidence)
    steering = max(-1.0, min(1.0, error * Kp))

    # Visualization overlay on front image
    img_center = w / 2.0
    cv2.line(front_img, (int(img_center), h - 10), (int(img_center), roi_top), (0, 0, 255), 2)
    cv2.line(front_img, (int(raw_center), h - 10), (int(raw_center), roi_top), (255, 0, 0), 2)

    return steering, curvature


def get_lane_visualization_data(img_bgr):
    """Returns rich lane data for visualization overlays."""
    if img_bgr is None:
        return None

    h, w, _ = img_bgr.shape
    roi_top = int(h * CFG['roi_top_ratio'])
    left_x, right_x = _detect_lane_points(img_bgr)

    lane_center = w / 2.0
    confidence = 0.0
    left_mean = None
    right_mean = None

    if left_x and right_x:
        left_mean = np.mean(left_x)
        right_mean = np.mean(right_x)
        lane_center = (left_mean + right_mean) / 2.0
        confidence = 1.0
    elif left_x:
        left_mean = np.mean(left_x)
        lane_center = left_mean + CFG['assumed_lane_width_px']
        right_mean = lane_center + CFG['assumed_lane_width_px'] / 2
        confidence = 0.55
    elif right_x:
        right_mean = np.mean(right_x)
        lane_center = right_mean - CFG['assumed_lane_width_px']
        left_mean = lane_center - CFG['assumed_lane_width_px'] / 2
        confidence = 0.55

    return {
        'roi_top': roi_top,
        'img_width': w,
        'img_height': h,
        'left_x': left_x,
        'right_x': right_x,
        'left_mean': left_mean,
        'right_mean': right_mean,
        'lane_center': lane_center,
        'confidence': confidence,
        'curvature': estimate_curvature(left_x, right_x, w),
    }
