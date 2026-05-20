from config import SPEED_CONTROL as CFG

_speed_integral = 0.0


def compute_speed_control(curvature, speed_kmh):
    global _speed_integral

    if curvature > 0.001:
        target_speed = max(CFG['min_speed_kmh'],
                           min(CFG['max_speed_kmh'],
                               CFG['max_speed_kmh'] / (1.0 + curvature * CFG['curvature_factor'])))
    else:
        target_speed = CFG['default_speed_kmh']

    error = target_speed - speed_kmh

    Kp = CFG['kp']
    Ki = CFG['ki']
    _speed_integral += error
    _speed_integral = max(-CFG['integral_max'], min(CFG['integral_max'], _speed_integral))

    throttle = max(0.0, min(1.0, Kp * error + Ki * _speed_integral))

    brake = 0.0
    if error < -CFG['brake_threshold_kmh']:
        brake = min(CFG['brake_max'], abs(error) * CFG['brake_kp'])

    return throttle, brake, target_speed


def reset_speed_controller():
    global _speed_integral
    _speed_integral = 0.0
