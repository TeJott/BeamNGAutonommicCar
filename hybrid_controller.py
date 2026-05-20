import math
from lane_detection import get_fused_lane_steering
from speed_controller import compute_speed_control, reset_speed_controller


class HybridController:
    LEVEL_DDV2 = 3
    LEVEL_TRAFFIC_AI = 2
    LEVEL_LKA_ACC = 1
    LEVEL_CUSTOM_CV = 0
    LEVEL_EMERGENCY = -1

    def __init__(self, bng, vehicle, ai_driver):
        self.bng = bng
        self.vehicle = vehicle
        self.ai = ai_driver
        self.current_level = self.LEVEL_CUSTOM_CV
        self.transition_progress = 0.0
        self.transition_steps = 10
        self.prev_throttle = 0.0
        self.prev_steering = 0.0
        self.prev_brake = 0.0
        self.ddv2 = None
        self.ddv2_frame = 0

    def init_ddv2(self, checkpoint_path=None):
        try:
            from ddv2 import DDV2Inference
            self.ddv2 = DDV2Inference(checkpoint_path=checkpoint_path)
            if self.ddv2.load():
                print("[HYBRID] DDV2 zaladowany - dostepny poziom 3")
                return True
            else:
                print("[HYBRID] DDV2 nie zaladowany - sprawdz checkpoint i zaleznosci")
                self.ddv2 = None
                return False
        except ImportError as e:
            print(f"[HYBRID] DDV2 niedostepne: {e}")
            self.ddv2 = None
            return False

    def set_level(self, level):
        if level == self.current_level:
            return

        print(f"[HYBRID] Przelaczanie: poziom {self.current_level} -> {level}")

        if level == self.LEVEL_DDV2:
            if self.ddv2 is None:
                print("[HYBRID] DDV2 nie zaladowany - nie mozna przelaczyc")
                return
            self.ai.stop()
            reset_speed_controller()
            self.ddv2.reset()
            self.ddv2_frame = 0
            self.current_level = level
        elif level == self.LEVEL_TRAFFIC_AI:
            self.ai.start_traffic_mode()
            self.current_level = level
        elif level == self.LEVEL_LKA_ACC:
            self.ai.stop()
            self.ai.start_lka()
            self.ai.start_acc()
            reset_speed_controller()
            self.current_level = level
        elif level == self.LEVEL_CUSTOM_CV:
            self.ai.stop()
            reset_speed_controller()
            self.current_level = level
        elif level == self.LEVEL_EMERGENCY:
            self.ai.stop()
            self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
            self.current_level = level

        self.transition_progress = 0

    def update(self, images, speed_kmh):
        if self.current_level == self.LEVEL_EMERGENCY:
            self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
            return 0.0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY"

        health = self.ai.check_health()

        if self.current_level == self.LEVEL_TRAFFIC_AI and health == 'failed':
            print("[HYBRID] AI utknal - spadam do LKA+ACC")
            self.set_level(self.LEVEL_LKA_ACC)

        if self.current_level == self.LEVEL_LKA_ACC and health == 'failed':
            print("[HYBRID] LKA zawiodlo - spadam do CV")
            self.set_level(self.LEVEL_CUSTOM_CV)

        if self.current_level in (self.LEVEL_TRAFFIC_AI,):
            try:
                self.vehicle.sensors.poll()
                vel = self.vehicle.state.get('vel', (0, 0, 0))
                speed_ms = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                current_speed = speed_ms * 3.6
            except Exception:
                current_speed = speed_kmh

            return 0.0, 0.0, 0.0, current_speed, 0.0, "TRAFFIC_AI"

        if self.current_level == self.LEVEL_LKA_ACC:
            try:
                self.vehicle.sensors.poll()
                vel = self.vehicle.state.get('vel', (0, 0, 0))
                speed_ms = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                current_speed = speed_ms * 3.6
            except Exception:
                current_speed = speed_kmh

            steering, curvature = get_fused_lane_steering(images)
            return steering, 0.0, curvature, current_speed, 0.0, "LKA+ACC"

        if self.current_level == self.LEVEL_DDV2 and self.ddv2 and self.ddv2.is_available():
            try:
                self.vehicle.sensors.poll()
                vel = self.vehicle.state.get('vel', (0, 0, 0))
                speed_ms = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            except Exception:
                speed_ms = speed_kmh / 3.6

            if self.ddv2_frame % self.ddv2.frame_skip == 0:
                steering, throttle, brake = self.ddv2.step(images, speed_ms)
            else:
                steering, throttle, brake = self.ddv2.interpolate(self.ddv2_frame)

            self.ddv2_frame += 1
            self.vehicle.control(throttle=throttle, steering=steering, brake=brake)
            return steering, throttle, 0.0, speed_ms * 3.6, 0.0, "DDV2"

        if self.current_level == self.LEVEL_CUSTOM_CV:
            steering, curvature = get_fused_lane_steering(images)
            throttle, brake, target_speed = compute_speed_control(curvature, speed_kmh)

            if self.transition_progress < self.transition_steps:
                alpha = self.transition_progress / self.transition_steps
                throttle = self.prev_throttle * (1 - alpha) + throttle * alpha
                brake = self.prev_brake * (1 - alpha) + brake * alpha
                steering = self.prev_steering * (1 - alpha) + steering * alpha
                self.transition_progress += 1

            self.prev_throttle = throttle
            self.prev_steering = steering
            self.prev_brake = brake

            self.vehicle.control(throttle=throttle, steering=steering, brake=brake)
            return steering, throttle, curvature, speed_kmh, target_speed, "CUSTOM_CV"

        return 0.0, 0.0, 0.0, speed_kmh, 0.0, "UNKNOWN"

    def handle_key(self, key):
        if key == ord('1'):
            self.set_level(self.LEVEL_TRAFFIC_AI)
        elif key == ord('2'):
            self.set_level(self.LEVEL_CUSTOM_CV)
        elif key == ord('3'):
            self.set_level(self.LEVEL_LKA_ACC)
        elif key == ord('4'):
            if self.ddv2 and self.ddv2.is_available():
                self.set_level(self.LEVEL_DDV2)
            else:
                print("[HYBRID] DDV2 niedostepny - najpierw wywolaj init_ddv2()")
        elif key == ord('0'):
            self.set_level(self.LEVEL_EMERGENCY)
