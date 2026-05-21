import math
from lane_detection import get_fused_lane_steering, get_lane_visualization_data
from speed_controller import compute_speed_control, reset_speed_controller


class HybridController:
    LEVEL_NAV_DDV2 = 4
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
        self.navigation = None
        self.nav_command = 'follow_lane'
        self.nav_distance = 999.0

    def init_navigation(self, waypoints=None, auto_distance=500.0):
        """Initialize the navigation system."""
        try:
            from navigation import NavigationSystem
            self.navigation = NavigationSystem(self.bng)
            self.navigation.auto_distance = auto_distance

            if waypoints:
                self.navigation.set_manual_waypoints(waypoints)
                print(f"[HYBRID] Nawigacja: {len(waypoints)} waypointow manualnych")
            else:
                self.navigation.load_graph()
                print(f"[HYBRID] Nawigacja: tryb auto-routingu (dystans: {auto_distance}m)")

            return True
        except ImportError as e:
            print(f"[HYBRID] Nawigacja niedostepna: {e}")
            self.navigation = None
            return False

    def init_ddv2(self, checkpoint_path=None):
        try:
            from ddv2 import DDV2Inference
            self.ddv2 = DDV2Inference(checkpoint_path=checkpoint_path)
            if self.ddv2.load():
                print("[HYBRID] DDV2 zaladowany - dostepny poziom 3 i 4")
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

        labels = {
            self.LEVEL_NAV_DDV2: "DDV2+NAV",
            self.LEVEL_DDV2: "DDV2",
            self.LEVEL_TRAFFIC_AI: "TRAFFIC_AI",
            self.LEVEL_LKA_ACC: "LKA+ACC",
            self.LEVEL_CUSTOM_CV: "CUSTOM_CV",
            self.LEVEL_EMERGENCY: "EMERGENCY",
        }
        print(f"[HYBRID] Przelaczanie: {labels.get(self.current_level, '?')} -> {labels.get(level, '?')}")

        if level == self.LEVEL_NAV_DDV2:
            if self.ddv2 is None:
                print("[HYBRID] DDV2 nie zaladowany - nie mozna przelaczyc")
                return
            if self.navigation is None:
                self.init_navigation()
            self.ai.stop()
            reset_speed_controller()
            self.ddv2.reset()
            self.ddv2_frame = 0
            self.current_level = level
        elif level == self.LEVEL_DDV2:
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
        lane_data = get_lane_visualization_data(images.get('F'))
        confidence = lane_data['confidence'] if lane_data else 0.0

        if self.current_level == self.LEVEL_EMERGENCY:
            self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
            return 0.0, 0.0, 0.0, 0.0, 0.0, "EMERGENCY", lane_data, confidence, 1.0

        health = self.ai.check_health()

        if self.current_level == self.LEVEL_TRAFFIC_AI and health == 'failed':
            print("[HYBRID] AI utknal - spadam do LKA+ACC")
            self.set_level(self.LEVEL_LKA_ACC)

        if self.current_level == self.LEVEL_LKA_ACC and health == 'failed':
            print("[HYBRID] LKA zawiodlo - spadam do CV")
            self.set_level(self.LEVEL_CUSTOM_CV)

        speed_ms = speed_kmh / 3.6

        if self.current_level == self.LEVEL_TRAFFIC_AI:
            return 0.0, 0.0, 0.0, speed_kmh, 0.0, "TRAFFIC_AI", lane_data, confidence, 0.0

        if self.current_level == self.LEVEL_LKA_ACC:
            steering, curvature = get_fused_lane_steering(images)
            return steering, 0.0, curvature, speed_kmh, 0.0, "LKA+ACC", lane_data, confidence, 0.0

        if self.current_level in (self.LEVEL_DDV2, self.LEVEL_NAV_DDV2) \
                and self.ddv2 and self.ddv2.is_available():

            driving_command = 'follow_lane'
            if self.current_level == self.LEVEL_NAV_DDV2 and self.navigation:
                pos = self.vehicle.state.get('pos')
                if pos:
                    driving_command, nav_dist = self.navigation.update(pos)
                    self.nav_command = driving_command
                    self.nav_distance = nav_dist

            if self.ddv2_frame % self.ddv2.frame_skip == 0:
                steering, throttle, brake = self.ddv2.step(
                    images, speed_ms, driving_command=driving_command
                )
            else:
                steering, throttle, brake = self.ddv2.interpolate(self.ddv2_frame, speed_ms)

            self.ddv2_frame += 1
            self.vehicle.control(throttle=throttle, steering=steering, brake=brake)

            mode_label = "DDV2+NAV" if self.current_level == self.LEVEL_NAV_DDV2 else "DDV2"
            return steering, throttle, 0.0, speed_ms * 3.6, 0.0, mode_label, lane_data, confidence, brake

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
            return steering, throttle, curvature, speed_kmh, target_speed, "CUSTOM_CV", lane_data, confidence, brake

        return 0.0, 0.0, 0.0, speed_kmh, 0.0, "UNKNOWN", lane_data, confidence, 0.0

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
        elif key == ord('5'):
            if self.ddv2 and self.ddv2.is_available():
                self.set_level(self.LEVEL_NAV_DDV2)
            else:
                print("[HYBRID] DDV2 niedostepny - najpierw wywolaj init_ddv2()")
        elif key == ord('0'):
            self.set_level(self.LEVEL_EMERGENCY)
