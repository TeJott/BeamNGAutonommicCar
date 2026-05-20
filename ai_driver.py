import math

class BeamNGAIDriver:
    def __init__(self, bng, vehicle):
        self.bng = bng
        self.vehicle = vehicle
        self.mode = 'manual'
        self.lka = None
        self.use_lka = False
        self.use_acc = False
        self.stuck_counter = 0
        self.stuck_threshold = 5 * 15

    def start_traffic_mode(self):
        self.vehicle.ai.set_mode('traffic')
        self.mode = 'traffic'
        print("[AI] Tryb Traffic aktywny - pelna autonomiczna jazda")

    def start_waypoint_mode(self, waypoints, speed_kmh=80.0):
        route_speed_ms = speed_kmh / 3.6
        self.vehicle.ai.drive_using_waypoints(
            waypoints,
            drive_in_lane=True,
            avoid_cars=True,
            route_speed=route_speed_ms
        )
        self.mode = 'waypoint'
        print(f"[AI] Tryb Waypoint: {len(waypoints)} punktow, predkosc {speed_kmh} km/h")

    def start_lka(self, risk_level=2, steering_strength=15):
        try:
            from beamngpy.vehicle.lka import LaneKeepingAssist
            self.lka = LaneKeepingAssist(
                self.bng, self.vehicle,
                risk_level=risk_level,
                steering_strength=steering_strength,
                detect_yellow=False
            )
            self.lka.start()
            self.use_lka = True
            self.mode = 'lka'
            print(f"[AI] LKA aktywny (risk={risk_level}, strength={steering_strength})")
        except ImportError:
            print("[AI] LKA niedostepne - brak modulu beamngpy.vehicle.lka")
            self.use_lka = False

    def stop_lka(self):
        if self.lka:
            try:
                self.lka.stop()
            except Exception:
                pass
            self.lka = None
        self.use_lka = False

    def start_acc(self, target_speed_ms=13.0):
        try:
            from beamngpy.sensors import IdealRadar
            self.radar = IdealRadar(
                "acc_radar", self.bng, self.vehicle,
                is_send_immediately=False,
                physics_update_time=0.01,
            )
            self.vehicle.acc.start(self.vehicle.vid, target_speed_ms, True)
            self.use_acc = True
            print(f"[AI] ACC aktywny: cel {target_speed_ms * 3.6:.0f} km/h")
        except Exception as e:
            print(f"[AI] ACC niedostepne: {e}")
            self.use_acc = False

    def stop(self):
        if self.use_lka:
            self.stop_lka()
        self.vehicle.ai.set_mode('disabled')
        self.mode = 'manual'
        self.stuck_counter = 0
        print("[AI] Wylaczony - sterowanie manualne")

    def check_health(self):
        try:
            self.vehicle.sensors.poll()
            vel = self.vehicle.state.get('vel', (0, 0, 0))
            speed = math.sqrt(vel[0]**2 + vel[1]**2)

            if self.mode in ('traffic', 'waypoint') and speed < 1.0:
                self.stuck_counter += 1
                if self.stuck_counter > self.stuck_threshold:
                    return 'failed'
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
            return 'ok'
        except Exception:
            return 'degraded'

    def get_status(self):
        return {
            'mode': self.mode,
            'lka_active': self.use_lka,
            'acc_active': self.use_acc,
            'health': self.check_health(),
        }
