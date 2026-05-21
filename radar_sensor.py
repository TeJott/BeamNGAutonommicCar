import math


class RadarDetector:
    def __init__(self, bng, vehicle):
        self.bng = bng
        self.vehicle = vehicle
        self.radar = None
        self.active = False

    def start(self):
        try:
            from beamngpy.sensors import IdealRadar
            self.radar = IdealRadar(
                "visual_radar", self.bng, self.vehicle,
                is_send_immediately=False,
                physics_update_time=0.01,
            )
            self.active = True
            print("[RADAR] IdealRadar aktywny")
            return True
        except Exception as e:
            print(f"[RADAR] Nieudana inicjalizacja: {e}")
            self.active = False
            return False

    def get_nearby_vehicles(self):
        if not self.active or self.radar is None:
            return []

        try:
            data = self.radar.poll()
            vehicles = []

            for key in ['closestVehicles1', 'closestVehicles']:
                if key in data:
                    for i, v in enumerate(data[key]):
                        vehicles.append({
                            'rel_dist': v.get('relDist', 0.0),
                            'rel_vel': v.get('relVel', 0.0),
                            'rel_pos_y': v.get('relPosY', i * 1.5 - 2.0),
                        })
                    break

            if not vehicles and isinstance(data, list):
                for i, v in enumerate(data):
                    if isinstance(v, dict):
                        vehicles.append({
                            'rel_dist': v.get('relDist', v.get('dist', 10.0)),
                            'rel_vel': v.get('relVel', 0.0),
                            'rel_pos_y': v.get('relPosY', v.get('lat', i * 1.5 - 2.0)),
                        })

            return vehicles
        except Exception:
            return []

    def project_to_front_camera(self, vehicles, img_width=300, img_height=200,
                                fov_h_deg=90, max_range=100.0):
        boxes = []
        if not vehicles:
            return boxes

        fov_h_rad = math.radians(fov_h_deg)
        px_per_rad = img_width / fov_h_rad
        px_per_meter_h = img_height * 0.5 / max_range

        for v in vehicles:
            dist = abs(v.get('rel_dist', 10.0))
            if dist < 1.0 or dist > max_range:
                continue

            rel_pos_y = v.get('rel_pos_y', 0.0)
            angle = math.atan2(rel_pos_y, max(dist, 0.1))

            cx = int(img_width / 2 + angle * px_per_rad)
            cy = int(img_height * 0.6 - dist * px_per_meter_h * 2.5)

            w = max(12, int(70.0 / max(dist * 0.8, 1.0)))
            h = max(10, int(45.0 / max(dist * 0.8, 1.0)))

            cx = max(0, min(img_width - w, cx - w // 2))
            cy = max(0, min(img_height - h, cy))

            boxes.append((cx, cy, w, h, dist))

        return boxes
