import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import math
import cv2
import numpy as np
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Camera

from config import (BEAMNG_HOME, BEAMNG_HOST, BEAMNG_PORT, VEHICLE_NAME, VEHICLE_MODEL,
                    MAP_NAME, SCENARIO_NAME, SPAWN_POS, SPAWN_YAW_DEG,
                    CAM_RESOLUTION, CAM_UPDATE_TIME, CAM_SPECS, DISPLAY)
from ai_driver import BeamNGAIDriver
from hybrid_controller import HybridController
from radar_sensor import RadarDetector

# ==============================================================================
# INICJALIZACJA SYSTEMU BEAMNG
# ==============================================================================
print("[SYSTEM] Nawiazywanie polaczenia...")
bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=BEAMNG_HOME)
bng.open()

vehicle = Vehicle(VEHICLE_NAME, model=VEHICLE_MODEL, license='AUTONOMY')
scenario = Scenario(MAP_NAME, SCENARIO_NAME)

yaw_rad = math.radians(SPAWN_YAW_DEG) + math.pi
q_z = math.sin(yaw_rad / 2.0)
q_w = math.cos(yaw_rad / 2.0)

scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=(0, 0, q_z, q_w))
scenario.make(bng)

print("[SYSTEM] Wczytywanie mapy i scenariusza...")
bng.scenario.load(scenario)
time.sleep(3)
bng.scenario.start()
time.sleep(2)

# ==============================================================================
# INICJALIZACJA MULTI-KAMER
# ==============================================================================
print("[SYSTEM] Inicjalizacja kamer wokol pojazdu...")

cam_args = {
    'resolution': CAM_RESOLUTION,
    'requested_update_time': CAM_UPDATE_TIME,
    'is_streaming': True,
    'is_using_shared_memory': True,
}

cameras = {}
for name, spec in CAM_SPECS.items():
    instance_name = f'cam_{name}'
    cameras[name] = Camera(instance_name, bng, vehicle,
                           pos=spec['pos'], dir=spec['dir'], **cam_args)

cameras_dict = {
    'FL': cameras['FL'], 'F': cameras['F'], 'FR': cameras['FR'],
    'BL': cameras['BL'], 'B': cameras['B'], 'BR': cameras['BR'],
}

# Tiny window just for keyboard input — BeamNG renders the real view
cv2.namedWindow(DISPLAY['window_name'], cv2.WINDOW_NORMAL)
cv2.resizeWindow(DISPLAY['window_name'], 320, 240)

# ==============================================================================
# INICJALIZACJA AI, KONTROLERA HYBRYDOWEGO I RADARU
# ==============================================================================
ai_driver = BeamNGAIDriver(bng, vehicle)
controller = HybridController(bng, vehicle, ai_driver)
radar = RadarDetector(bng, vehicle)
radar.start()

# ==============================================================================
# PETLA GLOWNA
# ==============================================================================
print("[SYSTEM] Start petli glownej...")
print("STEROWANIE:")
print("  1 - BeamNG Traffic AI (pelna autonomia)")
print("  2 - Custom CV (detekcja pasow + kontrola predkosci)")
print("  3 - LKA + ACC (asysta pasa + tempomat)")
print("  4 - DiffusionDriveV2 (end-to-end, wymaga GPU + checkpoint)")
print("  5 - DDV2 + Nawigacja (DDV2 z komendami skretu)")
print("  0 - STOP awaryjny")
print("  q - Wyjscie")
print("  UWAGA: Ogladaj okno BeamNG dla plynnosci — to okno to tylko input klawiszy.")

controller.init_ddv2()
controller.init_navigation()

frame = 0
speed_kmh = 0
radar_vehicles = []
radar_boxes = []
# Tiny placeholder image for the key-input window
tiny_view = np.zeros((240, 320, 3), dtype=np.uint8)

try:
    while True:
        bng.step(1)
        images = {}

        for name, cam in cameras_dict.items():
            try:
                stream_data = cam.stream()
                if stream_data is not None and 'colour' in stream_data:
                    img_pil = stream_data['colour']
                    images[name] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                else:
                    images[name] = np.zeros((CAM_RESOLUTION[1], CAM_RESOLUTION[0], 3), dtype=np.uint8)
            except Exception:
                images[name] = np.zeros((CAM_RESOLUTION[1], CAM_RESOLUTION[0], 3), dtype=np.uint8)

        try:
            vehicle.sensors.poll()
            vel = vehicle.state.get('vel', (0, 0, 0))
            speed_ms = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            speed_kmh = speed_ms * 3.6
        except Exception:
            pass

        result = controller.update(images, speed_kmh)
        target_steering, throttle, curvature, disp_speed, target_speed, mode_label, lane_data, confidence, brake = result

        if mode_label in ("TRAFFIC_AI", "LKA+ACC"):
            disp_speed = speed_kmh

        if frame % 3 == 0:
            radar_vehicles = radar.get_nearby_vehicles()
            radar_boxes = radar.project_to_front_camera(radar_vehicles)

        # Tiny front camera + speed overlay for key-input window (every frame)
        if 'F' in images:
            cv2.resize(images['F'], (320, 240), dst=tiny_view, interpolation=cv2.INTER_NEAREST)
        cv2.putText(tiny_view, f"{mode_label} | {int(disp_speed)} km/h", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow(DISPLAY['window_name'], tiny_view)

        frame += 1

        if frame % DISPLAY['frame_interval'] == 0:
            nav_cmd = getattr(controller, 'nav_command', '')
            nav_dist = getattr(controller, 'nav_distance', 0.0)
            nav_str = f" | NAV: {nav_cmd} {nav_dist:.0f}m" if nav_cmd else ""
            mode_info = f"[AUTONOMY] F:{frame} | {mode_label} | "
            mode_info += f"S:{target_steering:+.2f} | V:{int(disp_speed)}km/h | "
            mode_info += f"T:{throttle:.2f} B:{brake:.2f}"
            if 'DDV2' in mode_label:
                from ddv2.inference import DDV2Inference
                if controller.ddv2 and controller.ddv2.current_trajectory is not None:
                    traj = controller.ddv2.current_trajectory
                    wp0 = traj[0] if len(traj) > 0 else [0, 0, 0]
                    wp3 = traj[min(3, len(traj)-1)] if len(traj) > 3 else wp0
                    d0 = math.sqrt(wp0[0]**2 + wp0[1]**2)
                    d3 = math.sqrt(wp3[0]**2 + wp3[1]**2)
                    mode_info += f" | WP0:{d0:.1f}m WP3:{d3:.1f}m"
            mode_info += nav_str
            print(mode_info)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        controller.handle_key(key)

except KeyboardInterrupt:
    print("\n[SYSTEM] Przerwano recznie.")
except Exception as e:
    print(f"\n[SYSTEM] Blad krytyczny: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("[SYSTEM] Czyszczenie...")
    cv2.destroyAllWindows()
    bng.close()
    print("[SYSTEM] Koniec.")
