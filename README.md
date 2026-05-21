# BeamNG Autonomous Car — hybrydowy system autonomicznej jazdy

System autonomicznego sterowania pojazdem w BeamNG.tech v0.38.5.0. Wykorzystuje 6 kamer, radar i wielopoziomowy kontroler hybrydowy do prowadzenia auta po mapie `west_coast_usa`.

## Wymagania

- **BeamNG.tech v0.38.5.0** (domyślnie `G:\Gry\BeamNG.tech.v0.38.5.0`)
- **Python 3.8+** z pakietami:
  ```
  pip install beamngpy opencv-python numpy
  ```
- (opcjonalnie) **PyTorch + CUDA + DiffusionDriveV2** — tylko dla trybu 4

## Uruchamianie

1. **Włącz BeamNG.tech** — musi działać przed odpaleniem skryptu
2. **Odpal skrypt:**
   ```powershell
   cd "c:\Users\jedru\Documents\ClaudeCodeProjects\BeamNGAutonommicCar"
   python main.py
   ```
3. Pojawi się okno `360 Camera View` z widokiem z kamer i pełną wizualizacją

## Sterowanie

| Klawisz | Tryb | Opis |
|---------|------|------|
| **1** | Traffic AI | Pełna autonomia — AI BeamNG prowadzi po drogach, omija przeszkody, przestrzega pasów |
| **2** | Custom CV | Własny algorytm — detekcja pasów (Canny + Hough) + PI speed controller |
| **3** | LKA + ACC | Asystent pasa + adaptacyjny tempomat (wbudowane systemy BeamNG) |
| **4** | DDV2 | DiffusionDriveV2 — end-to-end model AI (wymaga GPU + wag z HuggingFace) |
| **0** | STOP | Zatrzymanie awaryjne — hamulec 100% |
| **q** | Wyjście | Zamyka program i BeamNG |

## Tryby szczegółowo

### Tryb 1 — Traffic AI
Wbudowane AI BeamNG (`vehicle.ai.set_mode('traffic')`).

**Jak steruje:** Silnik BeamNG (Lua) — skrypt nie wysyła `vehicle.control()`

**Co potrafi:**
- Zna siatkę dróg, skrzyżowania, limity prędkości
- Omija inne pojazdy, przestrzega świateł i znaków
- Dynamicznie dostosowuje prędkość do warunków

**Wady:** Czarna skrzynka — nie wiadomo co AI "widzi" i jakie decyzje podejmuje

---

### Tryb 2 — Custom CV
Własny pipeline komputerowej wizji (`lane_detection.py` + `speed_controller.py`).

**Jak steruje:** Skrypt przez `vehicle.control(throttle, steering, brake)`

**Pipeline:**
1. CLAHE → adaptacyjne Canny → HoughLinesP → detekcja lewego/prawego pasa
2. Fuzja kamery przedniej z FL/FR gdy brakuje jednej linii
3. Wygładzanie EMA środka pasa (`0.7 × prev + 0.3 × raw`)
4. P-controller: `steering = błąd × 0.007 × confidence`
5. PI speed controller: krzywizna → prędkość docelowa 25-100 km/h

**Co potrafi:**
- Utrzymuje pas na prostych i łagodnych zakrętach
- Adaptacyjne progi Canny — działa w tunelach i przy słabym oświetleniu
- Zwalnia na ostrych zakrętach, przyspiesza na autostradzie

**Wady:** Nie zna mapy, nie widzi innych aut, nie rozpoznaje znaków, zgubi się bez linii

---

### Tryb 3 — LKA + ACC
Wbudowane systemy BeamNG: `LaneKeepingAssist` + adaptacyjny tempomat.

**Jak steruje:** BeamNG — LKA daje korekty kierownicy, ACC kontroluje gaz/hamulec

**Pipeline LKA:**
1. Binary threshold → bird's eye perspective warp
2. Sliding window → polynomial lane fitting (2-go stopnia)
3. Pomiar krzywizny w metrach (nie pikselach)
4. Korekty kierownicy przy wyjeżdżaniu z pasa

**Pipeline ACC:**
1. IdealRadar mierzy odległość i prędkość względną auta z przodu
2. Utrzymuje bezpieczny dystans — samo zwalnia i przyspiesza

**Co potrafi lepiej niż Tryb 2:**
- Dokładniejsza detekcja pasa (perspective transform)
- Utrzymuje dystans od auta z przodu (radar)

**Wady:** LKA tylko koryguje kierownicę — nie prowadzi auta przez ostre zakręty. To asystent, nie autopilot.

---

### Tryb 4 — DiffusionDriveV2 (opcjonalny)
End-to-end model AI (dyfuzja + reinforcement learning) z MIT license.

**Wymagania dodatkowe:**
```bash
git clone https://github.com/hustvl/DiffusionDriveV2.git
pip install torch torchvision
huggingface-cli download hustvl/DiffusionDriveV2
```

**Jak działa:** 6 kamer → ResNet-34 → truncated diffusion → K trajektorii → pure pursuit controller

**Wyzwania:** Domain gap NAVSIM→BeamNG, opóźnienie inferencji ~100-200ms, wymaga GPU

## Porównanie trybów

| Cecha | Tryb 1 (Traffic AI) | Tryb 2 (Custom CV) | Tryb 3 (LKA+ACC) | Tryb 4 (DDV2) |
|-------|---------------------|---------------------|-------------------|---------------|
| Detekcja pasa | Wbudowana | Canny + Hough (piksele) | Binary + sliding window (metry) | ResNet-34 |
| Sterowanie | Pełne (AI) | Pełne (P-controller) | Korekcyjne (nudge) | Pełne (trajektoria) |
| Świadomość mapy | Tak | Nie | Nie | Nie |
| Omijanie przeszkód | Tak | Nie | Tak (ACC, dystans) | Trajektorie |
| Działa w tunelach | Tak | Tak (CLAHE) | Nie wiadomo | Zależy od danych |
| Prędkość | Dynamiczna | PI z krzywizny | ACC z radaru | Z trajektorii |
| Wymagania | - | - | - | GPU + checkpoint |
| Kod | Silnik BeamNG | Nasz | Wbudowany BeamNG | NASZ + DDV2 repo |

## Struktura projektu

```
BeamNGAutonommicCar/
├── main.py                    # Punkt wejścia, połączenie BeamNG, pętla główna
├── config.py                  # Wszystkie stałe (BEAMNG_HOME, PID, kamery...)
├── lane_detection.py          # Detekcja pasów: CLAHE, Canny, Hough, fuzja kamer
├── speed_controller.py        # Kontroler PI prędkości oparty na krzywiźnie
├── visualization.py           # Renderowanie UI: pasy, speedometr, minimapa, HUD
├── radar_sensor.py            # IdealRadar — wykrywanie innych pojazdów
├── ai_driver.py               # Wrapper AI BeamNG: traffic, waypoint, LKA, ACC
├── hybrid_controller.py       # Wielopoziomowy kontroler z auto-przełączaniem
└── ddv2/
    ├── __init__.py             # Pakiet integracji DiffusionDriveV2
    ├── camera_adapter.py       # Adapter 6 kamer BeamNG → format DDV2/NAVSIM
    ├── model_loader.py         # Ładowanie wag DDV2 (PyTorch)
    ├── trajectory_to_control.py # Pure pursuit: trajektoria → steering/throttle/brake
    └── inference.py            # Pipeline inferencji DDV2 z frame-skip
```

## Elementy wizualizacji

```
[============ MODE BAR (kolorowy pasek trybu) ============]
[ FRONT CAMERA 660x440  ][ BIRD'S EYE  ][ SPEEDOMETER  ]
[ - zielone linie lewe   ][   minimapa   ][  półokrągły   ]
[ - pomarańczowe prawe   ][   z góry     ][  gauge 0-140  ]
[ - bounding boxy radar  ][              ][               ]
[FL][FR][ B ][BL][BR]  (5x małych kamer)
[== TELEMETRIA: THR | BRK | STEER | SPD | FRAME ========]
[== TRAJEKTORIA: przewidywana ścieżka ==================]
```

## Naprawione problemy (względem oryginalnego kodu)

| Problem | Rozwiązanie |
|---------|-------------|
| Ciemność / tunele | CLAHE + adaptacyjne progi Canny zależne od jasności sceny |
| Tylko przednia kamera | Fuzja kamer FL/FR + wygładzanie EMA + P-controller z wagą pewności |
| Stała prędkość (throttle=0.15) | PI speed controller zależny od krzywizny: 25 km/h na zakrętach, do 100 km/h na prostej |
| Brak detekcji przeszkód | IdealRadar + bounding boxy na obrazie z kamery |
| Brak trybu autonomicznego | 4 poziomy: Traffic AI → LKA+ACC → Custom CV → DDV2 |

## Konfiguracja

Wszystkie parametry w [`config.py`](config.py):

- `BEAMNG_HOME` — ścieżka do BeamNG.tech
- `SPAWN_POS` / `SPAWN_YAW_DEG` — pozycja startowa auta
- `LANE_DETECTION` — progi Canny, CLAHE, Hough, EMA, PID
- `SPEED_CONTROL` — prędkości min/max, nastawy PI
- `CAM_SPECS` — pozycje i kierunki 6 kamer
- `DISPLAY` — rozmiar okna, interwał logowania

---

Wszelkie prawa zastrzeżone. Dzięki za oglądanie!
