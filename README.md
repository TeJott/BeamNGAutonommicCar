# BeamNG Autonomous Car — hybrydowy system autonomicznej jazdy

System autonomicznego sterowania pojazdem w BeamNG.tech v0.38.5.0. Wykorzystuje 6 kamer, radar, wielopoziomowy kontroler hybrydowy oraz model DiffusionDriveV2 do prowadzenia auta po mapie `west_coast_usa`.

## Wymagania

- **BeamNG.tech v0.38.5.0** (ścieżka do ustawienia w `config.py`)
- **Python 3.8+** z pakietami:
  ```
  pip install beamngpy opencv-python numpy
  ```
- **GPU NVIDIA + CUDA** (zalecane) — dla trybów DDV2 (4 i 5)
- **PyTorch + zależności DDV2** (do inferencji AI):
  ```
  pip install torch>=2.0.1 torchvision diffusers einops timm
  pip install cloudpickle joblib omegaconf hydra-core
  ```

## Szybki start

1. **Włącz BeamNG.tech** — musi działać przed odpaleniem skryptu
2. **Odpal skrypt:**
   ```powershell
   cd <katalog_projektu>
   python main.py
   ```
3. Pojawi się okno `360 Camera View` z widokiem z kamer i pełną wizualizacją

## Sterowanie

| Klawisz | Tryb | Opis |
|---------|------|------|
| **1** | Traffic AI | Pełna autonomia — AI BeamNG prowadzi po drogach, omija przeszkody, przestrzega pasów |
| **2** | Custom CV | Własny algorytm — detekcja pasów (Canny + Hough) + PI speed controller |
| **3** | LKA + ACC | Asystent pasa + adaptacyjny tempomat (wbudowane systemy BeamNG) |
| **4** | DDV2 | DiffusionDriveV2 — end-to-end model AI, 6 kamer, inferencja co 3 klatki |
| **5** | DDV2+NAV | DDV2 + nawigacja — model otrzymuje komendy skrętu (lewo/prawo/prosto) z systemu nawigacji |
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
Własny pipeline komputerowej wizji ([lane_detection.py](lane_detection.py) + [speed_controller.py](speed_controller.py)).

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

### Tryb 4 — DiffusionDriveV2 (DDV2)
End-to-end model AI: 6 kamer → ResNet-34 backbone → truncated diffusion → 4 trajektorie → pure pursuit controller.

**Model:** [hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2) (MIT license), wariant SEL (93.2M parametrów)

**Architektura:**
- **Wejście:** panorama 1024×256 z 6 kamer (FL→F→FR→BR→B→BL, każda ~170px) + LiDAR BEV 256×256 (syntetyczny z radaru) + wektor ego status (8D: prędkość, przyspieszenie, komenda jazdy)
- **Backbone:** ResNet-34 (enkoder obrazu + LiDAR)
- **Głowica trajektorii:** Truncated diffusion z GMM anchorami (20 grup × 8 timestepów) + coarse-to-fine scorer
- **Wyjście:** 4 trajektorie (8, 3) w lokalnych współrzędnych ego — x, y, yaw

**Jak działa w projekcie:**
1. [`feature_builder.py`](ddv2/feature_builder.py) — buduje panoramę 6-kamer, syntetyczny BEV z IdealRadar, wektor ego status z komendą jazdy
2. [`model_loader.py`](ddv2/model_loader.py) — ładuje `V2TransfuserModel` z checkpointu (364MB), monkey-patch PDM scoringu dla inferencji
3. [`inference.py`](ddv2/inference.py) — pipeline co 3 klatki (frame-skip), pure pursuit → steering/throttle/brake

**Wyzwania:** Domain gap NAVSIM→BeamNG, model trenowany na 3 kamerach NAVSIM (my dajemy 6), inferencja ~200ms na GPU

---

### Tryb 5 — DDV2 + Nawigacja (DDV2+NAV)
DDV2 z systemem nawigacji dostarczającym komendy skrętu.

**System nawigacji** ([navigation.py](navigation.py)):
- Pobiera graf dróg z BeamNG (`NavigraphData`) — A* pathfinding
- Auto-routing: znajduje trasę ~500m do przodu po siatce dróg
- Detekcja manewrów: analiza zmiany bearingu na krawędziach grafu (>30° = skręt)
- Wykrywanie skrzyżowań (węzły o stopniu >2)
- 4 komendy: `follow_lane`, `turn_left`, `turn_right`, `keep_straight`
- Tryb manualny: lista waypointów zdefiniowanych przez użytkownika

**Jak DDV2 używa nawigacji:**
1. NavigationSystem zwraca komendę + dystans do manewru
2. Komenda kodowana jako 4-dim one-hot w wektorze ego status (pozycje 4-7)
3. DDV2 warunkuje generację trajektorii na podstawie komendy — wie, gdzie skręcić

**Wizualizacja:** Panel nawigacji w pasku trybu: `← TURN LEFT 150m` / `^^ STRAIGHT ^^` / `TURN RIGHT -->`

---

## Porównanie trybów

| Cecha | Tryb 1 (Traffic AI) | Tryb 2 (Custom CV) | Tryb 3 (LKA+ACC) | Tryb 4 (DDV2) | Tryb 5 (DDV2+NAV) |
|-------|---------------------|---------------------|-------------------|---------------|-------------------|
| Detekcja pasa | Wbudowana | Canny + Hough (piksele) | Binary + sliding window (metry) | ResNet-34 | ResNet-34 |
| Sterowanie | Pełne (AI) | Pełne (P-controller) | Korekcyjne (nudge) | Pełne (trajektoria) | Pełne (trajektoria) |
| Świadomość mapy | Tak | Nie | Nie | Nie | **Tak — A*** |
| Omijanie przeszkód | Tak | Nie | Tak (ACC, dystans) | Trajektorie | Trajektorie |
| Nawigacja na skrzyżowaniach | Tak (AI) | Nie | Nie | Nie | **Tak** |
| Działa w tunelach | Tak | Tak (CLAHE) | Nie wiadomo | Zależy od danych | Zależy od danych |
| Prędkość | Dynamiczna | PI z krzywizny | ACC z radaru | Z trajektorii | Z trajektorii |
| Wymagania | - | - | - | GPU + checkpoint | GPU + checkpoint |
| Kod | Silnik BeamNG | Nasz | Wbudowany BeamNG | Nasz + DDV2 repo | Nasz + DDV2 repo + nawigacja |

## Struktura projektu

```
BeamNGAutonommicCar/
├── main.py                    # Punkt wejścia, połączenie BeamNG, pętla główna
├── config.py                  # Wszystkie stałe (BEAMNG_HOME, PID, kamery, NAV, DDV2...)
├── lane_detection.py          # Detekcja pasów: CLAHE, Canny, Hough, fuzja kamer
├── speed_controller.py        # Kontroler PI prędkości oparty na krzywiźnie
├── visualization.py           # Renderowanie UI: pasy, speedometr, minimapa, HUD, nawigacja
├── radar_sensor.py            # IdealRadar — wykrywanie innych pojazdów
├── ai_driver.py               # Wrapper AI BeamNG: traffic, waypoint, LKA, ACC
├── hybrid_controller.py       # Wielopoziomowy kontroler (5 trybów) z auto-przełączaniem
├── navigation.py              # System nawigacji: A*, detekcja skrętów, waypointy
├── ddv2/
│   ├── __init__.py             # Pakiet integracji DiffusionDriveV2
│   ├── ddv2_imports.py         # Stuby NAVSIM/nuPlan — izolacja modelu DDV2
│   ├── camera_adapter.py       # Adapter 6 kamer → panorama 1024×256
│   ├── feature_builder.py      # Builder feature: panorama, BEV, ego status
│   ├── model_loader.py         # Ładowanie V2TransfuserModel (93.2M) z checkpointu
│   ├── trajectory_to_control.py # Pure pursuit: trajektoria → steering/throttle/brake
│   ├── inference.py            # Pipeline inferencji DDV2 z frame-skip + nav commands
│   ├── checkpoints/
│   │   └── diffusiondrivev2_sel.ckpt   # Wagi modelu SEL (364MB)
│   └── DiffusionDriveV2/       # Sklonowane repozytorium DDV2 (śledzone przez git)
└── navsim/                     # (z repozytorium DDV2)
```

## Elementy wizualizacji

```
[============ MODE BAR (kolorowy pasek trybu) ============]
[ ← TURN LEFT 150m  ]  (panel nawigacji — tylko tryb 5)
[ FRONT CAMERA 660x440  ][ BIRD'S EYE  ][ SPEEDOMETER  ]
[ - zielone linie lewe   ][   minimapa   ][  półokrągły   ]
[ - pomarańczowe prawe   ][   z góry     ][  gauge 0-140  ]
[ - bounding boxy radar  ][              ][               ]
[FL][FR][ B ][BL][BR]  (5x małych kamer)
[== TELEMETRIA: THR | BRK | STEER | SPD | FRAME ========]
[== TRAJEKTORIA: przewidywana ścieżka ==================]
```

## Konfiguracja

Wszystkie parametry w [`config.py`](config.py):

- `BEAMNG_HOME` — ścieżka do BeamNG.tech
- `SPAWN_POS` / `SPAWN_YAW_DEG` — pozycja startowa auta
- `LANE_DETECTION` — progi Canny, CLAHE, Hough, EMA, PID
- `SPEED_CONTROL` — prędkości min/max, nastawy PI
- `CAM_SPECS` — pozycje i kierunki 6 kamer
- `DISPLAY` — rozmiar okna, interwał logowania
- `NAVIGATION` — dystans auto-routingu, próg kąta skrętu, lookahead
- `DDV2` — frame-skip, ścieżka do checkpointu

## DDV2 — szczegóły techniczne

### Izolacja od NAVSIM/nuPlan

Model DDV2 jest trenowany w ekosystemie NAVSIM i importuje 30+ modułów z nuPlan/NAVSIM. Zamiast instalować cały nuplan-devkit (ogromny, trudny na Windows), projekt używa **stubów importowych**:

- [`ddv2_imports.py`](ddv2/ddv2_imports.py) — pre-populuje `sys.modules` stubami dla wszystkich zależności NAVSIM/nuPlan
- Każdy stub to minimalna klasa lub funkcja o tym samym interfejsie — wystarczająca, by Python zaimportował prawdziwy kod DDV2
- Prawdziwy model `V2TransfuserModel` z repozytorium DDV2 jest używany bez modyfikacji

### Formaty tensorów

| Tensor | Shape | Opis |
|--------|-------|------|
| Panorama 6 kamer | `(1, 3, 256, 1024)` | FL→F→FR→BR→B→BL poziomo, RGB, [0,1] |
| LiDAR BEV | `(1, 1, 256, 256)` | Zera lub syntetyczny z IdealRadar |
| Ego status | `(1, 8)` | [vx, vy, ax, ay, cmd_follow, cmd_left, cmd_right, cmd_straight] |
| Wyjście — trajektorie | `(4, 8, 3)` | 4 propozycje × 8 timestepów × (x, y, yaw) w lokalnych współrzędnych ego |

### GPU vs CPU

Model działa na CPU (~2-3s na inferencję) i GPU (~150-250ms). Zdecydowanie zalecane GPU NVIDIA z CUDA.

## Naprawione problemy (względem oryginalnego kodu)

| Problem | Rozwiązanie |
|---------|-------------|
| Ciemność / tunele | CLAHE + adaptacyjne progi Canny zależne od jasności sceny |
| Tylko przednia kamera | Fuzja kamer FL/FR + wygładzanie EMA + P-controller z wagą pewności |
| Stała prędkość (throttle=0.15) | PI speed controller zależny od krzywizny: 25 km/h na zakrętach, do 100 km/h na prostej |
| Brak detekcji przeszkód | IdealRadar + bounding boxy na obrazie z kamery |
| Brak trybu autonomicznego | 5 poziomów: Traffic AI → LKA+ACC → Custom CV → DDV2 → DDV2+NAV |
| DDV2 jako placeholder | Prawdziwy model V2TransfuserModel 93.2M, stuby importowe zamiast nuplan-devkit |
| Brak nawigacji | NavigationSystem z A* na grafie dróg BeamNG, detekcja skrętów, waypointy |

---

Wszelkie prawa zastrzeżone.
