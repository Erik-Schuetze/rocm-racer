# PCSX2 SDL3 Controller Integration

Research notes covering how PCSX2 2.6.x consumes SDL3 input and how our uinput virtual gamepad is wired into it.

## PCSX2 uses SDL3, not SDL2

PCSX2 2.6.x (and later) migrated to **SDL3**. This matters because:

- SDL3 renamed the gamepad API (`SDL_GameController*` → `SDL_Gamepad*`, `SDL_CONTROLLER_BUTTON_*` → `SDL_GAMEPAD_BUTTON_*`)
- The `PCSX2.ini` binding names changed accordingly — old SDL2-era names like `Cross`, `Circle`, `L2`, `R2` are silently ignored; SDL3 names must be used
- A backwards-compatibility shim exists for the old positional face button names (`A`, `B`, `X`, `Y`) but it triggers a migration warning; do not rely on it

## How PCSX2 loads controller mappings

Source: `pcsx2/Input/SDLInputSource.cpp`, `SetHints()`:

```cpp
static constexpr const char* CONTROLLER_DB_FILENAME = "game_controller_db.txt";

if (FileSystem::FileExists(upath.c_str()))
    SDL_SetHint(SDL_HINT_GAMECONTROLLERCONFIG_FILE, upath.c_str());
```

PCSX2 checks for `game_controller_db.txt` in the user data directory (`~/.config/PCSX2/`) **before** calling `SDL_InitSubSystem`. If the file exists it is passed to SDL3 via `SDL_HINT_GAMECONTROLLERCONFIG_FILE` so all mappings in it are loaded before any device is enumerated.

**Consequence for our virtual gamepad:** the file must exist before PCSX2 starts. `main.py` writes it automatically on every run.

## SDL3 GUID format for our uinput device

The SDL3 device GUID is 16 bytes, formatted as a 32-character hex string:

```
Bytes 0-1  (LE16): bustype
Bytes 2-3         : 0x0000 padding
Bytes 4-5  (LE16): vendor
Bytes 6-7         : 0x0000 padding
Bytes 8-9  (LE16): product
Bytes 10-11       : 0x0000 padding
Bytes 12-13 (LE16): version
Bytes 14-15       : 0x0000 padding
```

Our `VirtualGamepad` is constructed with:

| Field   | Value  |
|---------|--------|
| bustype | `BUS_USB` = `0x0003` |
| vendor  | `0x0000` |
| product | `0x0000` |
| version | `0x0003` |

Resulting GUID: **`03000000000000000000000003000000`**

This GUID is the primary key in `game_controller_db.txt` and must exactly match the identity of the evdev device that SDL3 enumerates.

## SDL3 game controller mapping format

```
GUID,Device Name,<button/axis mappings>,platform:Linux,
```

Button tokens: `a:bN` (SDL button N = evdev button at scan index N)  
Axis tokens: `leftx:aN`, `lefttrigger:+aN` (the `+` prefix means positive half-axis — for triggers that go 0→255, not −128→127)  
Hat token: `dpup:h0.1` (hat 0, bit-flag direction: up=1, right=2, down=4, left=8)

Our mapping (also defined as `SDL_MAPPING` in `virtual_gamepad.py`):

```
03000000000000000000000003000000,rocm-racer Virtual Gamepad,
a:b0,b:b1,y:b2,x:b3,
leftshoulder:b4,rightshoulder:b5,
back:b8,start:b9,
leftstick:b10,rightstick:b11,
leftx:a0,lefty:a1,lefttrigger:+a2,
rightx:a3,righty:a4,righttrigger:+a5,
dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,
platform:Linux,
```

### Button index derivation

SDL3 assigns joystick button indices by scanning the evdev `EV_KEY` bit field in ascending BTN code order, counting only codes that are present in the device's capability mask:

| evdev code | Name        | SDL index | SDL gamepad role |
|------------|-------------|-----------|------------------|
| `0x130`    | `BTN_SOUTH` | b0        | `a` → FaceSouth (Cross)   |
| `0x131`    | `BTN_EAST`  | b1        | `b` → FaceEast  (Circle)  |
| `0x132`    | `BTN_C`     | —         | absent, gap skipped       |
| `0x133`    | `BTN_NORTH` | b2        | `y` → FaceNorth (Triangle)|
| `0x134`    | `BTN_WEST`  | b3        | `x` → FaceWest  (Square)  |
| `0x135`    | `BTN_Z`     | —         | absent, gap skipped       |
| `0x136`    | `BTN_TL`    | b4        | `leftshoulder`  (L1)      |
| `0x137`    | `BTN_TR`    | b5        | `rightshoulder` (R1)      |
| `0x138`    | `BTN_TL2`   | b6        | digital L2 fallback       |
| `0x139`    | `BTN_TR2`   | b7        | digital R2 fallback       |
| `0x13a`    | `BTN_SELECT`| b8        | `back`  (Select)          |
| `0x13b`    | `BTN_START` | b9        | `start` (Start)           |
| `0x13c`    | `BTN_MODE`  | —         | absent, gap skipped       |
| `0x13d`    | `BTN_THUMBL`| b10       | `leftstick`  (L3)         |
| `0x13e`    | `BTN_THUMBR`| b11       | `rightstick` (R3)         |

### Axis index derivation

SDL3 assigns joystick axis indices by scanning the evdev `EV_ABS` bit field in ascending ABS code order:

| ABS code | Name        | SDL axis | SDL gamepad role                      |
|----------|-------------|----------|---------------------------------------|
| `0`      | `ABS_X`     | a0       | `leftx`        (left stick X)         |
| `1`      | `ABS_Y`     | a1       | `lefty`        (left stick Y)         |
| `2`      | `ABS_Z`     | a2       | `lefttrigger`  (L2, 0–255 → `+a2`)   |
| `3`      | `ABS_RX`    | a3       | `rightx`       (right stick X)        |
| `4`      | `ABS_RY`    | a4       | `righty`       (right stick Y)        |
| `5`      | `ABS_RZ`    | a5       | `righttrigger` (R2, 0–255 → `+a5`)   |
| `16/17`  | `ABS_HAT0X/Y` | hat 0  | d-pad (mapped with `h0.N` tokens)    |

## PCSX2.ini binding names (SDL3)

Source: `s_sdl_button_setting_names[]` and `s_sdl_axis_setting_names[]` in `SDLInputSource.cpp`.

### Buttons

| `[Pad1]` key | SDL3 name            | PS2 button  |
|--------------|----------------------|-------------|
| `Cross`      | `SDL-0/FaceSouth`    | ✕           |
| `Circle`     | `SDL-0/FaceEast`     | ○           |
| `Triangle`   | `SDL-0/FaceNorth`    | △           |
| `Square`     | `SDL-0/FaceWest`     | □           |
| `L1`         | `SDL-0/LeftShoulder` | L1          |
| `R1`         | `SDL-0/RightShoulder`| R1          |
| `L3`         | `SDL-0/LeftStick`    | L3          |
| `R3`         | `SDL-0/RightStick`   | R3          |
| `Select`     | `SDL-0/Back`         | Select      |
| `Start`      | `SDL-0/Start`        | Start       |
| `Up`         | `SDL-0/DPadUp`       | D-pad Up    |
| `Down`       | `SDL-0/DPadDown`     | D-pad Down  |
| `Left`       | `SDL-0/DPadLeft`     | D-pad Left  |
| `Right`      | `SDL-0/DPadRight`    | D-pad Right |

### Axes

| `[Pad1]` key | SDL3 name              | PS2 input      |
|--------------|------------------------|----------------|
| `L2`         | `SDL-0/+LeftTrigger`   | L2 analog      |
| `R2`         | `SDL-0/+RightTrigger`  | R2 analog      |
| `LLeft`      | `SDL-0/-LeftX`         | Left stick ←   |
| `LRight`     | `SDL-0/+LeftX`         | Left stick →   |
| `LUp`        | `SDL-0/-LeftY`         | Left stick ↑   |
| `LDown`      | `SDL-0/+LeftY`         | Left stick ↓   |
| `RLeft`      | `SDL-0/-RightX`        | Right stick ←  |
| `RRight`     | `SDL-0/+RightX`        | Right stick →  |
| `RUp`        | `SDL-0/-RightY`        | Right stick ↑  |
| `RDown`      | `SDL-0/+RightY`        | Right stick ↓  |

The `+`/`-` prefix sets the axis direction (`InputModifier::None` vs `InputModifier::Negate`). For half-axes like triggers, `+` means "only the positive half counts as pressed".

## Hotplug support

PCSX2 polls SDL events every frame via `SDLInputSource::PollEvents()` and handles:

- `SDL_EVENT_GAMEPAD_ADDED` → opens the device as a full gamepad (game controller mapping exists)
- `SDL_EVENT_JOYSTICK_ADDED` (only when `!SDL_IsGamepad()`) → opens as a raw joystick

Devices connected after PCSX2 starts are fully supported. However, creating the virtual gamepad **before** launching PCSX2 is more reliable because the mapping file is guaranteed to be loaded before enumeration begins.

## NFS Underground 2 PS2 control scheme

In NFS Underground 2 (PS2, NTSC-U), the default on-foot/driving controls are:

| Action      | PS2 button |
|-------------|------------|
| Accelerate  | **✕ (Cross)** |
| Brake/Reverse | ○ (Circle) |
| Handbrake   | □ (Square)  |
| Nitrous     | △ (Triangle) |
| Steer       | Left stick X |

**Cross (✕) is accelerate**, not R2. This is the PS2 default for EA Black Box racing games of this era. `BTN_SOUTH` → SDL `FaceSouth` → PCSX2 `Cross` maps to this.

## Fallback: raw joystick bindings

If the `game_controller_db.txt` GUID does not match (e.g. after a kernel or SDL update changes the GUID derivation), SDL3 will not promote the device to gamepad mode. It will still be enumerated as a raw joystick under `SDL_EVENT_JOYSTICK_ADDED`. In that case, use raw joystick binding names instead:

| Action        | Raw binding       |
|---------------|-------------------|
| Cross (accel) | `SDL-0/JoyButton0`|
| L2            | `SDL-0/+JoyAxis2` |
| R2            | `SDL-0/+JoyAxis5` |
| Left stick X+ | `SDL-0/+JoyAxis0` |
| Left stick X- | `SDL-0/-JoyAxis0` |

To verify which mode SDL3 assigned, check the PCSX2 console log for either `"Gamepad N inserted"` (gamepad path) or `"Joystick N inserted"` (raw joystick path).

## Keyboard fallback (xdotool)

When `/dev/uinput` access is unavailable, keypresses can be injected directly into the PCSX2 X11 window using `xdotool`. This requires no special OS permissions — only X11 session access, which any desktop user already has.

```bash
pacman -S xdotool
```

`memory_readers/keyboard_controller.py` wraps `xdotool key --window <id>` and finds the PCSX2 renderer window by PID:

```python
kbd = KeyboardController(pid=pcsx2_proc.pid)
kbd.press("k", duration=0.1)   # K = Cross (✕) = accelerate in NFSU2
```

`xdotool` targets the window directly — PCSX2 does not need focus.

### PCSX2 default keyboard bindings (Pad1)

| Key | PS2 button | NFSU2 action |
|-----|------------|--------------|
| `k` | Cross ✕   | Accelerate   |
| `l` | Circle ○  | Brake/Reverse|
| `j` | Square □  | Handbrake    |
| `i` | Triangle △| Nitrous      |
| `a` / `d` | Left stick | Steer |
