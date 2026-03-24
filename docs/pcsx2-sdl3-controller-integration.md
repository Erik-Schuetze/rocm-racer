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
| Accelerate  | **✕ (Cross)** or Right stick ↑ |
| Brake/Reverse | □ (Square) or Right stick ↓ |
| Change camera | △ (Triangle) |
| Look back   | ○ (Circle)  |
| E-brake     | R1          |
| Nitrous     | L1          |
| Steer       | Left stick X |
| Shift up    | R2 (manual only) |
| Shift down  | L2 (manual only) |

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

## Input timing: why sending input too early silently fails

The virtual gamepad must send its first input **after** PCSX2 has fully loaded the savestate and the game is running. Sending input earlier appears to work (no errors) but the game never sees it.

### The problem

When PCSX2 loads a savestate, it restores the PS2 PAD (controller) state from `PAD.bin` inside the `.p2s` archive. This overwrites whatever SDL has reported so far. If a button was already held when the savestate finishes loading, the game sees the restored "no buttons pressed" PAD state and, because the button is already down, **no new press event is generated** — the game never detects a transition.

A physical controller doesn't have this problem because the user presses buttons *after* the game is visibly running.

### PCSX2 startup timeline (typical, from emulog)

| Time     | Event                                      |
|----------|--------------------------------------------|
| T+0.00s  | PCSX2 process starts                       |
| T+0.004s | EE/IOP memory allocated (visible in maps)  |
| T+0.57s  | SDL3 input initialised, mappings loaded    |
| T+0.79s  | `PAD.bin` found in savestate (PAD restored)|
| T+0.89s  | SPU2 + GS restore from savestate           |
| T+0.95s  | SDL opens virtual gamepad as SDL-0         |
| T+~1.0s  | Game starts rendering (first frame)        |

The gamepad connection at T+0.95s happens *during or just after* the savestate restore. Any button state present at that instant is read by SDL's `PollAllValues()` (via `EVIOCGKEY`), but the resulting SDL event is either consumed before the PAD restore completes or overwritten by the restored PAD state.

### The fix: `wait_for_pcsx2_ready()`

`main.py` monitors `~/.config/PCSX2/logs/emulog.txt` for the line:

```
SDLInputSource: Opened gamepad 1 (instance id 1, player id 0): rocm-racer Virtual Gamepad
```

This confirms both that the savestate has been loaded and that SDL has connected the virtual gamepad. After detecting this marker, a configurable post-ready delay (default 2 seconds) lets the game run several frames and establish its baseline controller state before any input is sent.

### Important: PCSX2 must have `EnableControllerLogs = true`

The `Opened gamepad` marker is logged by `SDLInputSource`. If controller logging is disabled, the emulog won't contain this line and the readiness check falls back to a fixed timeout. Verify the setting in `~/.config/PCSX2/inis/PCSX2.ini`:

```ini
[Logging]
EnableControllerLogs = true
```

(PCSX2 defaults this to `true`; only set it explicitly if you've changed it.)

### Emulog location and staleness

The emulog at `~/.config/PCSX2/logs/emulog.txt` is overwritten on each PCSX2 launch. `wait_for_pcsx2_ready()` polls it every 250ms. Because the file is rewritten from scratch, there is no risk of matching a stale marker from a previous run — the marker only appears once the *current* PCSX2 instance has connected the gamepad.

### Checklist for debugging "car doesn't move"

1. **Verify gamepad detection** — check emulog for `"Opened gamepad"` and `"Gamepad 0 has 6 axes and 12 buttons"`
2. **Verify bindings** — `~/.config/PCSX2/inis/PCSX2.ini` `[Pad1]` section must use SDL3 names (`SDL-0/FaceSouth`, not `SDL-0/Button0`)
3. **Verify timing** — the `[rocm-racer]` console output should show `"PCSX2 ready"` *before* `"Test mode: accelerating"`
4. **Verify PauseOnFocusLoss** — must be `false` in `PCSX2.ini` under `[UI]`, otherwise the game pauses when PCSX2 doesn't have window focus
5. **Verify game state** — the savestate must have the car on a driveable road (not in a menu, cutscene, or garage)
6. **Try increasing post-ready delay** — pass a longer delay if the game takes more time to settle (e.g. on slower hardware): edit `wait_for_pcsx2_ready(post_ready_delay=5.0)` in `main.py`

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

## PINE IPC: reading PS2 memory without ptrace

PCSX2 has a built-in IPC server called **PINE** (PCSX2 IPC Network Extension) that provides direct read/write access to PS2 EE memory over a Unix domain socket. This bypasses all ptrace, `/proc/pid/mem`, and kernel permission issues.

### Why PINE is needed

On Arch Linux, the `pcsx2-qt` binary has filesystem capabilities (`cap_net_admin,cap_net_raw=eip`). The kernel automatically sets `PR_SET_DUMPABLE=0` for cap-elevated binaries, which makes `/proc/pid/mem` owned by `root:root` and inaccessible to the same user — even with `ptrace_scope=0`. Neither `PTRACE_SEIZE` nor `suid_dumpable=1` reliably fixes this at runtime.

PINE communicates over a Unix domain socket and reads EE memory from within the PCSX2 process itself, so no external memory access is needed.

### Enabling PINE

In `~/.config/PCSX2/inis/PCSX2.ini`:

```ini
EnablePINE = true
PINESlot = 28011
```

Or in the PCSX2 GUI: **Settings → Advanced → Enable PINE IPC**.

**PCSX2 must be restarted** after enabling PINE. The socket is created during PCSX2 startup at:

```
$XDG_RUNTIME_DIR/pcsx2.sock     (typically /run/user/1000/pcsx2.sock)
```

If `XDG_RUNTIME_DIR` is unset, it falls back to `/tmp/pcsx2.sock`.

### How the memory reader uses PINE

`memory_readers/nfsu2_memory.py` uses PINE exclusively for all memory access:

1. Connect to the PINE Unix socket (`$XDG_RUNTIME_DIR/pcsx2.sock`)
2. All memory reads go through PINE using PS2-side addresses directly — no PID detection, no EE base resolution, no `/proc/pid/mem` fallback
3. PINE must be enabled in PCSX2 (`EnablePINE = true` in `PCSX2.ini`)

### PINE protocol (for reference)

The protocol is a simple binary format over the Unix socket:

| Direction | Format |
|-----------|--------|
| Request   | `[total_size u32 LE][opcode u8][params...]` (can batch multiple commands) |
| Response  | `[total_size u32 LE][result_code u8 (0=OK, 0xFF=FAIL)][data...]` |

Key opcodes: `Read8=0`, `Read16=1`, `Read32=2`, `Read64=3`, `Write32=6`, `Version=8`, `SaveState=9`, `LoadState=0xA`, `Status=0xF`.

The `memory_readers/pine_client.py` module implements a pure-Python client with support for batched reads (multiple Read32 commands in a single round-trip), which is used for bulk 32MB EE RAM scans during calibration.

### Verifying PINE is working

```bash
python -c "
from memory_readers.pine_client import PINEClient
with PINEClient() as pine:
    print('Connected to PINE')
    print(f'Version: {pine.get_version()}')
    print(f'Status: {pine.get_status()}')  # 0=Running
    print(f'Title: {pine.get_game_title()}')
"
```
