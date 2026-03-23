# rocm-racer

`rocm-racer` is a reinforcement learning project for training an autonomous driving agent against PlayStation 2 Need for Speed titles running in PCSX2 on Arch Linux with AMD ROCm.

The current starting point is **Need for Speed: Underground 2 (US)**. Support for **Need for Speed: Most Wanted (US)** is planned alongside it, but initial bring-up and offset calibration should assume Underground 2 first.

## Requirements

- Arch Linux
- Python 3.12
- AMD ROCm 7.1
- PyTorch ROCm build
- PCSX2
- US game images for:
  - `Need for Speed: Underground 2 (USA, Canada)`
  - `Need for Speed: Most Wanted - Black Edition (USA)`

## Setup

Follow these steps in order on a fresh machine.

### 1. Clone and create the Python environment

```bash
git clone <repo-url> rocm-racer
cd rocm-racer
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm support first (see requirements.txt for details)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.3

pip install -r requirements.txt
```

### 2. Install PCSX2

Install PCSX2 from the AUR or your preferred Arch package source:

```bash
yay -S pcsx2-qt
```

Verify the binary is at `/usr/bin/pcsx2-qt` (the path expected by `main.py`).

### 3. Place game ISOs

Copy both US game images into the `iso/` directory with these exact filenames:

```
iso/Need for Speed - Underground 2 (USA, Canada).iso
iso/Need for Speed - Most Wanted - Black Edition (USA).iso
```

### 4. Register the memory card with PCSX2

The repo ships a pre-configured memory card at `memcards/rocm-racer.ps2` that contains the 100% completed Underground 2 save. PCSX2 must be told to use it.

Copy (or symlink) it into the PCSX2 memcards directory:

```bash
mkdir -p ~/.config/PCSX2/memcards
cp memcards/rocm-racer.ps2 ~/.config/PCSX2/memcards/rocm-racer.ps2
```

Then assign it to Slot 1 inside PCSX2:

1. Open PCSX2.
2. Go to **Settings → Memory Cards**.
3. Assign `rocm-racer.ps2` to **Slot 1**.

Until this is done the Underground 2 save file will not be visible in-game.

> **Recreating the memory card from scratch:** If you need to rebuild it, install
> `mymcplusplus` (`pip install mymcplusplus[gui]`), create a blank card inside PCSX2
> named `rocm-racer`, and import a North American 100% save from
> [GameFAQs](https://gamefaqs.gamespot.com/gamecube/920466-need-for-speed-underground-2/saves#playstation-2-ps3-virtual-memory-card-save-zip-north-america).
> Reference: [PCSX2 memcard docs](https://pcsx2.net/docs/configuration/memcards/#using-mymc-1).

### 5. Verify the highway loop save state

The training anchor save state is already committed to the repo:

```
savestates/rocm-racer-nfsu2-highway.p2s
```

`main.py` loads it automatically at startup — no additional steps are needed unless you want to create a new one. To recreate it:

1. Boot **Need for Speed: Underground 2** in PCSX2 with `rocm-racer.ps2` in Slot 1.
2. Load the 100% completed save and start a free-drive session on the highway loop.
3. Once on the highway, save a state via the PCSX2 save state menu.
4. Replace `savestates/rocm-racer-nfsu2-highway.p2s` with the resulting `.p2s` file.

### 6. Add your user to the `input` group

The virtual gamepad uses `/dev/uinput`. Grant your user write access and re-login:

```bash
sudo usermod -aG input $USER
# log out and back in for the change to take effect
```

### 7. Controller bindings — automated

`main.py` handles the two controller setup steps automatically on every run:

1. Writes `~/.config/PCSX2/game_controller_db.txt` with the SDL3 game controller mapping for the virtual gamepad **before** PCSX2 starts. PCSX2 loads this file via `SDL_HINT_GAMECONTROLLERCONFIG_FILE` during SDL init, so the device is recognised as a gamepad immediately.
2. Creates the uinput virtual gamepad device **before** launching PCSX2 so it is present during SDL device enumeration.

No manual controller mapping in the PCSX2 GUI is required. The `[Pad1]` section of `~/.config/PCSX2/inis/PCSX2.ini` uses SDL3 binding names (`FaceSouth`, `+LeftTrigger`, etc.) already configured by this project.

## Running

```bash
source venv/bin/activate
python main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--game` | `nfsu2` | Which game ISO to launch (`nfsu2` or `nfsmw`) |
| `--statefile <path>` | `savestates/rocm-racer-nfsu2-highway.p2s` | Override the save state loaded at startup |
| `--timesteps <n>` | `1000000` | Total PPO training timesteps |
| `--tensorboard-log <dir>` | *(none)* | Directory for TensorBoard logs |
| `--no-launch` | *(off)* | Skip launching PCSX2 (assume it is already running) |

## Project status

The repository currently contains:

- a Linux `/proc`-based memory reader scaffold for Underground 2 telemetry
- a custom Gymnasium environment scaffold for PCSX2-driven RL

The next major steps are:

- calibrate real RAM offsets for Underground 2
- wire emulator/controller input injection
- add PPO and SAC training entrypoints
