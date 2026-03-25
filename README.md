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

### Training

```bash
# 1. Calibrate telemetry offsets (one-time, interactive)
python main.py --calibrate

# 2. (Optional) Create extra starting positions for multi-start training
#    Place rocm-racer-nfsu2-highway-1.p2s … highway-9.p2s in savestates/
#    then load them into PINE slots:
python main.py --setup-savestates

# 3. Run PPO training
python main.py --train --timesteps 1000000

# Resume from a checkpoint
python main.py --train --load-model models/ppo_nfsu2_20260325_final.zip
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--game` | `nfsu2` | Which game ISO to launch (`nfsu2` or `nfsmw`) |
| `--statefile <path>` | `savestates/rocm-racer-nfsu2-highway.p2s` | Override the save state loaded at startup |
| `--timesteps <n>` | `1000000` | Total PPO training timesteps |
| `--tensorboard-log <dir>` | `runs/` | Directory for TensorBoard logs |
| `--no-launch` | *(off)* | Skip launching PCSX2 (assume it is already running) |
| `--train` | *(off)* | Run PPO training loop |
| `--calibrate` | *(off)* | Discover vehicle struct via differential scanning |
| `--setup-savestates` | *(off)* | Load extra `.p2s` files into PINE slots for multi-start |
| `--num-envs <n>` | `1` | Number of parallel PCSX2 environments |
| `--turbo` | *(off)* | Run PCSX2 at 2× emulation speed |
| `--load-model <path>` | *(none)* | Resume training from a saved model |
| `--checkpoint-freq <n>` | `10000` | Save a model checkpoint every N timesteps |
| `--device` | `cuda` | PyTorch device (`cuda`, `cpu`) |
| `--no-preview` | *(off)* | Disable the live OpenCV preview window |

## Training features

- **Freeze/resume around gradient updates:** Environments are frozen (inputs zeroed, state saved to PINE slot 9) before each PPO gradient update and restored afterward, preventing the "free distance" bug where cars drift uncontrolled during GPU compute.
- **Multi-savestate random starts:** Place up to 9 `.p2s` files in `savestates/` (named `rocm-racer-nfsu2-highway-1.p2s` through `9.p2s` — the suffix matches the PINE slot), run `--setup-savestates`, and each episode will randomly pick a starting position. PCSX2 has 10 quicksave slots (0–9); slot 0 is reserved for freeze/resume, leaving slots 1–9 for training starts.
- **Steering smoothness reward:** A gentle per-step penalty (`-0.005 × |Δsteering|`) discourages jittery oscillation without penalizing legitimate swerving or stabilization.
- **Curriculum distance escalation:** Training starts with a 500m success goal. When 50% of the last 50 episodes reach the goal, it is automatically raised by 500m (500 → 1000 → 1500 → …).

## Project status

The agent is actively training on the NFSU2 highway loop using PPO with multimodal observations (96×96 grayscale frame stack + telemetry). Approximately 25% of runs reach the 1000m distance goal. The repository includes:

- PINE IPC-based telemetry extraction with automated calibration (`--calibrate`)
- Multimodal CNN+MLP feature extractor for SB3 PPO
- Virtual gamepad (uinput) with SDL3 controller mapping
- Frame capture via grim/hyprctl on Wayland
- Multi-instance parallel training with `ThreadedVecEnv`
- Curriculum learning with auto-escalating distance goals
- Gradient-update freeze/resume to prevent free-distance reward bugs
