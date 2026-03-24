# rocm-racer

## Project Overview
`rocm-racer` is a Reinforcement Learning project designed to train an autonomous driving agent in *Need for Speed: Underground 2* running on the PCSX2 emulator. The agent relies exclusively on continuous control and telemetry data extracted directly from the emulator's memory block, bypassing visual pixel processing entirely.

## Hardware and Stack Constraints
* **OS:** Arch Linux
* **GPU:** AMD Radeon RX 7900 XTX (RDNA3)
* **Compute Framework:** AMD ROCm 7.1
* **Machine Learning Stack:** Python 3.12, PyTorch (ROCm build), Stable-Baselines3, Gymnasium

## Algorithm Focus
* **Primary Algorithm:** Proximal Policy Optimization (PPO).
* **Objective:** PPO is utilized for its stability in continuous action spaces and robustness against minor frame-pacing inconsistencies inherent to emulation.

## Execution Strategy
1. **Environment Initialization:** The Python training script (`main.py`) programmatically launches the PCSX2 process via CLI, passing the game ISO and the `-statefile` argument to boot directly into a pre-configured free-drive Save State.
2. **Memory Access:** The script connects to PCSX2's built-in PINE IPC server (Unix socket) to read the 32 MB Emotion Engine (EE) RAM directly using PS2-side addresses. No kernel permission changes required.
3. **Telemetry Extraction:** The agent reads known struct offsets (position, velocity, speed, rotation, RPM, gear) relative to a calibrated vehicle struct base pointer in EE RAM.
4. **Episodic Reset:** Upon failure (e.g., wall collision, zero velocity), the Gymnasium wrapper sends a command to PCSX2 to instantly reload the anchor Save State (`.p2s`) on the highway loop to restart the training cycle.
