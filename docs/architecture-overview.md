# rocm-racer: Architecture and Software Pipeline Overview

This document outlines the system architecture, data pipelines, and execution flow for `rocm-racer`, a multimodal Reinforcement Learning agent designed for continuous driving control in *Need for Speed: Underground 2* via the PCSX2 emulator.

### Technical Deep-Dive

**1. Hardware and Compute Framework**
The environment is optimized for AMD silicon running on a rolling-release Linux distribution.
* **Operating System:** Arch Linux
* **GPU:** AMD Radeon RX 7900 XTX (RDNA3 / `gfx1100`)
* **Compute Stack:** AMD ROCm 7.1 (`HSA_OVERRIDE_GFX_VERSION=11.0.0`)
* **Machine Learning Stack:** Python 3.12, PyTorch (ROCm-optimized build), Stable-Baselines3, Gymnasium

**2. Emulator Execution and State Pipeline**
The training loop utilizes a headless-capable, programmatic boot sequence to ensure a sterile and reproducible environment.
* **Emulator:** PCSX2 (Qt build) configured for **1x Native** internal resolution and **4:3** aspect ratio.
* **Initialization:** `main.py` launches the emulator via CLI `subprocess.Popen`, injecting the `NeedForSpeed.iso` and a `-statefile` argument to boot directly into a pre-configured Save State (`.p2s`).
* **Environment Sterility:** The Save State is anchored in free-drive mode on a continuous highway loop, derived from a **100% completion .max save** to bypass all tutorial scripts and locked regions.
* **Memory Access:** The Python environment connects to PCSX2's built-in PINE IPC server (a Unix socket at `$XDG_RUNTIME_DIR/pcsx2.sock`) to read the 32 MB Emotion Engine (EE) RAM directly using PS2-side addresses. No `/proc/pid/mem`, no ptrace, no kernel permission changes required. Enable PINE in `~/.config/PCSX2/inis/PCSX2.ini` with `EnablePINE = true`.

**3. Multimodal Observation Space**
The agent observes the environment through a dual-pipeline architecture, merging visual spatial awareness with absolute internal telemetry.



* **Visual Pipeline (Exteroception):** * Captures the emulator rendering window using OpenCV (`cv2`) or PIL. 
  * The game is set to the **bumper camera** perspective with the **HUD disabled** to enforce spatial invariance.
  * Frames are converted to grayscale and downsampled to an **84x84 pixel** matrix to minimize VRAM usage.
  * A frame stack of **4 consecutive frames** is passed to a **Convolutional Neural Network (CNN)** to calculate dynamic object trajectories.
* **Telemetry Pipeline (Proprioception):**
  * Extracts hardcoded `Float32` offsets relative to the vehicle's base pointer in the EE RAM.
  * Key metrics include absolute speed (converted from m/s to **km/h**) and X/Y/Z global coordinates (in **meters**).
  * This 1D array is processed through a standard **Multi-Layer Perceptron (MLP)**.
* **Latent Fusion:** The outputs of the CNN and MLP are concatenated into a single feature vector before entering the policy network.

**4. Action Space and Reward Structure**
* **Action Output:** A continuous vector array `[Steering (-1.0 to 1.0), Throttle (0.0 to 1.0), Brake (0.0 to 1.0)]` controlling the analog inputs of the virtual PS2 controller.
* **Reward Function:** A continuous mathematical model prioritizing forward velocity (target > **100 km/h**) and uninterrupted track progression, heavily penalizing zero-velocity states, reverse driving, and sudden deceleration indicative of wall collisions.

**5. Training Algorithm**
* **Algorithm:** Proximal Policy Optimization (**PPO**).
* **Objective:** PPO provides stable, on-policy gradient updates that are highly resistant to the minor frame-pacing irregularities and latency spikes inherent to running an emulation layer alongside a PyTorch training loop.

### Key Takeaways

* **Compute Stack:** The project relies on **Arch Linux** and **ROCm 7.1** to accelerate PyTorch training on the **RX 7900 XTX**.
* **Automation:** PCSX2 boot sequences and episodic resets are fully automated via CLI arguments and Save States to bypass UI menus.
* **Multimodal Fusion:** The observation space combines an **84x84 pixel** grayscale frame stack (processed via CNN) with memory-extracted speed and coordinate telemetry (processed via MLP).
* **Visual Optimization:** Utilizing the bumper camera and disabling the HUD removes spatial noise and prevents the neural network from overfitting to static UI elements.
* **Algorithm:** **PPO** is utilized as the primary training algorithm due to its robust stability in continuous action spaces.
