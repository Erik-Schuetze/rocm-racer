# rocm-racer: To-Be-Implemented Tracker

This document compares the intended architecture (`architecture-overview.md`) against the current state of the codebase, identifying what is fully implemented, partially implemented, and still missing.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented |
| ⚠️ | Partially implemented |
| ❌ | Not yet implemented (stub or missing entirely) |

---

## 1. Hardware & Compute Framework

| Requirement | Status | Notes |
|---|---|---|
| Arch Linux / ROCm 7.1 target | ✅ | Documented in README; `HSA_OVERRIDE_GFX_VERSION=11.0.0` noted |
| PyTorch ROCm-optimized build | ✅ | Specified in `requirements.txt`; installed separately as documented |
| `stable-baselines3`, `gymnasium`, `numpy`, `evdev` | ✅ | All present in `requirements.txt` |

---

## 2. Emulator Execution & State Pipeline

| Requirement | Status | Notes |
|---|---|---|
| PCSX2 launched via `subprocess.Popen` with `-nogui -batch` | ✅ | Implemented in `main.py::launch_pcsx2()` |
| ISO injected via CLI argument | ✅ | `-- /path/to/game.iso` passed correctly |
| `-statefile` argument to boot into save state | ✅ | Implemented in `main.py::launch_pcsx2()` |
| Highway loop `.p2s` save state (sterile anchor) | ✅ | `savestates/rocm-racer-nfsu2-highway.p2s` exists |
| 100% completion `.max` save (bypass tutorials) | ✅ | `saves/need-for-speed-underground-2.5488.max` exists |
| Poll `/proc/[pid]/maps` for 32 MB EE memory block | ✅ | Implemented in `main.py::wait_for_memory_map()` |
| Establish `/proc/[pid]/mem` read hook | ✅ | Implemented in `memory_readers/nfsu2_memory.py` |
| **Episode reset via savestate reload** | ❌ | `PCSX2RacerEnv.reset()` does not yet re-launch/reload PCSX2 savestate; the env exists but is not wired into `main.py` |

---

## 3. Multimodal Observation Space

### 3a. Visual Pipeline (Exteroception) — CNN Branch

| Requirement | Status | Notes |
|---|---|---|
| Capture emulator rendering window via OpenCV (`cv2`) or PIL | ❌ | No screen capture code exists anywhere in the repo |
| Convert frames to grayscale | ❌ | Not implemented |
| Downsample to **84×84** pixels | ❌ | Not implemented |
| Stack **4 consecutive frames** into observation | ❌ | Not implemented |
| Feed frame stack to a **CNN** | ❌ | No CNN model defined; `agents/ppo_agent.py` is a stub |
| Bumper camera perspective + HUD disabled (manual PCSX2 setting) | ✅ | Documented in `implementation-notes-for-human.md` as a manual step |

### 3b. Telemetry Pipeline (Proprioception) — MLP Branch

| Requirement | Status | Notes |
|---|---|---|
| Read `Float32` offsets from EE RAM via `/proc/[pid]/mem` | ✅ | Implemented in `nfsu2_memory.py` |
| Absolute speed (m/s → km/h conversion) | ⚠️ | Field present in `TelemetryOffsets`; **offset value is a placeholder** — must be verified |
| X / Y / Z global coordinates (meters) | ⚠️ | Fields present; **offset values are placeholders** |
| Rotation (X / Y / Z) | ⚠️ | Fields present; **offset values are placeholders** |
| Track progress flag | ⚠️ | Field present; **offset value is a placeholder** |
| Feed 1D telemetry array to **MLP** | ⚠️ | Observation space defined as 8D vector in `pcsx2_env.py`; SB3 default MLP policy used — no explicit custom MLP defined |

### 3c. Latent Fusion

| Requirement | Status | Notes |
|---|---|---|
| Concatenate CNN output + MLP output into single feature vector | ❌ | Not implemented; current observation is telemetry-only (no CNN branch) |
| Custom `MultiInputPolicy` or combined feature extractor | ❌ | No custom policy class exists; `agents/ppo_agent.py` is a stub |

---

## 4. Action Space & Reward Structure

| Requirement | Status | Notes |
|---|---|---|
| Continuous action vector: `[Steering ∈ [-1,1], Throttle ∈ [0,1], Brake ∈ [0,1]]` | ✅ | Defined in `pcsx2_env.py` action space |
| Map actions to virtual DualShock2 analog axes via `VirtualGamepad` | ✅ | Implemented in `virtual_gamepad.py` (ABS_X, ABS_RZ, ABS_Z) |
| Reward: forward velocity (target > 100 km/h) | ✅ | Speed reward term in `pcsx2_env.py::_compute_reward()` |
| Reward: uninterrupted track progression | ✅ | Progress reward term implemented |
| Penalty: zero-velocity states | ✅ | Zero-speed penalty implemented |
| Penalty: reverse driving | ✅ | Reverse penalty implemented |
| Penalty: sudden deceleration (wall collision proxy) | ✅ | Collision penalty implemented |

---

## 5. Training Algorithm

| Requirement | Status | Notes |
|---|---|---|
| PPO as primary algorithm | ✅ | `stable_baselines3.PPO` instantiated in `main.py::train_ppo()` |
| PPO hyperparameters configured | ✅ | `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `lr=3e-4` |
| **PPO wired to `PCSX2RacerEnv`** | ❌ | `train_ppo()` and `PCSX2RacerEnv` both exist but are never connected; `main.py` currently runs keyboard test loop only |
| `model.learn()` called in main training loop | ❌ | Not yet called in `main()` |
| Model checkpointing / saving | ❌ | No `CheckpointCallback` or `model.save()` call exists |
| TensorBoard logging | ⚠️ | Parameter plumbed through `train_ppo()`, but `model.learn()` is never called so logs are never written |
| SAC as alternative algorithm | ❌ | `agents/sac_agent.py` is a 7-line stub |

---

## 6. Supporting Infrastructure

| Item | Status | Notes |
|---|---|---|
| SDL3 game controller DB written before PCSX2 launch | ✅ | `main.py::write_controller_db()` |
| Keyboard fallback controller (`xdotool`) | ✅ | `keyboard_controller.py` fully implemented |
| NFSMW memory reader | ❌ | `nfsmw_memory.py` is a 10-line stub |
| Unit / integration tests | ❌ | No test files in repo |
| CI/CD pipeline | ❌ | No GitHub Actions / workflow files |
| Custom PPO agent wrapper | ❌ | `agents/ppo_agent.py` is a 7-line stub |

---

## Priority Implementation Backlog

The following items must be completed, roughly in dependency order:

### 🔴 Critical (Blocking Training)

1. **Calibrate `TelemetryOffsets` in `nfsu2_memory.py`**  
   All six offsets (speed, pos_x/y/z, rot_x/y/z, flags) are placeholder values.  
   Use Cheat Engine (or `scanmem`) against a live PCSX2 session to locate the vehicle struct in EE RAM.  
   Reference: `docs/underground-2-telemetry-memory-map.md`

2. **Wire `PCSX2RacerEnv` into `main.py`**  
   Replace the current keyboard test loop in `main()` with:
   ```python
   env = PCSX2RacerEnv(config)
   model = train_ppo(env, args.timesteps, args.tensorboard_log)
   model.save("models/ppo_nfsu2")
   ```

3. **Implement episode reset in `PCSX2RacerEnv.reset()`**  
   On each episode start, reload the PCSX2 savestate (via socket command or re-launch) to restore the sterile highway anchor.

### 🟡 Important (Core Architecture Gaps)

4. **Visual observation pipeline (CNN branch)**  
   Capture the PCSX2 window, convert to grayscale 84×84, stack 4 frames.  
   Add `opencv-python` (or `Pillow`) to `requirements.txt`.  
   Define a `FrameStack` wrapper or use `gymnasium.wrappers.FrameStack`.

5. **Latent fusion policy (CNN + MLP)**  
   Define a custom `CombinedExtractor` feature extractor combining:
   - CNN branch for visual frames (`spaces.Box` image input)
   - MLP branch for telemetry vector  
   Pass as `policy_kwargs={"features_extractor_class": CombinedExtractor}` to `PPO(...)`.  
   Implement in `agents/ppo_agent.py`.

6. **Model checkpointing**  
   Add `stable_baselines3.common.callbacks.CheckpointCallback` to `model.learn()` so training progress survives interruptions.

### 🟢 Future / Nice-to-Have

7. **SAC agent** (`agents/sac_agent.py`) — alternative algorithm for improved sample efficiency.

8. **NFSMW memory reader** (`memory_readers/nfsmw_memory.py`) — same RenderWare engine, independent offset calibration required.

9. **Unit tests** — at minimum, mock-based tests for `NFSU2MemoryReader` and `PCSX2RacerEnv` to validate reward shaping logic without a live emulator.

10. **Evaluation / rollout callback** — periodic evaluation episodes (deterministic policy) logged to TensorBoard for training progress monitoring.
