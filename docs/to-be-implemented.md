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
| Poll `/proc/[pid]/maps` for 32 MB EE memory block | ❌ | Removed — PINE IPC is used instead |
| Establish `/proc/[pid]/mem` read hook | ❌ | Removed — PINE IPC is used instead |
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
| Feed frame stack to a **CNN** | ❌ | No CNN model defined |
| Bumper camera perspective + HUD disabled (manual PCSX2 setting) | ✅ | Documented in `implementation-notes-for-human.md` as a manual step |

### 3b. Telemetry Pipeline (Proprioception) — MLP Branch

| Requirement | Status | Notes |
|---|---|---|
| Read `Float32` offsets from EE RAM via PINE IPC | ✅ | Implemented in `nfsu2_memory.py` |
| Absolute speed (m/s → km/h conversion) | ⚠️ | Field present in `TelemetryOffsets`; offsets from new doc, needs live verification |
| X / Y / Z global coordinates (meters) | ⚠️ | Fields present; offsets from new doc, needs live verification |
| Rotation (X / Y / Z) | ⚠️ | Fields present; offsets from new doc, needs live verification |
| Track progress flag | ⚠️ | Field present; **offset value is a placeholder** |
| Feed 1D telemetry array to **MLP** | ⚠️ | Observation space defined as 8D vector in `pcsx2_env.py`; SB3 default MLP policy used — no explicit custom MLP defined |

### 3c. Latent Fusion

| Requirement | Status | Notes |
|---|---|---|
| Concatenate CNN output + MLP output into single feature vector | ❌ | Not implemented; current observation is telemetry-only (no CNN branch) |
| Custom `MultiInputPolicy` or combined feature extractor | ❌ | No custom policy class exists |

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
| PPO as primary algorithm | ⚠️ | `stable_baselines3.PPO` referenced; `train_ppo()` removed during cleanup, needs re-implementation |
| PPO hyperparameters configured | ❌ | Removed during cleanup; needs re-implementation when training loop is wired |
| **PPO wired to `PCSX2RacerEnv`** | ❌ | `PCSX2RacerEnv` exists but is not connected to any training loop |
| `model.learn()` called in main training loop | ❌ | Not yet called in `main()` |
| Model checkpointing / saving | ❌ | No `CheckpointCallback` or `model.save()` call exists |
| TensorBoard logging | ❌ | Not yet implemented; needs training loop |
| SAC as alternative algorithm | ❌ | Not implemented (stub removed during cleanup) |

---

## 6. Supporting Infrastructure

| Item | Status | Notes |
|---|---|---|
| SDL3 game controller DB written before PCSX2 launch | ✅ | `main.py::write_controller_db()` |
| Keyboard fallback controller (`xdotool`) | ❌ | Removed during cleanup (unused fallback) |
| NFSMW memory reader | ❌ | Removed during cleanup (empty stub) |
| Unit / integration tests | ❌ | No test files in repo |
| CI/CD pipeline | ❌ | No GitHub Actions / workflow files |
| Custom PPO agent wrapper | ❌ | Removed during cleanup (empty stub) |

---

## Priority Implementation Backlog

The following items must be completed, roughly in dependency order:

### 🔴 Critical (Blocking Training)

1. **Calibrate `TelemetryOffsets` in `nfsu2_memory.py`**  
   Offsets updated from the new memory architecture doc. Run `--calibrate` to discover the vehicle struct base and static pointer via structural fingerprinting.  
   Reference: `docs/ps2-memory-architecture-and-telemetry-extraction-for-nfs-u2.md`

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

7. **SAC agent** — alternative algorithm for improved sample efficiency (not yet implemented).

8. **NFSMW memory reader** — same RenderWare engine, independent offset calibration required (not yet implemented).

9. **Unit tests** — at minimum, mock-based tests for `NFSU2MemoryReader` and `PCSX2RacerEnv` to validate reward shaping logic without a live emulator.

10. **Evaluation / rollout callback** — periodic evaluation episodes (deterministic policy) logged to TensorBoard for training progress monitoring.
