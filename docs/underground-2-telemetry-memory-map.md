# Need for Speed: Underground 2 — Telemetry Memory Map (SLUS-21065, PS2 NTSC-U)

> **Status:** Partially calibrated. Speed and position are empirically verified.
> Velocity and rotation addresses have been found but are not reliable enough for
> use — see the findings section below.

## Verified Fields (in active use)

These are discovered by `python main.py --calibrate` and stored in `saves/calibration.json`.
Addresses are dynamic (heap-allocated per session) — no static pointer has been found.

| Field | Format | Address (example) | Verified by |
|:------|:-------|:-----------------|:------------|
| **Scalar speed** | Float32, m/s | `0x00750390` | Matches in-game speedometer gauge |
| **Position X** | Float32, world units | `0x00750380` | Smooth movement across wall-hit, ~1 unit ≈ 1 m |
| **Position Y** | Float32, world units | `0x00750384` | Monotonically decreasing altitude on downhill road |
| **Position Z** | Float32, world units | `0x00750388` | Consistent small-magnitude lateral drift |

Position moves ~14 m per 0.5 s reading at 100 km/h (expected: ~14 m). ✅

## Unresolved Fields

| Field | Problem |
|:------|:--------|
| **Velocity (XYZ)** | Address passes calibration filters but explodes to ±30 000 after collision. Likely a physics scratch buffer, not a stable velocity vector. |
| **Rotation quaternion** | Calibration finds normalized 4-vectors, but they are degenerate `(cos θ, 0, sin θ, 0)` 2D unit vectors (wheel spin angles), not a 3D body-orientation quaternion. Flips sign every frame during straight driving. |

## Why PC Offsets Don't Apply

The original assumption was that PS2 NFSU2 used the same RenderWare `VehicleInfo`
struct layout as the PC build (0x020 pos, 0x070 vel, 0x090 speed). All 29–40
candidates from the early scanner read exactly `0.000` at those offsets while
driving at 90 km/h — **proving the PS2 EA Black Box build uses a different struct
layout, different allocation container, or a console-specific entity header**.

Gemini confirmed: no verified static pointer or struct layout exists in public
`.pnach` databases for SLUS-21065. The correct approach is empirical differential
scanning (implemented in `--calibrate`).

## Calibration Approach (implemented)

1. **Phase 1a** — Snapshot EE RAM while car is stopped
2. **Phase 1b** — Accelerate straight 3 s, snapshot (find speed: 0→15–35 m/s)
3. **Phase 1c** — Gentle 0.4 s left steer + 1 s settle, snapshot (direction changed)
4. **Phase 2** — Score candidates: quaternion (+3), velocity (+2), position (+2), heap (+1)
   - Speed candidate must stay in range in the turned snapshot (rejects vel_x aliases)
   - Quaternion must be normalized in both snapshots, non-identity, non-degenerate 2D
   - Velocity magnitude must match speed in both moving snapshots (0.7×–1.5×)
   - Position movement must be smooth across all 3 pairs (no teleportation)
5. **Phase 3** — Scan static range for pointer to speed address (none found yet)

## EE RAM Layout Reference

- `0x00000000–0x000FFFFF` — kernel / BIOS (filtered out)
- `0x00100000–0x002FFFFF` — low game data
- `0x00300000–0x007FFFFF` — heap (vehicle structs live here)
- `0x00800000–0x01FFFFFF` — upper heap / audio / streaming

## Axis Convention

Based on observed position movement during straight driving on the highway loop:

- **X** — primary travel axis on the highway savestate (increases while driving forward)
- **Y** — vertical / altitude (decreases slightly on downhill sections)
- **Z** — lateral / minor axis (small magnitude, changes when turning)
