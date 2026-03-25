from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
ISO_DIR = REPO_ROOT / "iso"
SNAP_DIR = REPO_ROOT / "saves" / "snapshots"
CANDIDATES_FILE = SNAP_DIR / "candidates.json"
CALIBRATION_FILE = REPO_ROOT / "saves" / "calibration.json"
MEMCARDS_DIR = REPO_ROOT / "memcards"
SAVESTATES_DIR = REPO_ROOT / "savestates"
PCSX2_BIN = Path("/usr/bin/pcsx2-qt")
PCSX2_CONFIG_DIR = Path.home() / ".config" / "PCSX2"
CONTROLLER_DB_PATH = PCSX2_CONFIG_DIR / "game_controller_db.txt"
ISO_MAP = {
    "nfsu2": ISO_DIR / "Need for Speed - Underground 2 (USA, Canada).iso",
    "nfsmw": ISO_DIR / "Need for Speed - Most Wanted - Black Edition (USA).iso",
}

# Repo-local save state for the NFSU2 highway loop training anchor.
DEFAULT_STATEFILE = SAVESTATES_DIR / "rocm-racer-nfsu2-highway.p2s"

# Save state with HUD/gauge enabled — used during calibration so the
# speedometer is visible for manual verification if needed.
CALIBRATION_STATEFILE = SAVESTATES_DIR / "rocm-racer-nfsu2-highway-calibration.p2s"

# ---------------------------------------------------------------------------
# SDL3 controller database
# ---------------------------------------------------------------------------

def write_controller_db() -> None:
    """Write the SDL3 game controller mapping for the virtual gamepad.

    PCSX2 loads ~/.config/PCSX2/game_controller_db.txt before SDL
    initialises input (via SDL_HINT_GAMECONTROLLERCONFIG_FILE).  Without
    this entry SDL3 will not recognise the uinput device as a gamepad and
    PCSX2 will log no connection message.
    """
    from memory_readers.virtual_gamepad import SDL_MAPPING

    PCSX2_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLLER_DB_PATH.write_text(SDL_MAPPING + "\n")
    print(f"[rocm-racer] Controller DB written to {CONTROLLER_DB_PATH}")

# ---------------------------------------------------------------------------
# PCSX2 process management
# ---------------------------------------------------------------------------


def set_pcsx2_speed_scalar(scalar: float = 1.0, ini_path: Path | None = None) -> None:
    """Patch NominalScalar under [Framerate] in PCSX2.ini."""
    ini = ini_path or (PCSX2_CONFIG_DIR / "inis" / "PCSX2.ini")
    if not ini.exists():
        return
    text = ini.read_text()
    import re
    text = re.sub(
        r"(?m)^(NominalScalar\s*=\s*)[\d.]+",
        rf"\g<1>{scalar}",
        text,
    )
    ini.write_text(text)


def launch_pcsx2(
    iso: Path,
    statefile: Path | None = None,
    turbo: bool = False,
    env_override: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch PCSX2 in headless batch mode.

    Parameters
    ----------
    iso : Path
        Path to the PS2 ISO file.
    statefile : Path, optional
        Savestate to load on startup.
    turbo : bool
        Set NominalScalar=2 in PCSX2.ini for 2× emulation speed.
    env_override : dict, optional
        Extra environment variables (used for multi-instance isolation).
    """
    # Determine which INI to patch (per-instance or global)
    ini_path = None
    if env_override and "XDG_CONFIG_HOME" in env_override:
        ini_path = Path(env_override["XDG_CONFIG_HOME"]) / "PCSX2" / "inis" / "PCSX2.ini"

    if turbo:
        set_pcsx2_speed_scalar(2.0, ini_path=ini_path)

    cmd = [
        str(PCSX2_BIN),
        "-nogui",
        "-batch",
    ]
    if statefile is not None:
        if not statefile.exists():
            print(
                f"[rocm-racer] WARNING: statefile not found: {statefile}\n"
                "  Create it manually inside PCSX2 and see "
                "docs/implementation-notes-for-human.md for instructions.",
                file=sys.stderr,
            )
        cmd += ["-statefile", str(statefile)]
    cmd += ["--", str(iso)]

    proc_env = None
    if env_override:
        proc_env = {**__import__("os").environ, **env_override}

    log_path = REPO_ROOT / "pcsx2.log"
    log_fh = open(log_path, "w")
    print(f"[rocm-racer] Launching: {' '.join(cmd)}")
    print(f"[rocm-racer] PCSX2 output → {log_path}")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=proc_env)


def wait_for_pcsx2_ready(
    timeout: float = 30.0,
    poll: float = 0.25,
    post_ready_delay: float = 2.0,
    emulog: Path | None = None,
) -> None:
    """Block until PCSX2 has loaded the savestate and the game is running.

    Monitors the PCSX2 emulog for the ``Opened gamepad`` marker, which
    confirms that both the savestate has been restored **and** SDL has
    connected the virtual gamepad.  After the marker is found a short
    delay lets the game establish its baseline controller state before
    any input is sent.

    Falls back to a fixed timeout if the emulog is missing or the marker
    never appears (e.g. when running without a virtual gamepad).

    Parameters
    ----------
    emulog : Path, optional
        Custom emulog location (for multi-instance isolation).
        Defaults to ``~/.config/PCSX2/logs/emulog.txt``.
    """
    if emulog is None:
        emulog = Path.home() / ".config" / "PCSX2" / "logs" / "emulog.txt"
    deadline = time.monotonic() + timeout

    print(f"[rocm-racer] Waiting for PCSX2 to finish initialisation (timeout={timeout}s)...")

    while time.monotonic() < deadline:
        try:
            text = emulog.read_text(errors="replace")
            if "Opened gamepad" in text:
                print("[rocm-racer] PCSX2 ready — gamepad connected by SDL.")
                break
        except OSError:
            pass
        time.sleep(poll)
    else:
        print(
            f"[rocm-racer] WARNING: readiness marker not found within {timeout}s — "
            "proceeding anyway."
        )

    # Give the game time to run a few frames and establish its baseline
    # controller state so that the first input we send is a clean
    # released→pressed transition rather than a pre-held button.
    print(f"[rocm-racer] Post-init delay: {post_ready_delay}s for game to settle...")
    time.sleep(post_ready_delay)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rocm-racer — NFS RL training launcher")
    parser.add_argument("--game", choices=list(ISO_MAP), default="nfsu2")
    parser.add_argument("--statefile", type=Path, default=DEFAULT_STATEFILE)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--no-launch", action="store_true",
                        help="Skip launching PCSX2 (assume it is already running)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Automatically find the vehicle struct address (launches PCSX2)")

    # Telemetry modes
    parser.add_argument("--telemetry", action="store_true",
                        help="Read and log live telemetry instead of sending input")
    parser.add_argument("--scan", type=float, default=None, metavar="VALUE",
                        help="Scan EE RAM for a Float32 value to locate the vehicle struct")
    parser.add_argument("--scan-tolerance", type=float, default=0.5,
                        help="Tolerance for --scan matches (default: ±0.5)")
    parser.add_argument("--vehicle-addr", type=lambda x: int(x, 0), default=None,
                        metavar="0xADDR",
                        help="PS2-side vehicle struct address (hex)")

    # Differential scan (Cheat-Engine-style)
    parser.add_argument("--snap", metavar="NAME",
                        help="Save an EE RAM snapshot for later diffing")
    parser.add_argument("--scan-diff", metavar="NAME",
                        help="Diff live EE RAM against saved snapshot NAME")
    parser.add_argument("--filter",
                        choices=["changed", "unchanged", "increased", "decreased"],
                        default="changed",
                        help="Filter for --scan-diff (default: changed)")
    parser.add_argument("--scan-reset", action="store_true",
                        help="Clear the candidate list and start fresh")
    parser.add_argument("--vision", action="store_true",
                        help="Test frame capture pipeline (PipeWire portal, saves sample frame)")
    parser.add_argument("--train", action="store_true",
                        help="Run PPO training loop (requires calibration)")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Save a model checkpoint every N timesteps (default: 10000)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device for training (default: cuda)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the live OpenCV preview window during training")
    parser.add_argument("--turbo", action="store_true",
                        help="Run PCSX2 at 2× speed to accelerate training")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel PCSX2 environments (default: 1)")
    parser.add_argument("--load-model", type=Path, default=None, metavar="PATH",
                        help="Load a saved model (.zip) and resume training from it")
    parser.add_argument("--setup-savestates", action="store_true",
                        help="Load each highway-N.p2s file into PINE slots 0–9 for multi-start training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso = ISO_MAP[args.game]

    # --- Calibrate mode: automated vehicle struct discovery ---
    if args.calibrate:
        _run_calibrate(args, iso)
        return

    # --- Scan reset: clear candidates ---
    if args.scan_reset:
        _run_scan_reset()
        return

    # --- Snapshot mode: dump EE RAM ---
    if args.snap is not None:
        _run_snap(args)
        return

    # --- Differential scan mode ---
    if args.scan_diff is not None:
        _run_scan_diff(args)
        return

    # --- Scan mode: no gamepad needed, just find PCSX2 and scan EE RAM ---
    if args.scan is not None:
        _run_scan(args)
        return

    # --- Telemetry mode: no gamepad needed, just read and log ---
    if args.telemetry:
        _run_telemetry(args, iso)
        return

    # --- Vision test mode: capture frames from PCSX2 ---
    if args.vision:
        _run_vision(args, iso)
        return

    # --- Training mode: PPO ---
    if args.train:
        _run_train(args, iso)
        return

    # --- Setup savestates: load .p2s files into PINE slots ---
    if args.setup_savestates:
        _run_setup_savestates(args, iso)
        return

    # --- Default: test mode (gamepad accelerate) ---
    _run_test(args, iso)


# ---------------------------------------------------------------------------
# Mode: --calibrate
# ---------------------------------------------------------------------------

def _run_calibrate(args: argparse.Namespace, iso: Path) -> None:
    """Automated vehicle struct discovery via differential scanning + quaternion anchoring.

    Phase 1: Differential scan — find Float32 addresses that were ~0.0 when
             stopped and are now in a speed-like range (15–35 m/s or 75–110 km/h).
    Phase 2: Score all candidates — for each speed candidate, search for quaternion,
             velocity triplets, and position triplets nearby.  Pick the best.
    Phase 3: Static pointer scan — find a pointer to the struct in static
             data range (0x003X–0x005X) for reload-safe calibration.
    """
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets
    from memory_readers.virtual_gamepad import VirtualGamepad
    from evdev import ecodes as e

    ACCEL_TIME = 3.0
    STEER_TIME = 0.4   # gentle left nudge for 3rd snapshot

    print("[rocm-racer] ══════════════════════════════════════════════════════════")
    print("[rocm-racer]  Calibration — differential scan + 3-snapshot verification")
    print("[rocm-racer] ══════════════════════════════════════════════════════════")

    write_controller_db()
    gamepad = VirtualGamepad()
    gamepad.open()
    statefile = args.statefile if args.statefile != DEFAULT_STATEFILE else CALIBRATION_STATEFILE
    pcsx2_proc = launch_pcsx2(iso, statefile=statefile)
    wait_for_pcsx2_ready()

    offsets = TelemetryOffsets(vehicle_struct_addr=1)  # dummy to bypass open() guard
    reader = NFSU2MemoryReader(offsets=offsets)
    reader.open()

    try:
        # ── Phase 1: differential scan (stopped → straight → turning) ──
        print("\n[calibrate] Phase 1a: snapshot while stopped (car must be stationary)...")
        snap_stopped = reader.snapshot_ee_ram()

        print(f"[calibrate] Phase 1b: accelerating straight for {ACCEL_TIME}s...")
        gamepad.hold_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
        time.sleep(ACCEL_TIME)
        snap_straight = reader.snapshot_ee_ram()

        print(f"[calibrate] Phase 1c: gentle left steer ({STEER_TIME}s) + continued accel...")
        gamepad.send(steering=-1.0, throttle=1.0, brake=0.0)
        time.sleep(STEER_TIME)
        gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
        # Let the car settle for a moment after the steer input
        time.sleep(1.0)
        snap_turned = reader.snapshot_ee_ram()
        gamepad.release_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

        speed_candidates = _phase1_find_speed_candidates(snap_stopped, snap_straight)
        print(f"[calibrate] Phase 1: {len(speed_candidates):,} speed-float candidates "
              f"(addresses ≥ 0x00100000)")

        if not speed_candidates:
            print("[calibrate] ERROR: No speed candidates found.")
            print("[calibrate]   Make sure the car is fully stopped before the scan,")
            print("              then reaches >60 km/h during acceleration.")
            return

        # ── Phase 2: score ALL candidates (quaternion + velocity + position) ──
        print(f"\n[calibrate] Phase 2: scoring all candidates "
              f"(quaternion + velocity + position search)...")
        scored: list[dict] = []

        for speed_addr, speed_val, speed_unit in speed_candidates:
            score = 0
            result: dict = {
                "speed_addr": speed_addr,
                "speed_val": speed_val,
                "speed_unit": speed_unit,
                "quat_addr": None,
                "quat_sq": None,
                "vel_triplet": None,
                "pos_triplet": None,
            }

            # Check the speed value in the turned snapshot — true scalar speed
            # should still be in a reasonable range (not drop because direction changed)
            f32_turned = np.frombuffer(snap_turned, dtype=np.float32)
            turned_idx = speed_addr // 4
            if turned_idx < len(f32_turned):
                turned_val = float(f32_turned[turned_idx])
                # If it's truly scalar speed, it should still be > 10 m/s or > 40 km/h
                # after a gentle steer (the car is still accelerating)
                if speed_unit == "m/s":
                    if not (10.0 <= turned_val <= 50.0):
                        continue  # not scalar speed — dropped after direction change
                else:
                    if not (40.0 <= turned_val <= 180.0):
                        continue

            # Quaternion search (±0x400) — needs both snapshots to reject identity quats
            quats = _phase2_quaternion_search(snap_stopped, snap_straight, speed_addr)
            if quats:
                result["quat_addr"] = quats[0][0]
                result["quat_sq"]   = quats[0][1]
                score += 3

            # Velocity + position triplet search (±0x200 around quaternion or speed)
            # Uses all 3 snapshots for validation
            anchor = result["quat_addr"] if result["quat_addr"] is not None else speed_addr
            offsets_info = _phase3_discover_struct_offsets(
                snap_stopped, snap_straight, snap_turned,
                speed_addr, speed_val, anchor, window=0x200
            )
            if offsets_info["vel_triplets"]:
                result["vel_triplet"] = offsets_info["vel_triplets"][0]
                score += 2
            if offsets_info["pos_triplets"]:
                result["pos_triplet"] = offsets_info["pos_triplets"][0]
                score += 2

            # Bonus: address in game heap region (0x003X+)
            if speed_addr >= 0x00300000:
                score += 1

            result["score"] = score
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Display top 15
        print(f"\n[calibrate]   ── Top 15 candidates by structural score ──")
        print(f"  {'Score':>5}  {'Speed Addr':>12}  {'Value':>8}  {'Unit':>4}  "
              f"{'Quat':>12}  {'Vel':>5}  {'Pos':>5}")
        print(f"  {'─'*5}  {'─'*12}  {'─'*8}  {'─'*4}  {'─'*12}  {'─'*5}  {'─'*5}")
        for r in scored[:15]:
            quat_str = f"0x{r['quat_addr']:08X}" if r['quat_addr'] else "—"
            vel_str  = "✓" if r['vel_triplet'] else "—"
            pos_str  = "✓" if r['pos_triplet'] else "—"
            print(f"  {r['score']:5d}  0x{r['speed_addr']:08X}  "
                  f"{r['speed_val']:8.3f}  {r['speed_unit']:>4}  "
                  f"{quat_str:>12}  {vel_str:>5}  {pos_str:>5}")

        best = scored[0]
        if best["score"] < 3:
            print(f"\n[calibrate] ⚠  No high-confidence match (best score: {best['score']}).")
            print("[calibrate]   Try a longer acceleration or a different savestate.")
            return

        best_speed_addr = best["speed_addr"]
        best_speed_val  = best["speed_val"]
        best_speed_unit = best["speed_unit"]
        best_quat_addr  = best["quat_addr"]

        print(f"\n[calibrate] ✓ Best candidate: speed @ 0x{best_speed_addr:08X} "
              f"({best_speed_val:.2f} {best_speed_unit}, score {best['score']})")

        if best_quat_addr is not None:
            print(f"[calibrate]   Quaternion @ 0x{best_quat_addr:08X} "
                  f"(Σ²={best['quat_sq']:.5f})")
        if best["vel_triplet"]:
            vt = best["vel_triplet"]
            print(f"[calibrate]   Velocity @ 0x{vt[0]:08X}, 0x{vt[1]:08X}, 0x{vt[2]:08X}")
        if best["pos_triplet"]:
            pt = best["pos_triplet"]
            print(f"[calibrate]   Position @ 0x{pt[0]:08X}, 0x{pt[1]:08X}, 0x{pt[2]:08X}")

        # ── Phase 3: static pointer discovery ──
        print(f"\n[calibrate] Phase 3: searching for static pointer to 0x{best_speed_addr:08X}...")
        static_ptrs = _phase4_find_static_pointers(snap_straight, best_speed_addr)

        static_ptr_addr = 0
        if static_ptrs:
            static_ptr_addr = static_ptrs[0]
            print(f"[calibrate]   ✓ Found {len(static_ptrs)} pointer(s):")
            for ptr in static_ptrs[:5]:
                print(f"      0x{ptr:08X}")
        else:
            print("[calibrate]   ⚠  No static pointer found. Direct address saved.")

        # ── Save calibration.json (absolute addresses) ──
        cal_data: dict = {
            "speed_addr":         f"0x{best_speed_addr:08X}",
            "speed_unit":          best_speed_unit,
            "speed_value_sample":  best_speed_val,
        }
        if static_ptr_addr:
            cal_data["static_pointer_addr"] = f"0x{static_ptr_addr:08X}"

        if best_quat_addr is not None:
            cal_data["quat_addr"] = f"0x{best_quat_addr:08X}"
            # Store individual rotation component addresses
            for i, label in enumerate(("rot_x_addr", "rot_y_addr", "rot_z_addr", "rot_w_addr")):
                cal_data[label] = f"0x{best_quat_addr + i * 4:08X}"

        if best["vel_triplet"]:
            vt = best["vel_triplet"]
            cal_data["vel_x_addr"] = f"0x{vt[0]:08X}"
            cal_data["vel_y_addr"] = f"0x{vt[1]:08X}"
            cal_data["vel_z_addr"] = f"0x{vt[2]:08X}"

        if best["pos_triplet"]:
            pt = best["pos_triplet"]
            cal_data["pos_x_addr"] = f"0x{pt[0]:08X}"
            cal_data["pos_y_addr"] = f"0x{pt[1]:08X}"
            cal_data["pos_z_addr"] = f"0x{pt[2]:08X}"

        cal_data["calibration_score"] = best["score"]

        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(cal_data, f, indent=2)
        print(f"\n[calibrate] ✓ Saved to {CALIBRATION_FILE}")
        print(f"[calibrate]   Run `python main.py --telemetry` to verify.")

    finally:
        reader.close()
        gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
            print("[rocm-racer] PCSX2 terminated.")


# ---------------------------------------------------------------------------
# Calibration phase helpers
# ---------------------------------------------------------------------------

def _phase1_find_speed_candidates(
    data_stopped: bytes,
    data_moving: bytes,
    min_addr: int = 0x00100000,
) -> list[tuple[int, float, str]]:
    """Phase 1: find speed float via differential scan.

    Finds Float32 addresses that were ~0.0 when stopped and are now in
    either m/s range (15–35) or km/h range (75–110) while moving.
    Excludes addresses below min_addr (kernel/BIOS area).

    Returns list of (ps2_addr, value_moving, unit_hint) tuples.
    """
    f32_s = np.frombuffer(data_stopped, dtype=np.float32)
    f32_m = np.frombuffer(data_moving, dtype=np.float32)

    near_zero    = np.isfinite(f32_s) & (np.abs(f32_s) < 0.5)
    in_ms_range  = np.isfinite(f32_m) & (f32_m >= 15.0) & (f32_m <= 35.0)
    in_kph_range = np.isfinite(f32_m) & (f32_m >= 60.0) & (f32_m <= 110.0)

    indices = np.where(near_zero & (in_ms_range | in_kph_range))[0]
    # Filter out kernel/BIOS area
    indices = indices[indices * 4 >= min_addr]

    results: list[tuple[int, float, str]] = []
    for idx in indices:
        addr = int(idx) * 4
        val  = float(f32_m[idx])
        unit = "m/s" if 15.0 <= val <= 35.0 else "km/h"
        results.append((addr, val, unit))

    return results


def _phase2_quaternion_search(
    data_stopped: bytes,
    data_moving: bytes,
    speed_addr: int,
    window: int = 0x400,
) -> list[tuple[int, float]]:
    """Phase 2: find normalized quaternion in ±window bytes around speed_addr.

    A quaternion is 4 adjacent Float32s where sum of squares ≈ 1.0 in BOTH
    snapshots.  Additional filters:
    - Rejects identity quaternions (1,0,0,0)
    - Rejects degenerate 2D unit vectors where 2+ components are always ~0
      (these are typically wheel spin angles, not car body orientation)
    - A proper 3D orientation quaternion must have at least 3 non-zero
      components (e.g. yaw-only: (0, sin(θ/2), 0, cos(θ/2)) has 2 non-zero)

    Uses float64 arithmetic to avoid overflow on large garbage values.
    Returns list of (quat_base_addr, quat_sq) sorted by quality score.
    """
    import math
    from memory_readers.nfsu2_memory import _EE_RAM_SIZE

    start = max(0, speed_addr - window) & ~3
    end   = min(_EE_RAM_SIZE - 16, speed_addr + window)

    f32_s = np.frombuffer(data_stopped, dtype=np.float32)
    f32_m = np.frombuffer(data_moving, dtype=np.float32)
    results: list[tuple[int, float]] = []

    for byte_off in range(start, end - 12, 4):
        idx = byte_off // 4
        if idx + 3 >= len(f32_s) or idx + 3 >= len(f32_m):
            break

        vals_s = f32_s[idx:idx + 4].astype(np.float64)
        vals_m = f32_m[idx:idx + 4].astype(np.float64)

        if not (np.all(np.isfinite(vals_s)) and np.all(np.isfinite(vals_m))):
            continue

        sq_s = float(np.sum(vals_s * vals_s))
        sq_m = float(np.sum(vals_m * vals_m))

        # Must be normalized in BOTH snapshots
        if not (0.98 < sq_s < 1.02 and 0.98 < sq_m < 1.02):
            continue

        # Reject identity quaternion (1,0,0,0) or (0,0,0,1) that stays constant
        is_identity_s = (abs(abs(vals_s[0]) - 1.0) < 0.01 and
                         abs(vals_s[1]) < 0.01 and abs(vals_s[2]) < 0.01 and
                         abs(vals_s[3]) < 0.01)
        is_identity_s |= (abs(vals_s[0]) < 0.01 and abs(vals_s[1]) < 0.01 and
                          abs(vals_s[2]) < 0.01 and abs(abs(vals_s[3]) - 1.0) < 0.01)
        is_identity_m = (abs(abs(vals_m[0]) - 1.0) < 0.01 and
                         abs(vals_m[1]) < 0.01 and abs(vals_m[2]) < 0.01 and
                         abs(vals_m[3]) < 0.01)
        is_identity_m |= (abs(vals_m[0]) < 0.01 and abs(vals_m[1]) < 0.01 and
                          abs(vals_m[2]) < 0.01 and abs(abs(vals_m[3]) - 1.0) < 0.01)
        if is_identity_s and is_identity_m:
            continue

        # Reject degenerate "2D unit vectors" — if 2+ components are ~0 in
        # BOTH snapshots, this is not a proper 3D orientation quaternion.
        # Real car heading quats have at least 2 non-trivial components
        # (e.g. yaw-only: (0, sin(θ/2), 0, cos(θ/2))).
        # Wheel spin angles often appear as (cos(θ), 0, sin(θ), 0).
        near_zero_s = sum(1 for v in vals_s if abs(v) < 0.01)
        near_zero_m = sum(1 for v in vals_m if abs(v) < 0.01)
        if near_zero_s >= 2 and near_zero_m >= 2:
            # Check if the SAME components are zero — degenerate 2D rotation
            zero_mask_s = [abs(v) < 0.01 for v in vals_s]
            zero_mask_m = [abs(v) < 0.01 for v in vals_m]
            if zero_mask_s == zero_mask_m:
                continue

        # Prefer quaternions that changed between snapshots (car rotated)
        # but not TOO much — driving straight should cause small heading drift,
        # not wild oscillation (which indicates wheel spin or other periodic values)
        delta = float(np.sum((vals_m - vals_s) ** 2))
        # Score: closer to 1.0 norm is better, small change preferred over huge change
        quality = abs(sq_m - 1.0) + abs(sq_s - 1.0) - min(delta, 0.05) * 5
        results.append((byte_off, sq_m, quality))

    # Sort by quality (lower is better)
    results.sort(key=lambda x: x[2])
    # Return in (addr, sq) format
    return [(addr, sq) for addr, sq, _ in results]


def _phase3_discover_struct_offsets(
    data_stopped: bytes,
    data_straight: bytes,
    data_turned: bytes,
    speed_addr: int,
    speed_val: float,
    quat_addr: int,
    window: int = 0x200,
) -> dict:
    """Phase 3: empirically discover position and velocity offsets using 3 snapshots.

    Uses stopped/straight/turned snapshots to validate candidates:
    - Velocity: 3 consecutive floats near-zero when stopped, with magnitude
      roughly matching speed_val in BOTH moving snapshots.  Each individual
      component must not exceed scalar speed (impossible for a projection).
      Velocity triplet must NOT overlap with speed_addr.
    - Position: 3 consecutive floats with world-scale values, where movement
      is smooth between all snapshot pairs (no teleportation jumps).

    Returns dict with vel_triplets, pos_triplets.
    """
    import math
    from memory_readers.nfsu2_memory import _EE_RAM_SIZE

    f32_s = np.frombuffer(data_stopped, dtype=np.float32)
    f32_m = np.frombuffer(data_straight, dtype=np.float32)
    f32_t = np.frombuffer(data_turned, dtype=np.float32)

    start = max(0, quat_addr - window) & ~3
    end   = min(_EE_RAM_SIZE - 4, quat_addr + window)

    # Get the speed value in the turned snapshot for velocity validation
    speed_idx = speed_addr // 4
    speed_turned = float(f32_t[speed_idx]) if speed_idx < len(f32_t) else speed_val

    vel_candidates: list[tuple[int, float, float, float]] = []  # addr, stopped, straight, turned
    pos_candidates: list[tuple[int, float, float, float]] = []

    for byte_off in range(start, end, 4):
        idx = byte_off // 4
        if idx >= len(f32_s) or idx >= len(f32_m) or idx >= len(f32_t):
            break
        vs = float(f32_s[idx])
        vm = float(f32_m[idx])
        vt = float(f32_t[idx])
        if not (math.isfinite(vs) and math.isfinite(vm) and math.isfinite(vt)):
            continue

        # Velocity candidate: near 0 stopped, meaningful while moving in BOTH snapshots
        if abs(vs) < 0.1 and abs(vm) > 0.5 and abs(vt) > 0.5:
            # Each component must not exceed scalar speed * 1.2
            if abs(vm) <= speed_val * 1.2 and abs(vt) <= speed_turned * 1.2:
                vel_candidates.append((byte_off, vs, vm, vt))

        # Position candidate: world-scale, both finite, changed but not teleported
        if 1.0 < abs(vs) < 100_000 and 1.0 < abs(vm) < 100_000 and 1.0 < abs(vt) < 100_000:
            delta_sm = abs(vm - vs)
            delta_mt = abs(vt - vm)
            # Both deltas should be < 500m (reasonable for 3s + 1.2s of driving)
            if delta_sm < 500 and delta_mt < 500:
                pos_candidates.append((byte_off, vs, vm, vt))

    def _triplets(addrs: list[int]) -> list[tuple[int, int, int]]:
        out = []
        for i in range(len(addrs) - 2):
            a, b, c = addrs[i], addrs[i+1], addrs[i+2]
            if b == a + 4 and c == b + 4:
                out.append((a, b, c))
        return out

    # ---- Velocity triplet validation ----
    vel_triplets_raw = _triplets(sorted(a for a, _, _, _ in vel_candidates))
    vel_lookup = {a: (vs, vm, vt) for a, vs, vm, vt in vel_candidates}
    vel_triplets_scored: list[tuple[tuple[int, int, int], float]] = []

    for tri in vel_triplets_raw:
        # Reject triplets overlapping with speed_addr
        if any(addr == speed_addr for addr in tri):
            continue

        # Get values in straight snapshot
        vals_m = [vel_lookup.get(a, (0, 0, 0))[1] for a in tri]
        mag_m = math.sqrt(sum(v*v for v in vals_m))

        # Get values in turned snapshot
        vals_t = [vel_lookup.get(a, (0, 0, 0))[2] for a in tri]
        mag_t = math.sqrt(sum(v*v for v in vals_t))

        # Magnitude should be close to scalar speed in BOTH moving snapshots
        if speed_val > 0 and speed_turned > 0:
            ratio_m = mag_m / speed_val
            ratio_t = mag_t / speed_turned
            # Tighter tolerance: 0.7x–1.5x of scalar speed
            if 0.7 < ratio_m < 1.5 and 0.7 < ratio_t < 1.5:
                # Score: prefer close to 1.0 in both
                score = abs(ratio_m - 1.0) + abs(ratio_t - 1.0)
                vel_triplets_scored.append((tri, score))

    vel_triplets_scored.sort(key=lambda x: x[1])
    vel_triplets = [t for t, _ in vel_triplets_scored]

    # ---- Position triplet validation ----
    pos_triplets_raw = _triplets(sorted(a for a, _, _, _ in pos_candidates))
    pos_lookup = {a: (vs, vm, vt) for a, vs, vm, vt in pos_candidates}
    pos_triplets_scored: list[tuple[tuple[int, int, int], float]] = []

    for tri in pos_triplets_raw:
        # Compute 3D displacement stopped→straight and straight→turned
        deltas_sm = []
        deltas_mt = []
        for addr in tri:
            vs, vm, vt = pos_lookup.get(addr, (0, 0, 0))
            deltas_sm.append(vm - vs)
            deltas_mt.append(vt - vm)
        dist_sm = math.sqrt(sum(d*d for d in deltas_sm))
        dist_mt = math.sqrt(sum(d*d for d in deltas_mt))

        changed_sm = sum(1 for d in deltas_sm if abs(d) > 0.5)
        changed_mt = sum(1 for d in deltas_mt if abs(d) > 0.5)
        if changed_sm < 2:
            continue

        # Both displacements should be moderate (1–300m), not teleportation
        if not (1.0 < dist_sm < 300 and 0.5 < dist_mt < 200):
            continue

        # The ratio of displacement to expected travel should be reasonable
        # At ~20 m/s for 3s → ~60m, then ~25 m/s for 1.2s → ~30m
        expected_sm = speed_val * 3.0  # rough expected distance
        ratio_sm = dist_sm / expected_sm if expected_sm > 1 else 999
        if not (0.3 < ratio_sm < 3.0):
            continue

        # Score: prefer ratios close to 1.0 and movement in both phases
        score = abs(ratio_sm - 1.0)
        pos_triplets_scored.append((tri, score))

    pos_triplets_scored.sort(key=lambda x: x[1])
    pos_triplets = [t for t, _ in pos_triplets_scored]

    return {
        "vel_triplets": vel_triplets,
        "pos_triplets": pos_triplets,
    }


def _phase4_find_static_pointers(data: bytes, vehicle_base: int) -> list[int]:
    """Phase 4: find static pointers to the vehicle struct.

    Searches all 32 MB for uint32 values matching the vehicle base address
    (with and without kseg mirror bits).  Returns addresses in the static
    data range (0x003XXXXX-0x005XXXXX), sorted by address.
    """
    u32 = np.frombuffer(data, dtype=np.uint32)

    targets = np.array([
        vehicle_base,
        vehicle_base | 0x80000000,   # kseg0
        vehicle_base | 0xA0000000,   # kseg1
    ], dtype=np.uint32)

    pointers: list[int] = []
    for target in targets:
        matches = np.where(u32 == target)[0]
        for idx in matches:
            byte_addr = int(idx) * 4
            if 0x00300000 <= byte_addr <= 0x005FFFFF:
                pointers.append(byte_addr)

    pointers.sort()
    return pointers


# ---------------------------------------------------------------------------
# Mode: --scan
# ---------------------------------------------------------------------------

def _run_scan(args: argparse.Namespace) -> None:
    """Search 32 MB EE RAM for a Float32 value to help locate the vehicle struct."""
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets

    target = args.scan
    tol = args.scan_tolerance
    print(f"[rocm-racer] Scanning EE RAM for Float32 ≈ {target} (±{tol})...")

    offsets = TelemetryOffsets(vehicle_struct_addr=1)  # dummy nonzero to avoid guard
    reader = NFSU2MemoryReader(offsets=offsets)
    reader.open()
    matches = reader.scan_ee_ram(target, tolerance=tol)
    reader.close()

    if not matches:
        print("[rocm-racer] No matches found. Try adjusting the value or tolerance.")
        return

    print(f"\n[rocm-racer] Found {len(matches)} matches:\n")
    print(f"  {'PS2 Address':>14s}    {'If speed (+0x090)':>16s}    Value")
    print(f"  {'─' * 14}    {'─' * 16}    {'─' * 10}")

    # Re-read to show actual values
    reader.open()
    for ps2_addr in matches[:50]:  # cap output at 50
        raw = reader._read_f32(ps2_addr)
        struct_base = ps2_addr - 0x090
        print(f"  0x{ps2_addr:08X}    0x{struct_base:08X}    {raw:.4f}")
    reader.close()

    if len(matches) > 50:
        print(f"\n  ... and {len(matches) - 50} more (narrow the tolerance)")

    print(
        f"\n[rocm-racer] If you searched for absolute speed (m/s), the vehicle struct"
        f"\n  base is at PS2 address = match - 0x090."
        f"\n  Verify with:  python main.py --vehicle-addr 0x<ADDR> --telemetry --no-launch"
    )


# ---------------------------------------------------------------------------
# Mode: --snap
# ---------------------------------------------------------------------------

def _run_snap(args: argparse.Namespace) -> None:
    """Save a named snapshot of the 32 MB EE RAM for later diffing."""
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = SNAP_DIR / f"{args.snap}.bin"

    offsets = TelemetryOffsets(vehicle_struct_addr=1)  # dummy
    reader = NFSU2MemoryReader(offsets=offsets)
    reader.open()
    data = reader.snapshot_ee_ram()
    reader.close()

    snap_path.write_bytes(data)
    print(f"[rocm-racer] Snapshot saved: {snap_path}  ({len(data):,} bytes)")


# ---------------------------------------------------------------------------
# Mode: --scan-diff
# ---------------------------------------------------------------------------

def _run_scan_diff(args: argparse.Namespace) -> None:
    """Diff live EE RAM against a saved snapshot (Cheat-Engine-style narrowing)."""
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets

    snap_path = SNAP_DIR / f"{args.scan_diff}.bin"
    if not snap_path.exists():
        print(
            f"[rocm-racer] ERROR: snapshot '{args.scan_diff}' not found at {snap_path}\n"
            f"  Take one first:  python main.py --snap {args.scan_diff} --no-launch",
            file=sys.stderr,
        )
        sys.exit(1)

    old_data = snap_path.read_bytes()
    filter_mode = args.filter

    # Load previous candidates (if any)
    candidates: list[int] | None = None
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE, "r") as f:
            candidates = json.load(f)
        print(f"[rocm-racer] Loaded {len(candidates):,} candidates from previous scan.")
    else:
        print("[rocm-racer] No previous candidates — scanning all ~8 M float slots.")

    # Read live EE RAM
    offsets = TelemetryOffsets(vehicle_struct_addr=1)  # dummy
    reader = NFSU2MemoryReader(offsets=offsets)
    reader.open()
    new_data = reader.snapshot_ee_ram()
    reader.close()

    # Diff
    results = NFSU2MemoryReader.diff_scan(old_data, new_data, filter_mode, candidates)

    # Save surviving candidates
    surviving = [addr for addr, _, _ in results]
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(surviving, f)

    prev_count = len(candidates) if candidates else "8,388,608"
    print(
        f"[rocm-racer] Filter '{filter_mode}': {prev_count} → {len(surviving):,} candidates"
    )

    # Show results if manageable
    if len(surviving) <= 100:
        print(f"\n  {'PS2 Address':>14s}    {'Old Value':>12s}    {'New Value':>12s}")
        print(f"  {'─' * 14}    {'─' * 12}    {'─' * 12}")
        for ps2_addr, old_val, new_val in results:
            print(f"  0x{ps2_addr:08X}    {old_val:12.4f}    {new_val:12.4f}")
    else:
        # Show first 20 as a preview
        print(f"\n  Showing first 20 of {len(surviving):,}:")
        print(f"  {'PS2 Address':>14s}    {'Old Value':>12s}    {'New Value':>12s}")
        print(f"  {'─' * 14}    {'─' * 12}    {'─' * 12}")
        for ps2_addr, old_val, new_val in results[:20]:
            print(f"  0x{ps2_addr:08X}    {old_val:12.4f}    {new_val:12.4f}")
        print(f"\n  Narrow further with another --scan-diff or --scan-reset to restart.")

    print(f"\n[rocm-racer] Candidates saved to {CANDIDATES_FILE}")


# ---------------------------------------------------------------------------
# Mode: --scan-reset
# ---------------------------------------------------------------------------

def _run_scan_reset() -> None:
    """Clear the candidate list to start a fresh differential scan."""
    if CANDIDATES_FILE.exists():
        CANDIDATES_FILE.unlink()
        print("[rocm-racer] Candidate list cleared. Next --scan-diff will scan all addresses.")
    else:
        print("[rocm-racer] No candidate list to clear.")


# ---------------------------------------------------------------------------
# Mode: --telemetry
# ---------------------------------------------------------------------------

def _run_telemetry(args: argparse.Namespace, iso: Path) -> None:
    """Read and log live telemetry from a running PCSX2 instance."""
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets

    pcsx2_proc: subprocess.Popen | None = None

    # The reader auto-loads calibration.json in open()
    reader = NFSU2MemoryReader()

    if not args.no_launch:
        write_controller_db()
        from memory_readers.virtual_gamepad import VirtualGamepad
        gamepad = VirtualGamepad()
        gamepad.open()
        # Use calibration savestate (with gauge visible) so the speedometer
        # can be visually compared against telemetry readings.
        statefile = args.statefile if args.statefile != DEFAULT_STATEFILE else CALIBRATION_STATEFILE
        pcsx2_proc = launch_pcsx2(iso, statefile=statefile)
        wait_for_pcsx2_ready()

    reader.open()

    if not reader.is_calibrated():
        print(
            "[rocm-racer] ERROR: No calibration data found.\n"
            "  Run:  python main.py --calibrate",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[rocm-racer] Telemetry logging started (speed @ 0x{reader._speed_addr:08X})")
    print("[rocm-racer] Accelerating (Cross). Press Ctrl-C to stop.\n")

    # Hold accelerate so the car moves and telemetry shows live data
    if not args.no_launch:
        from evdev import ecodes as e
        gamepad.hold_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=1.0, brake=0.0)

    try:
        while True:
            try:
                sample = reader.read_telemetry()
                print(f"  {sample.fmt()}")
            except RuntimeError as exc:
                print(f"  [read error] {exc}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_launch:
            try:
                gamepad.release_button(e.BTN_SOUTH)
            except Exception:
                pass
            gamepad.close()
        reader.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()


# ---------------------------------------------------------------------------
# Mode: --vision
# ---------------------------------------------------------------------------

def _run_vision(args: argparse.Namespace, iso: Path) -> None:
    """
    Test the frame-capture pipeline against a running PCSX2 instance.

    Launches PCSX2 with the calibration savestate, opens the PipeWire
    portal (shows consent dialog once), then captures frames in a loop
    while printing FPS stats.  Saves the first valid frame to
    saves/vision_sample.png for visual inspection.
    """
    from memory_readers.frame_capture import FrameCapture, FrameCaptureConfig

    pcsx2_proc: subprocess.Popen | None = None

    if not args.no_launch:
        write_controller_db()
        from memory_readers.virtual_gamepad import VirtualGamepad
        gamepad = VirtualGamepad()
        gamepad.open()
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    cfg = FrameCaptureConfig()
    fc = FrameCapture(cfg)

    print(f"[vision] Locating PCSX2 window...")
    fc.open()
    print(f"[vision] Capture ready. obs shape: {fc.observation_shape}. Press Ctrl-C to stop.\n")

    sample_path = Path("saves/vision_sample.png")
    sample_saved = False
    frame_count = 0
    t_start = time.monotonic()

    try:
        while True:
            obs = fc.step()          # (N, H, W) uint8
            frame_count += 1

            if not sample_saved:
                # Save the most-recent single frame for inspection
                from PIL import Image as PILImage
                PILImage.fromarray(obs[-1]).save(sample_path)
                print(f"[vision] Sample frame saved → {sample_path}")
                sample_saved = True

            elapsed = time.monotonic() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                f"\r[vision] frames={frame_count:5d}  fps={fps:5.1f}  "
                f"obs={obs.shape} dtype={obs.dtype}",
                end="",
                flush=True,
            )
            time.sleep(0.1)

    except KeyboardInterrupt:
        print()
    finally:
        fc.close()
        if not args.no_launch:
            gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()


# ---------------------------------------------------------------------------
# Multi-savestate helpers
# ---------------------------------------------------------------------------

def _discover_savestate_files() -> list[tuple[int, Path]]:
    """Find highway savestate files matching rocm-racer-nfsu2-highway-N.p2s (N=0..9).

    Returns a list of (slot, path) tuples sorted by slot.  The file suffix
    matches the PINE slot directly: highway-0.p2s → slot 0, etc.
    """
    found: list[tuple[int, Path]] = []
    for slot in range(0, 10):
        p = SAVESTATES_DIR / f"rocm-racer-nfsu2-highway-{slot}.p2s"
        if p.exists():
            found.append((slot, p))
    return found


def _run_setup_savestates(args: argparse.Namespace, iso: Path) -> None:
    """One-time setup: load each highway-N.p2s file into PINE slot N.

    PINE's load_state(slot) only works with in-emulator numbered slots, not
    arbitrary .p2s files on disk.  This mode sequentially boots PCSX2 with
    each .p2s file and saves it into the matching PINE slot.

    Slot mapping (filename suffix = PINE slot):
      highway-0.p2s → slot 0
      highway-1.p2s → slot 1
      ...
      highway-9.p2s → slot 9

    After this, ``--train`` can use slots 0–9 for randomised episode starts.
    """
    from memory_readers.virtual_gamepad import VirtualGamepad

    saves = _discover_savestate_files()
    if not saves:
        print("[setup-savestates] No savestates found.")
        print(f"  Place files named rocm-racer-nfsu2-highway-1.p2s … "
              f"rocm-racer-nfsu2-highway-9.p2s in {SAVESTATES_DIR}")
        return

    write_controller_db()
    gamepad = VirtualGamepad()
    gamepad.open()

    from memory_readers.nfsu2_memory import NFSU2MemoryReader

    for slot, save_path in saves:
        print(f"[setup-savestates] Loading {save_path.name} into slot {slot}...")
        proc = launch_pcsx2(iso, statefile=save_path, turbo=False)
        wait_for_pcsx2_ready()
        reader = NFSU2MemoryReader()
        reader.open()
        reader.save_state(slot)
        time.sleep(0.5)
        print(f"[setup-savestates] Slot {slot} saved.")
        reader.close()
        proc.terminate()
        proc.wait(timeout=5)
        time.sleep(1.0)

    gamepad.close()

    slots = [s for s, _ in saves]
    print(f"\n[setup-savestates] Done — {len(saves)} savestates loaded into slots {slots}")
    print(f"[setup-savestates] Run training with:  python main.py --train")


# ---------------------------------------------------------------------------
# Mode: --train
# ---------------------------------------------------------------------------

def _run_train(args: argparse.Namespace, iso: Path) -> None:
    """
    PPO training loop — single or multi-environment.

    When ``--num-envs 1`` (default for backward compat), launches a single
    PCSX2 instance with DummyVecEnv.  When N > 1, uses InstanceManager for
    isolated PCSX2 instances and ThreadedVecEnv for parallel stepping.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.feature_extractor import MultimodalExtractor
    from agents.training_monitor import TrainingMonitorCallback
    from environments.pcsx2_env import PCSX2EnvConfig, PCSX2RacerEnv
    from memory_readers.frame_capture import FrameCapture, FrameCaptureConfig
    from memory_readers.nfsu2_memory import NFSU2MemoryReader
    from memory_readers.virtual_gamepad import VirtualGamepad

    num_envs: int = getattr(args, "num_envs", 1)
    instance_mgr = None
    pcsx2_proc: subprocess.Popen | None = None

    gamepads: list[VirtualGamepad] = []
    readers: list[NFSU2MemoryReader] = []
    frame_captures: list[FrameCapture] = []
    envs: list[PCSX2RacerEnv] = []

    write_controller_db()

    # ── Detect available savestate slots ──────────────────────────────
    discovered = _discover_savestate_files()
    if discovered:
        savestate_slots = tuple(s for s, _ in discovered)
        print(f"[train] Multi-start: {len(discovered)} savestates (slots {list(savestate_slots)})")
        print(f"[train]   Run --setup-savestates first if PINE slots are not yet populated.")
    else:
        # No extra saves — fall back to slot 0 (the default highway save
        # loaded at PCSX2 boot, saved to slot 0 below).
        savestate_slots = (0,)
        print("[train] Single start position (slot 0)")

    env_config = PCSX2EnvConfig(device=args.device, savestate_slots=savestate_slots)

    if num_envs == 1:
        # ── Single-env path (original, no isolation overhead) ──────────
        gamepad = VirtualGamepad()
        gamepad.open()
        gamepads.append(gamepad)

        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile, turbo=args.turbo)
        wait_for_pcsx2_ready()

        reader = NFSU2MemoryReader()
        reader.open()
        if not reader.is_calibrated():
            print(
                "[rocm-racer] ERROR: No calibration data found.\n"
                "  Run:  python main.py --calibrate",
                file=sys.stderr,
            )
            gamepad.close()
            pcsx2_proc.terminate()
            sys.exit(1)

        print("[train] Saving initial state to slot 0 for episode resets...")
        reader.save_state(0)
        time.sleep(0.5)
        readers.append(reader)

        fc = FrameCapture(FrameCaptureConfig())
        fc.open()
        frame_captures.append(fc)

        env = PCSX2RacerEnv(
            memory_reader=reader, gamepad=gamepad,
            config=env_config, frame_capture=fc,
        )
        envs.append(env)
        vec_env = DummyVecEnv([lambda: env])

    else:
        # ── Multi-env path (InstanceManager + ThreadedVecEnv) ──────────
        from environments.instance_manager import InstanceManager
        from environments.threaded_vec_env import ThreadedVecEnv

        instance_mgr = InstanceManager(
            num_envs=num_envs, iso=iso, statefile=args.statefile,
        )

        print(f"[train] Launching {num_envs} PCSX2 instances (staggered)...")

        for i in range(num_envs):
            # 1. Prepare config + create gamepad BEFORE launching PCSX2
            #    so SDL discovers it on startup.
            cfg = instance_mgr.prepare_instance(i)
            instance_mgr.instances.append(cfg)

            gamepad = VirtualGamepad(name=f"rocm-racer-env-{i}")
            gamepad.open()
            gamepads.append(gamepad)

            # 2. Launch this instance's PCSX2
            instance_mgr.launch_instance(
                cfg, turbo=args.turbo, gamepad_device=gamepad.device_path,
            )
            instance_mgr.wait_for_instance(cfg)

            # 3. Connect PINE to this instance's socket
            reader = NFSU2MemoryReader(pine_socket=cfg.pine_socket)
            reader.open()
            if not reader.is_calibrated():
                print(
                    f"[instance-{i}] ERROR: No calibration data. Run --calibrate first.",
                    file=sys.stderr,
                )
                instance_mgr.cleanup()
                sys.exit(1)

            print(f"[instance-{i}] Saving initial state to slot 0...")
            reader.save_state(0)
            time.sleep(0.5)
            readers.append(reader)

            # 4. Frame capture — create but don't open yet (windows not tiled)
            fc_cfg = FrameCaptureConfig(pcsx2_pid=cfg.pcsx2_pid)
            fc = FrameCapture(fc_cfg)
            frame_captures.append(fc)

            env = PCSX2RacerEnv(
                memory_reader=reader, gamepad=gamepad,
                config=env_config, frame_capture=fc,
            )
            envs.append(env)

        # Tile windows FIRST, then open frame captures with final positions
        instance_mgr.tile_windows()
        time.sleep(0.5)  # let Hyprland finish repositioning
        for fc in frame_captures:
            fc.open()

        vec_env = ThreadedVecEnv(envs)
        print(f"[train] {num_envs} environments ready.")

    # ── PPO model ──────────────────────────────────────────────────────

    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    tb_log = args.tensorboard_log or str(REPO_ROOT / "runs")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = f"ppo_nfsu2_{run_tag}"

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(models_dir),
        name_prefix=run_prefix,
        verbose=1,
    )

    monitor_cb = TrainingMonitorCallback(
        preview=not args.no_preview,
        preview_scale=4,
        preview_interval=5,
    )

    callbacks = CallbackList([checkpoint_cb, monitor_cb])

    if args.load_model:
        model_path = Path(args.load_model)
        if not model_path.exists():
            print(f"[train] ERROR: model not found: {model_path}")
            sys.exit(1)
        model = PPO.load(
            model_path,
            env=vec_env,
            device=env_config.device,
            tensorboard_log=tb_log,
        )
        reset_timesteps = False
        print(f"[train] Loaded model from {model_path}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=tb_log,
            device=env_config.device,
        )
        reset_timesteps = True
        print("[train] Created new model")

    print(f"[train] Starting PPO training for {args.timesteps:,} timesteps...")
    print(f"[train] Run tag: {run_tag}")
    print(f"[train] Environments: {num_envs}")
    print(f"[train] Checkpoints → {models_dir}/{run_prefix}_*_steps.zip")
    print(f"[train] TensorBoard → {tb_log}")
    if not args.no_preview:
        print("[train] Preview window: 'rocm-racer' (green=steering, red=accel/brake)")
    print("[train] Press Ctrl-C to stop and save final model.\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="ppo_nfsu2",
            reset_num_timesteps=reset_timesteps,
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving final model...")
    finally:
        final_path = str(models_dir / f"{run_prefix}_final")
        model.save(final_path)
        print(f"[train] Final model saved → {final_path}.zip")
        vec_env.close()
        if args.turbo:
            set_pcsx2_speed_scalar(1.0)
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
        if instance_mgr is not None:
            instance_mgr.cleanup()


# ---------------------------------------------------------------------------
# Mode: default (test — hold accelerate)
# ---------------------------------------------------------------------------

def _run_test(args: argparse.Namespace, iso: Path) -> None:
    """Hold accelerate button for manual testing."""
    from memory_readers.virtual_gamepad import VirtualGamepad
    from evdev import ecodes as e

    pcsx2_proc: subprocess.Popen | None = None
    gamepad: VirtualGamepad | None = None

    def _shutdown(sig, frame):
        print("\n[rocm-racer] Shutting down...")
        if gamepad is not None:
            try:
                gamepad.release_button(e.BTN_SOUTH)
            except Exception:
                pass
            gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 1. Write SDL3 controller DB so PCSX2 recognises the virtual gamepad.
    write_controller_db()

    # 2. Create the virtual gamepad BEFORE launching PCSX2.
    gamepad = VirtualGamepad()
    gamepad.open()

    # 3. Launch PCSX2 — it will find the gamepad during SDL init.
    if not args.no_launch:
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    # 4. Send accelerate input and hold it for testing.
    print("[rocm-racer] Test mode: accelerating (Cross + right stick up). Press Ctrl-C to stop.")
    try:
        gamepad.hold_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
        while True:
            time.sleep(1.0)
    finally:
        if gamepad is not None:
            gamepad.release_button(e.BTN_SOUTH)
            gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()


if __name__ == "__main__":
    main()
