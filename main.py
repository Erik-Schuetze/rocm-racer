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

def launch_pcsx2(iso: Path, statefile: Path | None = None) -> subprocess.Popen:
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

    log_path = REPO_ROOT / "pcsx2.log"
    log_fh = open(log_path, "w")
    print(f"[rocm-racer] Launching: {' '.join(cmd)}")
    print(f"[rocm-racer] PCSX2 output → {log_path}")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)


def wait_for_pcsx2_ready(
    timeout: float = 30.0,
    poll: float = 0.25,
    post_ready_delay: float = 2.0,
) -> None:
    """Block until PCSX2 has loaded the savestate and the game is running.

    Monitors the PCSX2 emulog for the ``Opened gamepad`` marker, which
    confirms that both the savestate has been restored **and** SDL has
    connected the virtual gamepad.  After the marker is found a short
    delay lets the game establish its baseline controller state before
    any input is sent.

    Falls back to a fixed timeout if the emulog is missing or the marker
    never appears (e.g. when running without a virtual gamepad).
    """
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

    # --- Default: test mode (gamepad accelerate) ---
    _run_test(args, iso)


# ---------------------------------------------------------------------------
# Mode: --calibrate
# ---------------------------------------------------------------------------

def _run_calibrate(args: argparse.Namespace, iso: Path) -> None:
    """Automated vehicle struct discovery via structural fingerprinting.

    Uses 16-byte alignment constraints, quaternion validation, and
    multi-field pattern matching to find the player vehicle struct in
    the 32 MB EE RAM.  Then discovers a static pointer so the calibration
    survives across savestate reloads.
    """
    import math
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets
    from memory_readers.virtual_gamepad import VirtualGamepad
    from evdev import ecodes as e

    ACCEL_TIME = 3.0     # seconds to hold accelerate
    STRUCT_MIN_SIZE = 0x168  # bytes — must cover through gear at 0x164

    print("[rocm-racer] ══════════════════════════════════════════════")
    print("[rocm-racer]  Calibration — structural fingerprint scanner ")
    print("[rocm-racer] ══════════════════════════════════════════════")

    # --- Setup: gamepad + PCSX2 ---
    write_controller_db()
    gamepad = VirtualGamepad()
    gamepad.open()
    pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
    wait_for_pcsx2_ready()

    offsets = TelemetryOffsets(vehicle_struct_addr=1)  # dummy
    reader = NFSU2MemoryReader(offsets=offsets)
    reader.open()

    try:
        # ── Phase 1: stationary structural fingerprint ──
        print("\n[calibrate] Phase 1: scanning stationary fingerprint (car must be stopped)...")
        snap_stopped = reader.snapshot_ee_ram()
        candidates = _phase1_stationary_scan(snap_stopped, STRUCT_MIN_SIZE)
        print(f"[calibrate] Phase 1: {len(candidates):,} candidates after fingerprint filter")

        if not candidates:
            print("[calibrate] ERROR: No candidates found. Is the car stopped in-gear?")
            return

        # ── Phase 2: acceleration delta ──
        print(f"\n[calibrate] Phase 2: accelerating for {ACCEL_TIME}s...")
        gamepad.hold_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
        time.sleep(ACCEL_TIME)

        snap_moving = reader.snapshot_ee_ram()

        gamepad.release_button(e.BTN_SOUTH)
        gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

        candidates = _phase2_acceleration_verify(
            snap_stopped, snap_moving, candidates
        )
        print(f"[calibrate] Phase 2: {len(candidates):,} candidates after acceleration filter")

        if not candidates:
            print("[calibrate] ERROR: No candidates survived acceleration filter.")
            return

        # ── Phase 3: structural integrity scoring ──
        print(f"\n[calibrate] Phase 3: scoring {len(candidates):,} candidates on structural integrity...")
        scored = _phase3_integrity_score(snap_stopped, snap_moving, candidates)

        # Display results
        print(f"\n[calibrate] ── Results ({'top 20' if len(scored) > 20 else 'all'}) ──")
        print(f"  {'Score':>5}  {'Struct Base':>14}  "
              f"{'Speed(stop)':>11}  {'Speed(move)':>11}  {'RPM(idle)':>9}  {'Quat Σ²':>7}")
        print(f"  {'─' * 5}  {'─' * 14}  {'─' * 11}  {'─' * 11}  {'─' * 9}  {'─' * 7}")
        for entry in scored[:20]:
            print(
                f"  {entry['score']:5d}  "
                f"0x{entry['base']:08X}  "
                f"{entry['speed_stopped']:11.4f}  "
                f"{entry['speed_moving']:11.4f}  "
                f"{entry['rpm_idle']:9.0f}  "
                f"{entry['quat_sq']:7.4f}"
            )

        best = scored[0]
        if best["score"] < 3:
            print(f"\n[calibrate] ⚠ No high-confidence match (best score: {best['score']}).")
            print("[calibrate] Try re-running calibration with a clean savestate.")
            return

        vehicle_base = best["base"]
        print(f"\n[calibrate] ✓ Best struct base: PS2 0x{vehicle_base:08X} "
              f"(score {best['score']})")

        # ── Phase 4: static pointer discovery ──
        print("\n[calibrate] Phase 4: searching for static pointer...")
        static_ptrs = _phase4_find_static_pointers(snap_moving, vehicle_base)

        static_ptr_addr = 0
        if static_ptrs:
            static_ptr_addr = static_ptrs[0]
            print(f"[calibrate] ✓ Found {len(static_ptrs)} static pointer(s):")
            for ptr in static_ptrs[:5]:
                print(f"    0x{ptr:08X}")
            print(f"[calibrate]   Using 0x{static_ptr_addr:08X}")
        else:
            print("[calibrate] ⚠ No static pointer found in 0x003X–0x005X range.")
            print("[calibrate]   Saving direct address (may break on savestate reload).")

        # ── Save calibration ──
        cal_data = {
            "vehicle_struct_addr": f"0x{vehicle_base:08X}",
        }
        if static_ptr_addr:
            cal_data["static_pointer_addr"] = f"0x{static_ptr_addr:08X}"
        cal_data["calibration_score"] = best["score"]

        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(cal_data, f, indent=2)
        print(f"\n[calibrate] Saved to {CALIBRATION_FILE}")
        print(f"[calibrate] Use with:  python main.py --telemetry")

    finally:
        reader.close()
        gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
            print("[rocm-racer] PCSX2 terminated.")


# ---------------------------------------------------------------------------
# Calibration phase helpers
# ---------------------------------------------------------------------------

def _phase1_stationary_scan(data: bytes, struct_min_size: int) -> list[int]:
    """Phase 1: find 16-byte-aligned bases matching a stationary vehicle.

    Checks: speed≈0, velocity≈(0,0,0), gear=1, quaternion normalized,
    position finite and in valid world range.

    Returns candidate base addresses (PS2 byte offsets).
    """
    from memory_readers.nfsu2_memory import _EE_RAM_SIZE

    f32 = np.frombuffer(data, dtype=np.float32)
    i32 = np.frombuffer(data, dtype=np.int32)

    max_base = _EE_RAM_SIZE - struct_min_size
    # 16-byte aligned candidates (every 4 float32 slots)
    bases = np.arange(0, max_base, 16, dtype=np.int64)

    def f_at(offset: int) -> np.ndarray:
        return f32[(bases + offset) // 4]

    def i_at(offset: int) -> np.ndarray:
        return i32[(bases + offset) // 4]

    mask = np.ones(len(bases), dtype=bool)

    # Speed ≈ 0 when stopped
    speed = f_at(0x090)
    mask &= np.isfinite(speed) & (np.abs(speed) < 0.5)

    # Velocity ≈ (0, 0, 0)
    for off in (0x070, 0x074, 0x078):
        v = f_at(off)
        mask &= np.isfinite(v) & (np.abs(v) < 0.1)

    # Gear = 1 (in-drive, stationary)
    mask &= i_at(0x164) == 1

    # Quaternion at +0x030: 4 floats, sum of squares ≈ 1.0
    r = [f_at(0x030 + i * 4) for i in range(4)]
    quat_sq = sum(ri * ri for ri in r)
    mask &= np.isfinite(quat_sq) & (quat_sq > 0.9) & (quat_sq < 1.1)

    # Position: finite and in valid world range
    for off in (0x020, 0x024, 0x028):
        p = f_at(off)
        mask &= np.isfinite(p) & (np.abs(p) < 100_000)

    return bases[mask].tolist()


def _phase2_acceleration_verify(
    data_stopped: bytes,
    data_moving: bytes,
    candidates: list[int],
) -> list[int]:
    """Phase 2: keep candidates that show realistic acceleration behavior.

    Checks: speed>0, velocity non-zero in XZ, RPM in engine range,
    gear still 1, position horizontally changed.
    """
    f32_s = np.frombuffer(data_stopped, dtype=np.float32)
    f32_m = np.frombuffer(data_moving, dtype=np.float32)
    i32_m = np.frombuffer(data_moving, dtype=np.int32)

    bases = np.array(candidates, dtype=np.int64)

    def f_at(arr: np.ndarray, offset: int) -> np.ndarray:
        return arr[(bases + offset) // 4]

    def i_at(arr: np.ndarray, offset: int) -> np.ndarray:
        return arr[(bases + offset) // 4]

    mask = np.ones(len(bases), dtype=bool)

    # Speed now positive and < 30 m/s (~108 km/h, first-gear range)
    speed = f_at(f32_m, 0x090)
    mask &= np.isfinite(speed) & (speed > 0.1) & (speed < 30.0)

    # Velocity has non-zero horizontal component
    vel_x = f_at(f32_m, 0x070)
    vel_z = f_at(f32_m, 0x078)
    mask &= (np.abs(vel_x) > 0.01) | (np.abs(vel_z) > 0.01)

    # RPM in engine range
    rpm = f_at(f32_m, 0x160)
    mask &= np.isfinite(rpm) & (rpm >= 500) & (rpm <= 12000)

    # Gear still = 1 (should not have shifted in 3s from standstill)
    mask &= i_at(i32_m, 0x164) == 1

    # Position moved horizontally since stopped
    pos_x_s = f_at(f32_s, 0x020)
    pos_z_s = f_at(f32_s, 0x028)
    pos_x_m = f_at(f32_m, 0x020)
    pos_z_m = f_at(f32_m, 0x028)
    horiz_dist_sq = (pos_x_m - pos_x_s) ** 2 + (pos_z_m - pos_z_s) ** 2
    mask &= np.isfinite(horiz_dist_sq) & (horiz_dist_sq > 1.0)

    return bases[mask].tolist()


def _phase3_integrity_score(
    data_stopped: bytes,
    data_moving: bytes,
    candidates: list[int],
) -> list[dict]:
    """Phase 3: score candidates on structural integrity.

    Higher score = more likely to be the real player vehicle struct.
    """
    import math

    f32_s = np.frombuffer(data_stopped, dtype=np.float32)
    f32_m = np.frombuffer(data_moving, dtype=np.float32)

    results: list[dict] = []
    for base in candidates:
        score = 0

        def _f(arr: np.ndarray, off: int) -> float:
            return float(arr[(base + off) // 4])

        # Quaternion precision (tighter tolerance)
        r = [_f(f32_s, 0x030 + i * 4) for i in range(4)]
        quat_sq = sum(v * v for v in r)
        if 0.99 < quat_sq < 1.01:
            score += 2
        elif 0.95 < quat_sq < 1.05:
            score += 1

        # RPM idle when stopped (~800-1200, not 0 or garbage)
        rpm_idle = _f(f32_s, 0x160)
        if math.isfinite(rpm_idle) and 500 <= rpm_idle <= 1500:
            score += 2

        # Speed exactly 0 when stopped
        speed_stopped = _f(f32_s, 0x090)
        if abs(speed_stopped) < 0.01:
            score += 1

        # Speed while moving is realistic
        speed_moving = _f(f32_m, 0x090)
        if 1.0 <= speed_moving <= 25.0:
            score += 1

        # RPM while moving in driving range
        rpm_moving = _f(f32_m, 0x160)
        if math.isfinite(rpm_moving) and 2000 <= rpm_moving <= 8000:
            score += 1

        # Vertical position roughly unchanged (not falling/flying)
        pos_y_s = _f(f32_s, 0x024)
        pos_y_m = _f(f32_m, 0x024)
        if math.isfinite(pos_y_s) and math.isfinite(pos_y_m):
            if abs(pos_y_m - pos_y_s) < 2.0:
                score += 1

        results.append({
            "base": base,
            "score": score,
            "speed_stopped": speed_stopped,
            "speed_moving": speed_moving,
            "rpm_idle": rpm_idle,
            "quat_sq": quat_sq,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _phase4_find_static_pointers(data: bytes, vehicle_base: int) -> list[int]:
    """Phase 4: find static pointers to the vehicle struct.

    Searches all 32 MB for uint32 values matching the vehicle base address
    (with and without kseg mirror bits).  Returns addresses in the static
    data range (0x003XXXXX–0x005XXXXX), sorted by address.
    """
    u32 = np.frombuffer(data, dtype=np.uint32)

    # The game may store the address as physical (kuseg) or kseg0/kseg1
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

    # Load calibration
    cal: dict = {}
    vehicle_addr = args.vehicle_addr
    static_ptr = 0

    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            cal = json.load(f)
        if "static_pointer_addr" in cal:
            static_ptr = int(cal["static_pointer_addr"], 0)
        if "vehicle_struct_addr" in cal and vehicle_addr is None:
            vehicle_addr = int(cal["vehicle_struct_addr"], 0)
        print(f"[rocm-racer] Auto-loaded calibration from {CALIBRATION_FILE}")

    if vehicle_addr is None and static_ptr == 0:
        print(
            "[rocm-racer] ERROR: --telemetry requires --vehicle-addr 0xADDR\n"
            "  or a saved calibration file (run --calibrate first).",
            file=sys.stderr,
        )
        sys.exit(1)

    offsets = TelemetryOffsets(
        static_pointer_addr=static_ptr,
        vehicle_struct_addr=vehicle_addr or 0,
    )
    reader = NFSU2MemoryReader(offsets=offsets)

    if not args.no_launch:
        write_controller_db()
        from memory_readers.virtual_gamepad import VirtualGamepad
        gamepad = VirtualGamepad()
        gamepad.open()
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    reader.open()
    if static_ptr:
        print(
            f"[rocm-racer] Telemetry logging started "
            f"(static ptr @ 0x{static_ptr:08X}, backend=PINE)"
        )
    else:
        print(
            f"[rocm-racer] Telemetry logging started "
            f"(vehicle struct @ PS2 0x{vehicle_addr:08X}, backend=PINE)"
        )
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
