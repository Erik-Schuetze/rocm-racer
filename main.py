from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

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
    """Automated vehicle struct discovery via differential memory scanning.

    Launches PCSX2, uses the virtual gamepad to alternate between braking
    (car stopped) and accelerating (car moving), takes EE RAM snapshots
    at each state, and narrows candidates using differential scans.
    Finally, auto-verifies surviving candidates by reading adjacent
    struct fields.
    """
    import math
    from memory_readers.nfsu2_memory import NFSU2MemoryReader, TelemetryOffsets
    from memory_readers.virtual_gamepad import VirtualGamepad
    from evdev import ecodes as e

    BRAKE_TIME = 1.5     # seconds to hold brake (decelerate, not reverse)
    COAST_TIME = 6.0     # seconds to wait with no input for car to reach 0
    ACCEL_TIME = 3.0     # seconds to hold accelerate
    MAX_CYCLES = 5       # max accel/brake narrowing cycles
    TARGET_CANDIDATES = 50  # stop narrowing when we reach this many

    print("[rocm-racer] ══════════════════════════════════════════════")
    print("[rocm-racer]  Calibration mode — automated struct finder  ")
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

    candidates: list[int] | None = None

    def _stop_car() -> None:
        """Brake via analog stick then coast to a full stop."""
        gamepad.send(steering=0.0, throttle=0.0, brake=1.0)
        time.sleep(BRAKE_TIME)
        gamepad.send(steering=0.0, throttle=0.0, brake=0.0)
        time.sleep(COAST_TIME)

    try:
        # Car is already stopped from savestate — take initial stopped snapshot
        print("\n[calibrate] Snapshot (stopped — from savestate)...")
        snap_stopped = reader.snapshot_ee_ram()
        time.sleep(1.0)
        snap_stopped_2 = reader.snapshot_ee_ram()

        results = NFSU2MemoryReader.diff_scan(
            snap_stopped, snap_stopped_2, "unchanged", candidates
        )
        prev_count = len(candidates) if candidates else "all"
        candidates = [addr for addr, _, _ in results]
        print(f"[calibrate] Filter 'unchanged@stop': {prev_count} → {len(candidates):,}")

        for cycle in range(1, MAX_CYCLES + 1):
            print(f"\n[calibrate] Cycle {cycle}/{MAX_CYCLES}")

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates.")
                break

            # --- Accelerate ---
            print(f"[calibrate] Accelerating for {ACCEL_TIME}s...")
            gamepad.hold_button(e.BTN_SOUTH)  # Cross = accelerate
            gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
            time.sleep(ACCEL_TIME)

            # --- Snapshot while moving (throttle still held) ---
            print("[calibrate] Snapshot (moving)...")
            snap_moving = reader.snapshot_ee_ram()
            time.sleep(1.0)
            snap_moving_2 = reader.snapshot_ee_ram()

            gamepad.release_button(e.BTN_SOUTH)
            gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

            # --- Filter: increased (stopped → moving) ---
            results = NFSU2MemoryReader.diff_scan(
                snap_stopped_2, snap_moving, "increased", candidates
            )
            candidates = [addr for addr, _, _ in results]
            print(f"[calibrate] Filter 'increased': → {len(candidates):,}")

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates.")
                break

            # --- Filter: unchanged while moving (eliminates frame
            # counters and animation state that keep changing) ---
            results = NFSU2MemoryReader.diff_scan(
                snap_moving, snap_moving_2, "unchanged", candidates
            )
            candidates = [addr for addr, _, _ in results]
            print(f"[calibrate] Filter 'unchanged@moving': → {len(candidates):,}")

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates.")
                break

            # --- Stop the car ---
            print("[calibrate] Stopping car...")
            _stop_car()

            print("[calibrate] Snapshot (stopped)...")
            snap_stopped = reader.snapshot_ee_ram()
            time.sleep(1.0)
            snap_stopped_2 = reader.snapshot_ee_ram()

            results = NFSU2MemoryReader.diff_scan(
                snap_moving, snap_stopped, "decreased", candidates
            )
            candidates = [addr for addr, _, _ in results]
            print(f"[calibrate] Filter 'decreased': → {len(candidates):,}")

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates.")
                break

            results = NFSU2MemoryReader.diff_scan(
                snap_stopped, snap_stopped_2, "unchanged", candidates
            )
            candidates = [addr for addr, _, _ in results]
            print(f"[calibrate] Filter 'unchanged@stop': → {len(candidates):,}")

        # --- Release all input ---
        try:
            gamepad.release_button(e.BTN_SOUTH)
        except Exception:
            pass
        gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

        if not candidates:
            print("\n[calibrate] ERROR: No candidates survived. Try re-running calibration.")
            return

        # --- Auto-verify candidates ---
        print(f"\n[calibrate] Verifying {len(candidates):,} candidates...")
        scored = _score_candidates(reader, candidates, gamepad)

        # --- Display results ---
        print(f"\n[calibrate] ── Results ({'top 20' if len(scored) > 20 else 'all'}) ──")
        print(f"  {'Score':>5}  {'Speed Addr':>14}  {'Struct Base':>14}  "
              f"{'Speed':>8}  {'RPM':>8}  {'Gear':>4}  {'Pos X':>10}  Speed Samples")
        print(f"  {'─' * 5}  {'─' * 14}  {'─' * 14}  {'─' * 8}  {'─' * 8}  {'─' * 4}  {'─' * 10}  {'─' * 30}")
        for entry in scored[:20]:
            samples_str = " → ".join(f"{v:.1f}" for v in entry.get("speed_samples", []))
            print(
                f"  {entry['score']:5d}  "
                f"0x{entry['speed_addr']:08X}  "
                f"0x{entry['struct_base']:08X}  "
                f"{entry['speed_ms']:8.2f}  "
                f"{entry['rpm']:8.0f}  "
                f"{entry['gear']:4d}  "
                f"{entry['pos_x']:10.2f}  "
                f"{samples_str}"
            )

        best = scored[0]
        if best["score"] >= 5:
            speed_addr = best["speed_addr"]
            print(f"\n[calibrate] ✓ Best speed address: PS2 0x{speed_addr:08X} "
                  f"(score {best['score']}/8)")

            # --- Struct discovery: probe ±0x200 around speed addr ---
            print("[calibrate] Probing neighborhood to find struct layout...")
            layout = _probe_struct_layout(reader, gamepad, speed_addr)

            # Save to calibration file
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            cal_data = {}
            for k, v in layout.items():
                if isinstance(v, int):
                    cal_data[k] = f"0x{v:08X}"
                else:
                    cal_data[k] = v
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(cal_data, f, indent=2)
            print(f"[calibrate] Saved to {CALIBRATION_FILE}")
            print(f"[calibrate] Use with:  python main.py --telemetry")
        else:
            print(f"\n[calibrate] ⚠ No high-confidence match (best score: {best['score']}/8).")
            print("[calibrate] Try running calibration again, or use --snap/--scan-diff manually.")

    finally:
        reader.close()
        gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
            print("[rocm-racer] PCSX2 terminated.")


def _probe_struct_layout(
    reader: "NFSU2MemoryReader",
    gamepad: "VirtualGamepad",
    speed_addr: int,
) -> dict:
    """Probe memory around the speed address to find actual struct offsets.

    Reads ±0x200 bytes around speed_addr while stopped and while driving,
    then identifies position (floats that change smoothly while driving),
    RPM (float in engine range), and gear (small int).

    Returns a dict of discovered addresses.
    """
    import math, struct as st
    from evdev import ecodes as e

    PROBE_RANGE = 0x200  # bytes before and after speed_addr
    start = max(0, (speed_addr - PROBE_RANGE) & ~3)  # align to 4-byte boundary
    end = speed_addr + PROBE_RANGE
    region_size = end - start

    def _read_region(fmt: str = "<f") -> dict[int, float | int]:
        """Read the probe region, interpreting each 4-byte slot with *fmt*."""
        default = float("nan") if fmt == "<f" else 0
        try:
            raw = reader._pine.read_bulk(start, region_size)
            vals = {}
            for i in range(0, region_size, 4):
                vals[start + i] = st.unpack_from(fmt, raw, i)[0]
            return vals
        except (RuntimeError, OSError):
            vals = {}
            read_fn = reader._read_f32 if fmt == "<f" else reader._read_i32
            for i in range(0, region_size, 4):
                addr = start + i
                try:
                    vals[addr] = read_fn(addr)
                except (RuntimeError, OSError):
                    vals[addr] = default
            return vals

    # Read while stopped
    stopped_vals = _read_region("<f")
    stopped_ints = _read_region("<i")

    # Accelerate for 3s then read again
    gamepad.hold_button(e.BTN_SOUTH)
    gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
    time.sleep(3.0)

    moving_vals = _read_region("<f")
    moving_ints = _read_region("<i")

    # Wait a moment and read a third time to detect position (still changing)
    time.sleep(1.0)
    moving2_vals = _read_region("<f")

    gamepad.release_button(e.BTN_SOUTH)
    gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

    # --- Find position candidates: floats that changed between stopped→moving
    # AND keep changing between moving→moving2 (they track position) ---
    pos_candidates: list[tuple[int, float, float, float]] = []
    for addr in stopped_vals:
        sv, mv, m2v = stopped_vals[addr], moving_vals.get(addr, float("nan")), moving2_vals.get(addr, float("nan"))
        if not all(math.isfinite(v) for v in (sv, mv, m2v)):
            continue
        # Position should change between all three readings
        if abs(mv - sv) > 1.0 and abs(m2v - mv) > 0.1:
            # And should be world-coordinate magnitude (not tiny or huge)
            if all(abs(v) < 100000 for v in (sv, mv, m2v)):
                pos_candidates.append((addr, sv, mv, m2v))

    # --- Find RPM: float in engine range while moving ---
    rpm_addr = None
    for addr, mv in moving_vals.items():
        if addr == speed_addr:
            continue
        if math.isfinite(mv) and 500.0 <= mv <= 15000.0:
            sv = stopped_vals.get(addr, float("nan"))
            # RPM should be low-ish when stopped (idle ~750) or zero
            if math.isfinite(sv) and sv < 2000.0:
                rpm_addr = addr
                break  # take first match

    # --- Find gear: int 1–6 while moving, 0-1 while stopped ---
    gear_addr = None
    for addr in moving_ints:
        mv = moving_ints[addr]
        sv = stopped_ints.get(addr, -1)
        if 1 <= mv <= 6 and 0 <= sv <= 1:
            gear_addr = addr
            break

    # --- Display findings ---
    print(f"\n[calibrate] ── Struct Discovery ──")
    print(f"  Speed addr:    0x{speed_addr:08X}")

    if pos_candidates:
        print(f"\n  Position candidates (changed while driving):")
        for addr, sv, mv, m2v in pos_candidates[:6]:
            delta = speed_addr - addr
            print(f"    0x{addr:08X} (speed{delta:+d}):  "
                  f"stopped={sv:10.2f}  moving={mv:10.2f}  moving2={m2v:10.2f}")

    if rpm_addr is not None:
        rv_s = stopped_vals.get(rpm_addr, float("nan"))
        rv_m = moving_vals.get(rpm_addr, float("nan"))
        print(f"\n  RPM addr:      0x{rpm_addr:08X} (speed{speed_addr - rpm_addr:+d})  "
              f"stopped={rv_s:.0f}  moving={rv_m:.0f}")
    else:
        print(f"\n  RPM addr:      not found")

    if gear_addr is not None:
        gv_s = stopped_ints.get(gear_addr, -1)
        gv_m = moving_ints.get(gear_addr, -1)
        print(f"  Gear addr:     0x{gear_addr:08X} (speed{speed_addr - gear_addr:+d})  "
              f"stopped={gv_s}  moving={gv_m}")
    else:
        print(f"  Gear addr:     not found")

    layout = {"speed_addr": speed_addr}
    if pos_candidates:
        layout["pos_candidates"] = [f"0x{a:08X}" for a, _, _, _ in pos_candidates[:3]]
    if rpm_addr is not None:
        layout["rpm_addr"] = rpm_addr
    if gear_addr is not None:
        layout["gear_addr"] = gear_addr

    return layout


def _score_candidates(
    reader: "NFSU2MemoryReader",
    candidates: list[int],
    gamepad: "VirtualGamepad",
) -> list[dict]:
    """Score candidate speed addresses with temporal live-driving verification.

    Takes multiple readings over time while accelerating and checks that
    the candidate's speed actually changes (not a constant like gravity).

    Returns a list of dicts sorted by score (descending).
    """
    import math
    from evdev import ecodes as e

    addr_list = []
    for speed_addr in candidates:
        struct_base = speed_addr - 0x24
        if struct_base >= 0:
            addr_list.append((speed_addr, struct_base))

    # --- Read while stopped ---
    print("[calibrate] Verification: reading while stopped...")
    stopped: dict[int, dict] = {}
    for speed_addr, base in addr_list:
        try:
            stopped[speed_addr] = {
                "speed": reader._read_f32(speed_addr),
                "pos_x": reader._read_f32(base + 0x00),
                "pos_y": reader._read_f32(base + 0x04),
                "pos_z": reader._read_f32(base + 0x08),
            }
        except (RuntimeError, OSError):
            continue

    # --- Accelerate at varying throttle and take 3 speed samples ---
    throttle_steps = [0.1, 0.25, 0.5]
    print(f"[calibrate] Verification: throttle steps {throttle_steps}...")
    gamepad.hold_button(e.BTN_SOUTH)  # Cross = accelerate (required by NFS U2)

    speed_samples: dict[int, list[float]] = {a: [] for a in stopped}
    for throttle in throttle_steps:
        gamepad.send(steering=0.0, throttle=throttle, brake=0.0)
        time.sleep(1.5)
        for speed_addr in stopped:
            try:
                speed_samples[speed_addr].append(reader._read_f32(speed_addr))
            except (RuntimeError, OSError):
                speed_samples[speed_addr].append(float("nan"))

    # --- Final moving read (full struct) ---
    moving: dict[int, dict] = {}
    for speed_addr, base in addr_list:
        if speed_addr not in stopped:
            continue
        try:
            moving[speed_addr] = {
                "speed": reader._read_f32(speed_addr),
                "pos_x": reader._read_f32(base + 0x00),
                "pos_y": reader._read_f32(base + 0x04),
                "pos_z": reader._read_f32(base + 0x08),
                "rpm": reader._read_f32(base + 0x1A4),
                "gear": reader._read_i32(base + 0x1B0),
            }
        except (RuntimeError, OSError):
            continue

    gamepad.release_button(e.BTN_SOUTH)
    gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

    # --- Score ---
    results = []
    for speed_addr, base in addr_list:
        s = stopped.get(speed_addr)
        m = moving.get(speed_addr)
        if s is None or m is None:
            continue

        score = 0
        s_spd = s["speed"]
        m_spd = m["speed"]
        samples = speed_samples.get(speed_addr, [])

        # 1. Speed near zero when stopped (< 1 m/s)
        if math.isfinite(s_spd) and 0.0 <= s_spd < 1.0:
            score += 1

        # 2. Speed positive and realistic when moving (1–100 m/s)
        if math.isfinite(m_spd) and 1.0 <= m_spd <= 100.0:
            score += 1

        # 3. Speed CHANGED over time (not a constant like gravity)
        #    Check that samples are mostly increasing and distinct
        finite_samples = [v for v in samples if math.isfinite(v)]
        if len(finite_samples) >= 2:
            spread = max(finite_samples) - min(finite_samples)
            if spread > 0.5:  # speed changed by at least 0.5 m/s
                score += 2  # strong signal — eliminates constants
                # Bonus: monotonically increasing (accelerating)
                if all(b >= a - 0.1 for a, b in zip(finite_samples, finite_samples[1:])):
                    score += 1

        # 4. Position changed between stopped and moving
        s_pos = (s["pos_x"], s["pos_y"], s["pos_z"])
        m_pos = (m["pos_x"], m["pos_y"], m["pos_z"])
        if all(math.isfinite(v) for v in s_pos + m_pos):
            dist_sq = sum((a - b) ** 2 for a, b in zip(s_pos, m_pos))
            if dist_sq > 1.0:
                score += 1

        # 5. RPM in engine range while moving (500–15000)
        m_rpm = m["rpm"]
        if math.isfinite(m_rpm) and 500.0 <= m_rpm <= 15000.0:
            score += 1

        # 6. Gear is a small positive integer while moving
        m_gear = m["gear"]
        if 1 <= m_gear <= 6:
            score += 1

        results.append({
            "speed_addr": speed_addr,
            "struct_base": base,
            "score": score,
            "speed_ms": m_spd,
            "speed_samples": finite_samples,
            "pos_x": m_pos[0],
            "rpm": m_rpm,
            "gear": m_gear,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


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
    print(f"  {'PS2 Address':>14s}    {'If speed (+0x24)':>14s}    Value")
    print(f"  {'─' * 14}    {'─' * 14}    {'─' * 10}")

    # Re-read to show actual values
    reader.open()
    for ps2_addr in matches[:50]:  # cap output at 50
        raw = reader._read_f32(ps2_addr)
        struct_base = ps2_addr - 0x24
        print(f"  0x{ps2_addr:08X}    0x{struct_base:08X}    {raw:.4f}")
    reader.close()

    if len(matches) > 50:
        print(f"\n  ... and {len(matches) - 50} more (narrow the tolerance)")

    print(
        f"\n[rocm-racer] If you searched for absolute speed (m/s), the vehicle struct"
        f"\n  base is at PS2 address = match - 0x24."
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

    # Load calibration — support both old (vehicle_struct_addr) and
    # new (individual speed_addr / pos / rpm / gear) formats.
    cal: dict = {}
    vehicle_addr = args.vehicle_addr
    if vehicle_addr is None and CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            cal = json.load(f)
        if "vehicle_struct_addr" in cal:
            vehicle_addr = int(cal["vehicle_struct_addr"], 0)
        elif "speed_addr" in cal:
            # New format — individual addresses
            vehicle_addr = 1  # non-zero sentinel; we read individual addrs
        print(f"[rocm-racer] Auto-loaded calibration from {CALIBRATION_FILE}")
    if vehicle_addr is None:
        print(
            "[rocm-racer] ERROR: --telemetry requires --vehicle-addr 0xADDR\n"
            "  or a saved calibration file (run --calibrate first).",
            file=sys.stderr,
        )
        sys.exit(1)

    offsets = TelemetryOffsets(vehicle_struct_addr=vehicle_addr)
    reader = NFSU2MemoryReader(offsets=offsets)

    # Resolve individual addresses from calibration
    speed_addr = int(cal["speed_addr"], 0) if "speed_addr" in cal else None
    pos_addrs = [int(a, 0) for a in cal.get("pos_candidates", [])]
    rpm_addr = int(cal["rpm_addr"], 0) if "rpm_addr" in cal else None
    gear_addr = int(cal["gear_addr"], 0) if "gear_addr" in cal else None
    use_individual = speed_addr is not None

    if not args.no_launch:
        write_controller_db()
        from memory_readers.virtual_gamepad import VirtualGamepad
        gamepad = VirtualGamepad()
        gamepad.open()
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    reader.open()
    if use_individual:
        addr_info = f"speed=0x{speed_addr:08X}"
        if pos_addrs:
            addr_info += f", pos[0]=0x{pos_addrs[0]:08X}"
        print(f"[rocm-racer] Telemetry logging started ({addr_info}, backend=PINE)")
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
                if use_individual:
                    spd = reader._read_f32(speed_addr)
                    pos = tuple(reader._read_f32(a) for a in pos_addrs) if pos_addrs else (0.0, 0.0, 0.0)
                    rpm = reader._read_f32(rpm_addr) if rpm_addr else 0.0
                    gear = reader._read_i32(gear_addr) if gear_addr else 0
                    print(
                        f"  Speed: {spd * 3.6:6.1f} km/h ({spd:.2f} m/s)  "
                        f"Pos: ({', '.join(f'{v:9.2f}' for v in pos)})  "
                        f"RPM: {rpm:7.0f}  Gear: {gear}"
                    )
                else:
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
