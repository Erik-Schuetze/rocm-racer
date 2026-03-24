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
# Training
# ---------------------------------------------------------------------------

def train_ppo(env, total_timesteps: int, tensorboard_log: str | None) -> None:
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cuda",
        tensorboard_log=tensorboard_log,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


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

    pcsx2_proc: subprocess.Popen | None = None

    def _shutdown(sig, frame):
        print("\n[rocm-racer] Shutting down...")
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

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

    BRAKE_TIME = 3.0     # seconds to hold brake (decelerate, not reverse)
    COAST_TIME = 5.0     # seconds to wait with no input for car to reach 0
    ACCEL_TIME = 5.0     # seconds to hold accelerate
    MAX_CYCLES = 3       # max accel/brake narrowing cycles
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
    reader = NFSU2MemoryReader(offsets=offsets, pid=pcsx2_proc.pid)
    reader.open()

    candidates: list[int] | None = None

    try:
        for cycle in range(1, MAX_CYCLES + 1):
            # --- Phase 1: brake to slow down, then coast to full stop ---
            # Square (BTN_WEST) is "Brake/Reverse" — holding it while
            # stopped makes the car reverse, so we brake briefly then
            # release everything and wait for the car to coast to speed 0.
            print(f"\n[calibrate] Cycle {cycle}/{MAX_CYCLES} — stopping car...")
            gamepad.hold_button(e.BTN_WEST)       # Square = brake
            gamepad.send(steering=0.0, throttle=0.0, brake=1.0)
            time.sleep(BRAKE_TIME)
            gamepad.release_button(e.BTN_WEST)
            gamepad.send(steering=0.0, throttle=0.0, brake=0.0)
            print(f"[calibrate] Coasting to zero for {COAST_TIME}s...")
            time.sleep(COAST_TIME)

            # --- Phase 2: snapshot while stopped ---
            print("[calibrate] Snapshotting EE RAM (car stopped)...")
            snap_stopped = reader.snapshot_ee_ram()

            # --- Phase 3: accelerate ---
            print(f"[calibrate] Accelerating for {ACCEL_TIME}s...")
            gamepad.hold_button(e.BTN_SOUTH)       # Cross = accelerate
            gamepad.send(steering=0.0, throttle=1.0, brake=0.0)
            time.sleep(ACCEL_TIME)

            # --- Phase 4: snapshot while moving ---
            print("[calibrate] Snapshotting EE RAM (car moving)...")
            snap_moving = reader.snapshot_ee_ram()

            # --- Phase 5: diff → "increased" ---
            results = NFSU2MemoryReader.diff_scan(
                snap_stopped, snap_moving, "increased", candidates
            )
            candidates = [addr for addr, _, _ in results]
            prev_label = "all" if cycle == 1 else "prev"
            print(f"[calibrate] Filter 'increased': {prev_label} → {len(candidates):,} candidates")

            # Release accelerate
            gamepad.release_button(e.BTN_SOUTH)
            gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates — skipping further cycles.")
                break

            # --- Phase 6: stop the car again ---
            print(f"[calibrate] Braking for {BRAKE_TIME}s...")
            gamepad.hold_button(e.BTN_WEST)
            gamepad.send(steering=0.0, throttle=0.0, brake=1.0)
            time.sleep(BRAKE_TIME)
            gamepad.release_button(e.BTN_WEST)
            gamepad.send(steering=0.0, throttle=0.0, brake=0.0)
            print(f"[calibrate] Coasting to zero for {COAST_TIME}s...")
            time.sleep(COAST_TIME)

            # --- Phase 7: snapshot while stopped again ---
            print("[calibrate] Snapshotting EE RAM (car stopped again)...")
            snap_stopped2 = reader.snapshot_ee_ram()

            # --- Phase 8: diff → "decreased" ---
            results = NFSU2MemoryReader.diff_scan(
                snap_moving, snap_stopped2, "decreased", candidates
            )
            candidates = [addr for addr, _, _ in results]
            print(f"[calibrate] Filter 'decreased': → {len(candidates):,} candidates")

            if len(candidates) <= TARGET_CANDIDATES:
                print(f"[calibrate] Reached ≤{TARGET_CANDIDATES} candidates.")
                break

        # --- Release all input ---
        try:
            gamepad.release_button(e.BTN_SOUTH)
            gamepad.release_button(e.BTN_WEST)
        except Exception:
            pass
        gamepad.send(steering=0.0, throttle=0.0, brake=0.0)

        if not candidates:
            print("\n[calibrate] ERROR: No candidates survived. Try re-running calibration.")
            return

        # --- Auto-verify candidates ---
        print(f"\n[calibrate] Verifying {len(candidates):,} candidates...")
        scored = _score_candidates(reader, candidates)

        # --- Display results ---
        print(f"\n[calibrate] ── Results ({'top 20' if len(scored) > 20 else 'all'}) ──")
        print(f"  {'Score':>5}  {'Speed Addr':>14}  {'Struct Base':>14}  "
              f"{'Speed':>8}  {'RPM':>8}  {'Gear':>4}  {'Pos X':>10}")
        print(f"  {'─' * 5}  {'─' * 14}  {'─' * 14}  {'─' * 8}  {'─' * 8}  {'─' * 4}  {'─' * 10}")
        for entry in scored[:20]:
            print(
                f"  {entry['score']:5d}  "
                f"0x{entry['speed_addr']:08X}  "
                f"0x{entry['struct_base']:08X}  "
                f"{entry['speed_ms']:8.2f}  "
                f"{entry['rpm']:8.0f}  "
                f"{entry['gear']:4d}  "
                f"{entry['pos_x']:10.2f}"
            )

        best = scored[0]
        if best["score"] >= 4:
            struct_addr = best["struct_base"]
            print(f"\n[calibrate] ✓ Best match: vehicle struct @ PS2 0x{struct_addr:08X} "
                  f"(score {best['score']}/6)")

            # Save to calibration file
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIBRATION_FILE, "w") as f:
                json.dump({"vehicle_struct_addr": f"0x{struct_addr:08X}"}, f, indent=2)
            print(f"[calibrate] Saved to {CALIBRATION_FILE}")
            print(f"[calibrate] Use with:  python main.py --telemetry")
        else:
            print(f"\n[calibrate] ⚠ No high-confidence match (best score: {best['score']}/6).")
            print("[calibrate] Try running calibration again, or use --snap/--scan-diff manually.")

    finally:
        reader.close()
        gamepad.close()
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
            print("[rocm-racer] PCSX2 terminated.")


def _score_candidates(
    reader: "NFSU2MemoryReader",
    candidates: list[int],
) -> list[dict]:
    """Score candidate speed addresses by reading surrounding struct fields.

    Each candidate is assumed to be the absolute speed float (+0x24).
    The struct base is candidate - 0x24.  We read position, RPM, and gear
    at the documented offsets and check if they look reasonable.

    Returns a list of dicts sorted by score (descending).
    """
    import math
    results = []

    for speed_addr in candidates:
        struct_base = speed_addr - 0x24
        if struct_base < 0:
            continue

        score = 0
        try:
            speed_ms = reader._read_f32(speed_addr)
            pos_x = reader._read_f32(struct_base + 0x00)
            pos_y = reader._read_f32(struct_base + 0x04)
            pos_z = reader._read_f32(struct_base + 0x08)
            rpm = reader._read_f32(struct_base + 0x1A4)
            gear = reader._read_i32(struct_base + 0x1B0)
        except (RuntimeError, OSError):
            continue

        # Speed should be a reasonable non-negative value (0–100 m/s = 0–360 km/h)
        if math.isfinite(speed_ms) and 0.0 <= speed_ms <= 100.0:
            score += 1

        # Position should be finite and at least one axis non-zero
        if all(math.isfinite(v) for v in (pos_x, pos_y, pos_z)):
            score += 1
            if any(abs(v) > 1.0 for v in (pos_x, pos_y, pos_z)):
                score += 1

        # RPM should be in a car-engine range
        if math.isfinite(rpm) and 0.0 <= rpm <= 15000.0:
            score += 1

        # Gear should be a small non-negative integer
        if 0 <= gear <= 6:
            score += 1
            # Bonus: gear > 0 while engine is running
            if gear >= 1 and rpm > 0:
                score += 1

        results.append({
            "speed_addr": speed_addr,
            "struct_base": struct_base,
            "score": score,
            "speed_ms": speed_ms,
            "pos_x": pos_x,
            "rpm": rpm,
            "gear": gear,
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

    vehicle_addr = args.vehicle_addr
    if vehicle_addr is None and CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            cal = json.load(f)
        vehicle_addr = int(cal["vehicle_struct_addr"], 0)
        print(f"[rocm-racer] Auto-loaded vehicle address from {CALIBRATION_FILE}")
    if vehicle_addr is None:
        print(
            "[rocm-racer] ERROR: --telemetry requires --vehicle-addr 0xADDR\n"
            "  or a saved calibration file (run --calibrate first).",
            file=sys.stderr,
        )
        sys.exit(1)

    offsets = TelemetryOffsets(vehicle_struct_addr=vehicle_addr)
    reader = NFSU2MemoryReader(offsets=offsets)

    if not args.no_launch:
        write_controller_db()
        from memory_readers.virtual_gamepad import VirtualGamepad
        gamepad = VirtualGamepad()
        gamepad.open()
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    reader.open()
    backend_info = f"backend={reader._backend}"
    if reader.ee_base is not None:
        backend_info += f", EE base=0x{reader.ee_base:016x}"
    print(
        f"[rocm-racer] Telemetry logging started "
        f"(vehicle struct @ PS2 0x{vehicle_addr:08X}, {backend_info})"
    )
    print("[rocm-racer] Press Ctrl-C to stop.\n")

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
