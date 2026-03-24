from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso = ISO_MAP[args.game]

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
    #    SDL3 enumerates devices during SDL_InitSubSystem — the uinput device
    #    must already exist at that point for reliable detection.
    gamepad = VirtualGamepad()
    gamepad.open()

    # 3. Launch PCSX2 — it will find the gamepad during SDL init.
    if not args.no_launch:
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_pcsx2_ready()

    # 4. Send accelerate input and hold it for testing.
    #    The post-init delay in wait_for_pcsx2_ready ensures we send input
    #    AFTER the savestate has fully loaded and the game is running, so
    #    the game sees a clean released→pressed transition.
    print("[rocm-racer] Test mode: accelerating (Cross + right stick up). Press Ctrl-C to stop.")
    try:
        # Digital button: Cross = accelerate
        gamepad.hold_button(e.BTN_SOUTH)
        # Analog axis: right stick up = accelerate (belt-and-suspenders)
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
