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

    print(f"[rocm-racer] Launching: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_memory_map(pid: int, hint: str = "pcsx2", timeout: float = 60.0, poll: float = 0.5) -> None:
    """Block until the PCSX2 virtual memory block appears in /proc/<pid>/maps."""
    maps_path = f"/proc/{pid}/maps"
    deadline = time.monotonic() + timeout

    print(f"[rocm-racer] Waiting for PCSX2 memory map (pid={pid}, hint='{hint}')...")
    while time.monotonic() < deadline:
        try:
            with open(maps_path, "r") as f:
                if any(hint in line for line in f):
                    print("[rocm-racer] Memory map ready.")
                    return
        except OSError:
            pass
        time.sleep(poll)

    raise TimeoutError(
        f"PCSX2 (pid={pid}) memory map with hint '{hint}' not found within {timeout}s."
    )


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
    parser.add_argument("--game", choices=list(GAME_ISOS), default="nfsu2")
    parser.add_argument("--statefile", type=Path, default=DEFAULT_STATEFILE)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--no-launch", action="store_true",
                        help="Skip launching PCSX2 (assume it is already running)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso = GAME_ISOS[args.game]

    from memory_readers.keyboard_controller import KeyboardController

    pcsx2_proc: subprocess.Popen | None = None

    def _shutdown(sig, frame):
        print("\n[rocm-racer] Shutting down...")
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not args.no_launch:
        pcsx2_proc = launch_pcsx2(iso, statefile=args.statefile)
        wait_for_memory_map(pcsx2_proc.pid)

    try:
        pid = pcsx2_proc.pid if pcsx2_proc else None
        kbd = KeyboardController(pid=pid)
        print("[rocm-racer] Test mode: pressing K (Cross/✕ = accelerate). Press Ctrl-C to stop.")
        while True:
            # K is the default PCSX2 keyboard binding for Cross (✕), which
            # is the accelerate button in NFS Underground 2.
            kbd.press("k", duration=0.1)
            time.sleep(0.016)
    finally:
        if pcsx2_proc is not None:
            pcsx2_proc.terminate()


if __name__ == "__main__":
    main()
