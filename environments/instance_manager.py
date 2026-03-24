"""
Multi-instance PCSX2 management for parallel RL training.

Each PCSX2 instance gets an isolated config/runtime directory so that
PINE sockets, emulogs, and savestates don't collide.  The InstanceManager
handles the full lifecycle: config preparation → staggered launch →
readiness polling → window tiling → cleanup.

Usage:
    mgr = InstanceManager(num_envs=4, iso=iso_path, statefile=state_path)
    instances = mgr.launch_all(turbo=True)
    mgr.tile_windows()
    ...
    mgr.cleanup()
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PCSX2_BIN = Path("/usr/bin/pcsx2-qt")
PCSX2_CONFIG_DIR = Path.home() / ".config" / "PCSX2"
INSTANCE_BASE = Path("/tmp/rocm-racer")


@dataclass
class InstanceConfig:
    """Per-instance paths and handles."""

    instance_id: int
    config_dir: Path        # XDG_CONFIG_HOME value
    runtime_dir: Path       # XDG_RUNTIME_DIR value → PINE socket here
    pine_socket: str        # Full path to pcsx2.sock
    emulog: Path            # Per-instance emulog
    pcsx2_proc: Optional[subprocess.Popen] = None
    pcsx2_pid: Optional[int] = None


# Directories in ~/.config/PCSX2 that should be symlinked (shared, read-only-ish)
_SYMLINK_DIRS = ["bios", "cache", "inputprofiles", "cheats", "covers", "patches",
                 "resources", "textures", "shaders"]

# Directories that must be per-instance (written at runtime)
_INSTANCE_DIRS = ["savestates", "logs", "sstates", "snaps"]


class InstanceManager:
    """Manages N isolated PCSX2 instances for parallel training."""

    def __init__(
        self,
        num_envs: int,
        iso: Path,
        statefile: Path | None = None,
        base_dir: Path = INSTANCE_BASE,
    ) -> None:
        self.num_envs = num_envs
        self.iso = iso
        self.statefile = statefile
        self.base_dir = base_dir
        self.instances: list[InstanceConfig] = []
        self._log_handles: list = []

    # ── config preparation ─────────────────────────────────────────────────

    def prepare_instance(self, i: int) -> InstanceConfig:
        """Create isolated config/runtime dirs for instance *i*."""
        inst_dir = self.base_dir / f"env-{i}"
        config_home = inst_dir / "config"
        runtime_dir = inst_dir / "runtime"
        pcsx2_cfg = config_home / "PCSX2"

        # Clean previous run
        if inst_dir.exists():
            shutil.rmtree(inst_dir)

        # Create dir structure
        pcsx2_cfg.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)
        for d in _INSTANCE_DIRS:
            (pcsx2_cfg / d).mkdir(exist_ok=True)

        # Copy inis/ (small, contains PCSX2.ini which must be per-instance)
        src_inis = PCSX2_CONFIG_DIR / "inis"
        if src_inis.exists():
            shutil.copytree(src_inis, pcsx2_cfg / "inis")
        else:
            (pcsx2_cfg / "inis").mkdir()

        # Ensure PINE is enabled in the copy
        ini_path = pcsx2_cfg / "inis" / "PCSX2.ini"
        if ini_path.exists():
            self._ensure_pine_enabled(ini_path)

        # Symlink shared directories
        for d in _SYMLINK_DIRS:
            src = PCSX2_CONFIG_DIR / d
            dst = pcsx2_cfg / d
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        # Copy controller DB
        src_db = PCSX2_CONFIG_DIR / "game_controller_db.txt"
        if src_db.exists():
            shutil.copy2(src_db, pcsx2_cfg / "game_controller_db.txt")

        pine_socket = str(runtime_dir / "pcsx2.sock")
        emulog = pcsx2_cfg / "logs" / "emulog.txt"

        cfg = InstanceConfig(
            instance_id=i,
            config_dir=config_home,
            runtime_dir=runtime_dir,
            pine_socket=pine_socket,
            emulog=emulog,
        )
        return cfg

    @staticmethod
    def _ensure_pine_enabled(ini_path: Path) -> None:
        """Make sure EnablePINE = true in the copied PCSX2.ini."""
        import re
        text = ini_path.read_text()
        if re.search(r"EnablePINE\s*=\s*true", text):
            return
        text = re.sub(
            r"(EnablePINE\s*=\s*)\S+",
            r"\g<1>true",
            text,
        )
        ini_path.write_text(text)

    # ── launch ─────────────────────────────────────────────────────────────

    def launch_instance(
        self,
        cfg: InstanceConfig,
        turbo: bool = False,
        gamepad_device: str | None = None,
    ) -> None:
        """Launch one PCSX2 process with isolated env vars."""
        # Patch speed scalar in per-instance INI if turbo requested
        if turbo:
            ini_path = Path(cfg.config_dir) / "PCSX2" / "inis" / "PCSX2.ini"
            if ini_path.exists():
                import re
                text = ini_path.read_text()
                text = re.sub(
                    r"(?m)^(NominalScalar\s*=\s*)[\d.]+",
                    r"\g<1>2",
                    text,
                )
                ini_path.write_text(text)

        cmd = [str(PCSX2_BIN), "-nogui", "-batch"]
        if self.statefile is not None:
            cmd += ["-statefile", str(self.statefile)]
        cmd += ["--", str(self.iso)]

        env = {
            **os.environ,
            "XDG_CONFIG_HOME": str(cfg.config_dir),
            "XDG_RUNTIME_DIR": str(cfg.runtime_dir),
        }
        # Restrict SDL to only see this instance's gamepad
        if gamepad_device:
            env["SDL_JOYSTICK_DEVICE"] = gamepad_device

        log_path = self.base_dir / f"env-{cfg.instance_id}" / "pcsx2.log"
        log_fh = open(log_path, "w")
        self._log_handles.append(log_fh)

        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        cfg.pcsx2_proc = proc
        cfg.pcsx2_pid = proc.pid
        print(f"[instance-{cfg.instance_id}] Launched PCSX2 (PID={proc.pid})")

    def wait_for_instance(
        self,
        cfg: InstanceConfig,
        timeout: float = 30.0,
        poll: float = 0.25,
        post_ready_delay: float = 2.0,
    ) -> None:
        """Block until this instance's PCSX2 is ready."""
        deadline = time.monotonic() + timeout
        print(f"[instance-{cfg.instance_id}] Waiting for PCSX2 ready (timeout={timeout}s)...")

        while time.monotonic() < deadline:
            try:
                text = cfg.emulog.read_text(errors="replace")
                if "Opened gamepad" in text:
                    print(f"[instance-{cfg.instance_id}] PCSX2 ready — gamepad connected.")
                    break
            except OSError:
                pass
            time.sleep(poll)
        else:
            print(
                f"[instance-{cfg.instance_id}] WARNING: readiness marker not found "
                f"within {timeout}s — proceeding anyway."
            )

        time.sleep(post_ready_delay)

    def launch_all(self, turbo: bool = False) -> list[InstanceConfig]:
        """Prepare, launch, and wait for all instances (staggered)."""
        self.instances = []
        for i in range(self.num_envs):
            cfg = self.prepare_instance(i)
            self.instances.append(cfg)

        # Staggered launch: gamepad creation happens in _run_train,
        # but PCSX2 launch + readiness wait is sequential here
        for cfg in self.instances:
            self.launch_instance(cfg, turbo=turbo)
            self.wait_for_instance(cfg)

        return self.instances

    # ── window tiling ──────────────────────────────────────────────────────

    def tile_windows(self) -> None:
        """Arrange PCSX2 windows in a grid on the primary monitor."""
        try:
            raw = subprocess.check_output(
                ["hyprctl", "monitors", "-j"], timeout=2.0
            )
            monitors = json.loads(raw)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("[instance-mgr] hyprctl not available — skipping window tiling")
            return

        if not monitors:
            return

        mon = monitors[0]
        mon_w, mon_h = mon["width"], mon["height"]
        mon_x, mon_y = mon.get("x", 0), mon.get("y", 0)

        n = len(self.instances)
        if n <= 0:
            return

        # Determine grid layout
        if n <= 2:
            cols, rows = n, 1
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 2

        cell_w = mon_w // cols
        cell_h = mon_h // rows

        # Get all PCSX2 windows and map by PID
        try:
            raw = subprocess.check_output(
                ["hyprctl", "clients", "-j"], timeout=2.0
            )
            clients = json.loads(raw)
        except Exception:
            return

        pid_to_addr: dict[int, str] = {}
        for c in clients:
            if c.get("class") == "pcsx2-qt":
                pid_to_addr[c.get("pid", 0)] = c.get("address", "")

        for idx, cfg in enumerate(self.instances):
            addr = pid_to_addr.get(cfg.pcsx2_pid or 0, "")
            if not addr:
                continue

            col = idx % cols
            row = idx // cols
            x = mon_x + col * cell_w
            y = mon_y + row * cell_h

            try:
                subprocess.run(
                    ["hyprctl", "dispatch", "movewindowpixel",
                     f"exact {x} {y}", f"address:{addr}"],
                    timeout=2.0, capture_output=True,
                )
                subprocess.run(
                    ["hyprctl", "dispatch", "resizewindowpixel",
                     f"exact {cell_w} {cell_h}", f"address:{addr}"],
                    timeout=2.0, capture_output=True,
                )
            except Exception:
                pass

        print(f"[instance-mgr] Tiled {n} windows in {cols}×{rows} grid ({cell_w}×{cell_h})")

    # ── cleanup ────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Terminate all PCSX2 processes and close log handles."""
        for cfg in self.instances:
            if cfg.pcsx2_proc is not None:
                try:
                    cfg.pcsx2_proc.terminate()
                    cfg.pcsx2_proc.wait(timeout=5.0)
                except Exception:
                    pass
                print(f"[instance-{cfg.instance_id}] PCSX2 terminated")

        for fh in self._log_handles:
            try:
                fh.close()
            except Exception:
                pass
        self._log_handles.clear()

        # Don't remove temp dirs — useful for debugging.
        # shutil.rmtree(self.base_dir, ignore_errors=True)
