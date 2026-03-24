from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from memory_readers.pine_client import PINEClient


# ---------------------------------------------------------------------------
# EE RAM constants
# ---------------------------------------------------------------------------
_EE_RAM_SIZE = 0x0200_0000  # 32 MB PS2 EE physical RAM
_PS2_ADDR_MASK = 0x1FFF_FFFF  # kseg0/kseg1 mirror mask

_EMULOG_PATH = Path.home() / ".config" / "PCSX2" / "logs" / "emulog.txt"
_EE_MEM_RE = re.compile(
    r"EE Main Memory\s+@ (0x[0-9A-Fa-f]+)\s*->"
)


# ---------------------------------------------------------------------------
# Telemetry layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetryOffsets:
    """Vehicle struct offsets from docs/underground-2-telemetry-memory-map.md.

    ``vehicle_struct_addr`` is the PS2-side address where the vehicle struct
    lives in EE RAM.  It must be discovered via ``--scan`` and set here (or
    passed at construction time).  A value of 0 means *uncalibrated*.
    """

    vehicle_struct_addr: int = 0x0000_0000  # UNCALIBRATED — see docs

    # Struct member offsets (relative to vehicle_struct_addr)
    position_x: int = 0x00        # Float32  lateral (meters)
    position_y: int = 0x04        # Float32  vertical (meters)
    position_z: int = 0x08        # Float32  longitudinal (meters)
    velocity_x: int = 0x10        # Float32  lateral m/s
    velocity_y: int = 0x14        # Float32  vertical m/s
    velocity_z: int = 0x18        # Float32  longitudinal m/s
    absolute_speed_ms: int = 0x24 # Float32  scalar speed m/s
    rotation_x: int = 0x30        # Float32  pitch
    rotation_y: int = 0x34        # Float32  yaw / heading
    rotation_z: int = 0x38        # Float32  roll
    engine_rpm: int = 0x1A4       # Float32
    current_gear: int = 0x1B0     # Int32


@dataclass(frozen=True)
class TelemetrySample:
    """One frame of vehicle telemetry."""

    # --- documented fields ---
    position: tuple[float, float, float]   # X, Y, Z  (meters)
    velocity: tuple[float, float, float]   # X, Y, Z  (m/s)
    speed_ms: float                        # absolute scalar speed (m/s)
    speed_kph: float                       # speed_ms × 3.6
    rotation: tuple[float, float, float]   # pitch, yaw, roll
    engine_rpm: float
    current_gear: int

    # --- env-compat stubs (need separate calibration) ---
    track_progress: float = 0.0
    reverse_flag: bool = False
    wall_collision_flag: bool = False

    def as_observation(self) -> np.ndarray:
        """8-D telemetry vector consumed by PCSX2RacerEnv."""
        return np.asarray(
            [
                self.speed_kph,
                self.position[0],
                self.position[1],
                self.position[2],
                self.rotation[0],
                self.rotation[1],
                self.rotation[2],
                self.track_progress,
            ],
            dtype=np.float32,
        )

    def fmt(self) -> str:
        """Human-readable one-liner for dashboard logging."""
        return (
            f"Speed: {self.speed_kph:6.1f} km/h  "
            f"Pos: ({self.position[0]:9.2f}, {self.position[1]:7.2f}, {self.position[2]:9.2f})  "
            f"Vel: ({self.velocity[0]:7.2f}, {self.velocity[1]:7.2f}, {self.velocity[2]:7.2f})  "
            f"Rot: ({self.rotation[0]:7.3f}, {self.rotation[1]:7.3f}, {self.rotation[2]:7.3f})  "
            f"RPM: {self.engine_rpm:7.0f}  Gear: {self.current_gear}"
        )


class NFSU2MemoryReader:
    """PCSX2 memory reader for NFSU2 vehicle telemetry.

    **Primary backend — PINE IPC** (Unix socket):
      Reads EE memory directly through PCSX2's built-in IPC server.
      No ptrace, no ``/proc/pid/mem``, no kernel permission changes.
      Enable PINE in ``~/.config/PCSX2/inis/PCSX2.ini``:
        ``EnablePINE = true``

    **Fallback — /proc/pid/mem** (requires matching UID + ptrace access):
      Used when PINE is unavailable.  Needs ``ptrace_scope=0`` and
      ``suid_dumpable=1`` on systems where PCSX2 has filesystem caps.
    """

    def __init__(
        self,
        pid: int | None = None,
        offsets: TelemetryOffsets | None = None,
        process_names: Iterable[str] = ("pcsx2-qt", "pcsx2", "PCSX2"),
        pine_socket: str | None = None,
    ) -> None:
        self.pid = pid
        self.offsets = offsets or TelemetryOffsets()
        self.process_names = tuple(process_names)
        self.pine_socket = pine_socket

        # Backend state — exactly one of these will be active after open()
        self._pine: PINEClient | None = None
        self.mem_fd: int | None = None
        self.ee_base: int | None = None
        self._backend: str | None = None  # "pine" or "procmem"

    # ----- lifecycle -----

    def open(self) -> None:
        if self._backend is not None:
            return

        # --- Try PINE first ---
        try:
            pine = PINEClient(socket_path=self.pine_socket)
            pine.connect()
            self._pine = pine
            self._backend = "pine"
            print("[rocm-racer] Connected to PCSX2 via PINE IPC (no ptrace needed).")
            return
        except (OSError, ConnectionRefusedError) as exc:
            print(f"[rocm-racer] PINE IPC unavailable ({exc}), falling back to /proc/pid/mem...")

        # --- Fallback: /proc/pid/mem ---
        self.pid = self.pid or self._detect_pid()
        self.ee_base = self._resolve_ee_base()
        try:
            self.mem_fd = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY)
            self._backend = "procmem"
        except PermissionError:
            self._raise_permission_error()

    def _raise_permission_error(self) -> None:
        raise PermissionError(
            f"Cannot read /proc/{self.pid}/mem.\n"
            "Recommended fix: enable PINE IPC in PCSX2:\n"
            "  Set EnablePINE = true in ~/.config/PCSX2/inis/PCSX2.ini\n"
            "  then restart PCSX2.\n"
            "Alternative (proc/mem fallback):\n"
            "  echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope"
        ) from None

    def close(self) -> None:
        if self._pine is not None:
            self._pine.close()
            self._pine = None
        if self.mem_fd is not None:
            os.close(self.mem_fd)
            self.mem_fd = None
        self._backend = None

    def __enter__(self) -> "NFSU2MemoryReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ----- telemetry -----

    def read_telemetry(self) -> TelemetrySample:
        self.open()
        o = self.offsets

        if o.vehicle_struct_addr == 0:
            raise RuntimeError(
                "Vehicle struct address is uncalibrated (0x00000000).\n"
                "Run:  python main.py --calibrate\n"
                "See docs/underground-2-telemetry-memory-map.md for details."
            )

        base = o.vehicle_struct_addr
        speed_ms = self._read_f32(base + o.absolute_speed_ms)
        return TelemetrySample(
            position=(
                self._read_f32(base + o.position_x),
                self._read_f32(base + o.position_y),
                self._read_f32(base + o.position_z),
            ),
            velocity=(
                self._read_f32(base + o.velocity_x),
                self._read_f32(base + o.velocity_y),
                self._read_f32(base + o.velocity_z),
            ),
            speed_ms=speed_ms,
            speed_kph=speed_ms * 3.6,
            rotation=(
                self._read_f32(base + o.rotation_x),
                self._read_f32(base + o.rotation_y),
                self._read_f32(base + o.rotation_z),
            ),
            engine_rpm=self._read_f32(base + o.engine_rpm),
            current_gear=self._read_i32(base + o.current_gear),
        )

    # ----- scan helpers -----

    def scan_ee_ram(
        self,
        target_value: float,
        tolerance: float = 0.5,
    ) -> list[int]:
        """Scan the entire 32 MB EE RAM for Float32 matches.

        Returns a list of PS2-side addresses where the value matches
        ``target_value`` within ±``tolerance``.
        """
        data = self.snapshot_ee_ram()
        matches: list[int] = []
        lo = target_value - tolerance
        hi = target_value + tolerance
        for offset in range(0, _EE_RAM_SIZE - 3, 4):
            val = struct.unpack_from("<f", data, offset)[0]
            if lo <= val <= hi:
                matches.append(offset)

        return matches

    def snapshot_ee_ram(self) -> bytes:
        """Read the entire 32 MB EE RAM as raw bytes."""
        self.open()

        if self._backend == "pine":
            print("[rocm-racer] Reading 32 MB EE RAM via PINE (this may take a moment)...")
            return self._pine.read_bulk(0, _EE_RAM_SIZE)

        # /proc/pid/mem path
        if self.ee_base is None:
            raise RuntimeError("EE base is unresolved.")
        data = os.pread(self.mem_fd, _EE_RAM_SIZE, self.ee_base)
        if len(data) < _EE_RAM_SIZE:
            raise RuntimeError(
                f"Short read of EE RAM: expected {_EE_RAM_SIZE} bytes, got {len(data)}."
            )
        return data

    @staticmethod
    def diff_scan(
        old_data: bytes,
        new_data: bytes,
        filter_mode: str,
        candidates: list[int] | None = None,
    ) -> list[tuple[int, float, float]]:
        """Compare two 32 MB EE RAM snapshots and return matching addresses.

        Args:
            old_data: Reference snapshot bytes (32 MB).
            new_data: Current snapshot bytes (32 MB).
            filter_mode: One of ``changed``, ``unchanged``, ``increased``,
                ``decreased``.
            candidates: Optional list of PS2 byte offsets to restrict the
                search to.  If *None*, scans all ~8 M float32 slots.

        Returns:
            List of ``(ps2_addr, old_value, new_value)`` tuples for addresses
            that pass the filter.
        """
        old_arr = np.frombuffer(old_data, dtype=np.float32)
        new_arr = np.frombuffer(new_data, dtype=np.float32)

        if candidates is not None:
            idx = np.array([a // 4 for a in candidates], dtype=np.int64)
            # Bounds-check
            valid = idx < len(old_arr)
            idx = idx[valid]
            old_vals = old_arr[idx]
            new_vals = new_arr[idx]
        else:
            idx = np.arange(len(old_arr), dtype=np.int64)
            old_vals = old_arr
            new_vals = new_arr

        finite = np.isfinite(old_vals) & np.isfinite(new_vals)
        diff = new_vals - old_vals

        if filter_mode == "changed":
            mask = finite & (np.abs(diff) > 1e-6)
        elif filter_mode == "unchanged":
            mask = finite & (np.abs(diff) < 1e-6)
        elif filter_mode == "increased":
            mask = finite & (diff > 1e-6)
        elif filter_mode == "decreased":
            mask = finite & (diff < -1e-6)
        else:
            raise ValueError(f"Unknown filter mode: {filter_mode!r}")

        matching_idx = idx[mask]
        old_matched = old_vals[mask]
        new_matched = new_vals[mask]

        return [
            (int(i) * 4, float(o), float(n))
            for i, o, n in zip(matching_idx, old_matched, new_matched)
        ]

    # ----- internals -----

    def _detect_pid(self) -> int:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            proc_comm = self._safe_read_text(f"/proc/{pid}/comm").strip()
            proc_cmdline = self._safe_read_text(f"/proc/{pid}/cmdline").replace("\x00", " ")
            if any(name in proc_comm or name in proc_cmdline for name in self.process_names):
                return pid
        names = ", ".join(self.process_names)
        raise RuntimeError(f"Unable to find a running PCSX2 process matching: {names}")

    def _resolve_ee_base(self) -> int:
        """Resolve the host-side base address of PCSX2's EE Main Memory.

        Primary: parse the PCSX2 emulog for the ``EE Main Memory @ 0x...``
        line.  Fallback: scan ``/proc/[pid]/maps`` for the fastmem region
        (a 4 GB anonymous rw-p mapping).
        """
        # --- try emulog first ---
        try:
            text = _EMULOG_PATH.read_text(errors="replace")
            m = _EE_MEM_RE.search(text)
            if m:
                addr = int(m.group(1), 16)
                print(f"[rocm-racer] EE base from emulog: 0x{addr:016x}")
                return addr
        except OSError:
            pass

        # --- fallback: scan /proc/pid/maps for fastmem (4 GB anon rw-p) ---
        if self.pid is None:
            raise RuntimeError("Cannot resolve EE base without a PID.")
        maps_path = f"/proc/{self.pid}/maps"
        with open(maps_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2 or "rw-p" not in parts[1]:
                    continue
                addr_range = parts[0]
                start_s, end_s = addr_range.split("-", 1)
                start = int(start_s, 16)
                end = int(end_s, 16)
                size = end - start
                # Fastmem is exactly 4 GB (0x1_0000_0000 bytes)
                if size == 0x1_0000_0000:
                    print(f"[rocm-racer] EE base from fastmem mapping: 0x{start:016x}")
                    return start

        raise RuntimeError(
            "Unable to resolve PCSX2 EE base address.\n"
            "Ensure PCSX2 is running and the emulog exists at:\n"
            f"  {_EMULOG_PATH}"
        )

    def _read_f32(self, ps2_addr: int) -> float:
        if self._backend == "pine":
            return self._pine.read_f32(ps2_addr)
        return struct.unpack("<f", self._read_bytes_procmem(ps2_addr, 4))[0]

    def _read_i32(self, ps2_addr: int) -> int:
        if self._backend == "pine":
            return self._pine.read_i32(ps2_addr)
        return struct.unpack("<i", self._read_bytes_procmem(ps2_addr, 4))[0]

    def _read_u8(self, ps2_addr: int) -> int:
        if self._backend == "pine":
            return self._pine.read8(ps2_addr)
        return struct.unpack("<B", self._read_bytes_procmem(ps2_addr, 1))[0]

    def _read_bytes_procmem(self, ps2_addr: int, size: int) -> bytes:
        """Read bytes via /proc/pid/mem (fallback backend)."""
        if self.mem_fd is None or self.ee_base is None:
            raise RuntimeError("Memory reader is not open (procmem backend).")

        masked = ps2_addr & _PS2_ADDR_MASK
        absolute = self.ee_base + masked
        data = os.pread(self.mem_fd, size, absolute)
        if len(data) != size:
            raise RuntimeError(
                f"Short read from /proc/{self.pid}/mem at 0x{absolute:x}: "
                f"expected {size} bytes, got {len(data)}."
            )
        return data

    @staticmethod
    def _safe_read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except OSError:
            return ""


__all__ = ["NFSU2MemoryReader", "TelemetryOffsets", "TelemetrySample"]
