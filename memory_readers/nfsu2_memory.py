from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from memory_readers.pine_client import PINEClient


# ---------------------------------------------------------------------------
# EE RAM constants
# ---------------------------------------------------------------------------
_EE_RAM_SIZE = 0x0200_0000  # 32 MB PS2 EE physical RAM


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
    """PCSX2 memory reader for NFSU2 vehicle telemetry via PINE IPC.

    Reads EE memory directly through PCSX2's built-in IPC server.
    No ptrace, no ``/proc/pid/mem``, no kernel permission changes.
    Enable PINE in ``~/.config/PCSX2/inis/PCSX2.ini``:
      ``EnablePINE = true``
    """

    def __init__(
        self,
        offsets: TelemetryOffsets | None = None,
        pine_socket: str | None = None,
    ) -> None:
        self.offsets = offsets or TelemetryOffsets()
        self.pine_socket = pine_socket
        self._pine: PINEClient | None = None

    # ----- lifecycle -----

    def open(self) -> None:
        if self._pine is not None:
            return
        pine = PINEClient(socket_path=self.pine_socket)
        pine.connect()
        self._pine = pine
        print("[rocm-racer] Connected to PCSX2 via PINE IPC.")

    def close(self) -> None:
        if self._pine is not None:
            self._pine.close()
            self._pine = None

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
        print("[rocm-racer] Reading 32 MB EE RAM via PINE (this may take a moment)...")
        return self._pine.read_bulk(0, _EE_RAM_SIZE)

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

    def _read_f32(self, ps2_addr: int) -> float:
        return self._pine.read_f32(ps2_addr)

    def _read_i32(self, ps2_addr: int) -> int:
        return self._pine.read_i32(ps2_addr)


__all__ = ["NFSU2MemoryReader", "TelemetryOffsets", "TelemetrySample"]
