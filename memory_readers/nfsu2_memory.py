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
    """Vehicle struct offsets discovered by ``--calibrate`` for NFSU2 on PS2 (SLUS-21065).

    PC RenderWare offsets do NOT map 1:1 to the PS2 build. All offsets here
    must be discovered empirically via differential scanning + quaternion
    anchoring and are stored in ``saves/calibration.json``.

    ``static_pointer_addr``: PS2 address (0x003X–0x005X) pointing to the
    dynamic vehicle struct. Auto-dereferenced on every ``read_telemetry()``.

    ``vehicle_struct_addr``: direct fallback / manual override.

    All struct offsets default to 0 (uncalibrated). Run ``--calibrate`` first.
    """

    static_pointer_addr: int = 0x0000_0000  # discovered by --calibrate
    vehicle_struct_addr: int = 0x0000_0000  # manual override / debugging

    # Struct member offsets — discovered empirically, relative to struct base.
    # Defaults are 0 (uncalibrated). Values are written to calibration.json.
    absolute_speed_ms: int = 0   # Float32  scalar speed (m/s or engine units)
    rotation_x: int = 0          # Float32  QuatRot[0]
    rotation_y: int = 0          # Float32  QuatRot[1]
    rotation_z: int = 0          # Float32  QuatRot[2]
    rotation_w: int = 0          # Float32  QuatRot[3]
    position_x: int = 0          # Float32  lateral (meters)
    position_y: int = 0          # Float32  vertical (meters)
    position_z: int = 0          # Float32  longitudinal (meters)
    velocity_x: int = 0          # Float32  lateral m/s
    velocity_y: int = 0          # Float32  vertical m/s
    velocity_z: int = 0          # Float32  longitudinal m/s
    # speed_unit: stored in calibration.json as string "m/s" or "km/h"


@dataclass(frozen=True)
class TelemetrySample:
    """One frame of vehicle telemetry."""

    position: tuple[float, float, float]   # X, Y, Z  (meters)
    velocity: tuple[float, float, float]   # X, Y, Z  (m/s)
    speed_ms: float                        # absolute scalar speed (m/s or engine units)
    speed_kph: float                       # speed_ms × 3.6 (approximate if unit unknown)
    rotation: tuple[float, float, float, float]  # QuatRot x, y, z, w

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
            f"Rot: ({self.rotation[0]:6.3f}, {self.rotation[1]:6.3f}, "
            f"{self.rotation[2]:6.3f}, {self.rotation[3]:6.3f})"
        )


class NFSU2MemoryReader:
    """PCSX2 memory reader for NFSU2 vehicle telemetry via PINE IPC.

    Reads EE memory directly through PCSX2's built-in IPC server.
    No ptrace, no ``/proc/pid/mem``, no kernel permission changes.
    Enable PINE in ``~/.config/PCSX2/inis/PCSX2.ini``:
      ``EnablePINE = true``

    Struct offsets are discovered empirically by ``--calibrate`` and loaded
    from ``saves/calibration.json`` at ``open()`` time.  PC RenderWare offsets
    do NOT map to the PS2 build and must not be assumed.
    """

    # Default calibration file location (can be overridden)
    DEFAULT_CALIBRATION_FILE = (
        __import__("pathlib").Path(__file__).parent.parent / "saves" / "calibration.json"
    )

    def __init__(
        self,
        offsets: TelemetryOffsets | None = None,
        pine_socket: str | None = None,
        calibration_file: str | None = None,
    ) -> None:
        self.offsets = offsets or TelemetryOffsets()
        self.pine_socket = pine_socket
        self._cal_file = (
            __import__("pathlib").Path(calibration_file)
            if calibration_file
            else self.DEFAULT_CALIBRATION_FILE
        )
        self._pine: PINEClient | None = None
        # Discovered offsets loaded from calibration.json
        self._speed_offset: int = self.offsets.absolute_speed_ms
        self._pos_offsets: tuple[int, int, int] = (
            self.offsets.position_x, self.offsets.position_y, self.offsets.position_z
        )
        self._vel_offsets: tuple[int, int, int] = (
            self.offsets.velocity_x, self.offsets.velocity_y, self.offsets.velocity_z
        )
        self._rot_offsets: tuple[int, int, int, int] = (
            self.offsets.rotation_x, self.offsets.rotation_y,
            self.offsets.rotation_z, self.offsets.rotation_w
        )
        self._speed_unit: str = "unknown"

    # ----- lifecycle -----

    def open(self) -> None:
        if self._pine is not None:
            return
        pine = PINEClient(socket_path=self.pine_socket)
        pine.connect()
        self._pine = pine
        print("[rocm-racer] Connected to PCSX2 via PINE IPC.")
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load discovered offsets from calibration.json if present."""
        import json
        if not self._cal_file.exists():
            return
        try:
            with open(self._cal_file) as f:
                cal = json.load(f)

            def _hex(key: str, default: int = 0) -> int:
                val = cal.get(key, default)
                return int(val, 0) if isinstance(val, str) else int(val)

            # Override offsets with calibration values if present
            if "speed_offset" in cal:
                self._speed_offset = _hex("speed_offset")
            if all(k in cal for k in ("pos_x_offset", "pos_y_offset", "pos_z_offset")):
                self._pos_offsets = (
                    _hex("pos_x_offset"), _hex("pos_y_offset"), _hex("pos_z_offset")
                )
            if all(k in cal for k in ("vel_x_offset", "vel_y_offset", "vel_z_offset")):
                self._vel_offsets = (
                    _hex("vel_x_offset"), _hex("vel_y_offset"), _hex("vel_z_offset")
                )
            if all(k in cal for k in ("rot_x_offset", "rot_y_offset", "rot_z_offset", "rot_w_offset")):
                self._rot_offsets = (
                    _hex("rot_x_offset"), _hex("rot_y_offset"),
                    _hex("rot_z_offset"), _hex("rot_w_offset")
                )
            if "speed_unit" in cal:
                self._speed_unit = cal["speed_unit"]

            # Also update static/direct address from calibration
            if "static_pointer_addr" in cal:
                self.offsets = TelemetryOffsets(
                    static_pointer_addr=_hex("static_pointer_addr"),
                    vehicle_struct_addr=_hex("vehicle_struct_addr", 0),
                )
            elif "vehicle_struct_addr" in cal:
                self.offsets = TelemetryOffsets(
                    vehicle_struct_addr=_hex("vehicle_struct_addr"),
                )

            print(f"[rocm-racer] Calibration loaded from {self._cal_file}")
        except Exception as exc:
            print(f"[rocm-racer] Warning: could not load calibration: {exc}")

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

    def resolve_vehicle_base(self) -> int:
        """Dereference static pointer to get current dynamic vehicle struct base.

        Falls back to ``vehicle_struct_addr`` if no pointer is configured.
        Raises ``RuntimeError`` if uncalibrated.
        """
        self.open()
        o = self.offsets
        if o.static_pointer_addr != 0:
            raw = self._pine.read_i32(o.static_pointer_addr) & 0xFFFFFFFF
            return raw & 0x1FFFFFFF
        elif o.vehicle_struct_addr != 0:
            return o.vehicle_struct_addr
        else:
            raise RuntimeError(
                "Vehicle struct address is uncalibrated.\n"
                "Run:  python main.py --calibrate"
            )

    def is_calibrated(self) -> bool:
        """Return True if offsets are loaded and struct address is known."""
        o = self.offsets
        return (o.static_pointer_addr != 0 or o.vehicle_struct_addr != 0) and self._speed_offset != 0

    def read_telemetry(self) -> TelemetrySample:
        self.open()
        if not self.is_calibrated():
            raise RuntimeError(
                "Vehicle struct address is uncalibrated.\n"
                "Run:  python main.py --calibrate"
            )
        base = self.resolve_vehicle_base()
        speed_raw = self._read_f32(base + self._speed_offset)
        # If unit is known km/h, convert to m/s; otherwise treat as m/s
        if self._speed_unit == "km/h":
            speed_ms = speed_raw / 3.6
            speed_kph = speed_raw
        else:
            speed_ms = speed_raw
            speed_kph = speed_raw * 3.6

        return TelemetrySample(
            position=(
                self._read_f32(base + self._pos_offsets[0]),
                self._read_f32(base + self._pos_offsets[1]),
                self._read_f32(base + self._pos_offsets[2]),
            ),
            velocity=(
                self._read_f32(base + self._vel_offsets[0]),
                self._read_f32(base + self._vel_offsets[1]),
                self._read_f32(base + self._vel_offsets[2]),
            ),
            speed_ms=speed_ms,
            speed_kph=speed_kph,
            rotation=(
                self._read_f32(base + self._rot_offsets[0]),
                self._read_f32(base + self._rot_offsets[1]),
                self._read_f32(base + self._rot_offsets[2]),
                self._read_f32(base + self._rot_offsets[3]),
            ),
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
