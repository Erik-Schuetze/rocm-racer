from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TelemetryOffsets:
    """Placeholder offsets relative to the resolved PCSX2 module base."""

    module_name_hint: str = "pcsx2"
    speed_kph: int = 0x0010_0000
    position_x: int = 0x0010_0004
    position_y: int = 0x0010_0008
    position_z: int = 0x0010_000C
    rotation_x: int = 0x0010_0010
    rotation_y: int = 0x0010_0014
    rotation_z: int = 0x0010_0018
    track_progress: int = 0x0010_001C
    reverse_flag: int = 0x0010_0020
    wall_collision_flag: int = 0x0010_0021


@dataclass(frozen=True)
class TelemetrySample:
    speed_kph: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    track_progress: float
    reverse_flag: bool
    wall_collision_flag: bool

    def as_observation(self) -> np.ndarray:
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


class NFSU2MemoryReader:
    """Linux-compatible PCSX2 process memory reader for NFSU2 telemetry."""

    def __init__(
        self,
        pid: int | None = None,
        offsets: TelemetryOffsets | None = None,
        process_names: Iterable[str] = ("pcsx2-qt", "pcsx2", "PCSX2"),
    ) -> None:
        self.pid = pid
        self.offsets = offsets or TelemetryOffsets()
        self.process_names = tuple(process_names)
        self.mem_fd: int | None = None
        self.module_base: int | None = None

    def open(self) -> None:
        if self.mem_fd is not None:
            return

        self.pid = self.pid or self._detect_pid()
        self.module_base = self._resolve_module_base()
        self.mem_fd = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY)

    def close(self) -> None:
        if self.mem_fd is None:
            return
        os.close(self.mem_fd)
        self.mem_fd = None

    def __enter__(self) -> "NFSU2MemoryReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read_telemetry(self) -> TelemetrySample:
        self.open()
        if self.module_base is None:
            raise RuntimeError("PCSX2 module base is unresolved.")

        offsets = self.offsets
        return TelemetrySample(
            speed_kph=self._read_f32(offsets.speed_kph),
            position=(
                self._read_f32(offsets.position_x),
                self._read_f32(offsets.position_y),
                self._read_f32(offsets.position_z),
            ),
            rotation=(
                self._read_f32(offsets.rotation_x),
                self._read_f32(offsets.rotation_y),
                self._read_f32(offsets.rotation_z),
            ),
            track_progress=self._read_f32(offsets.track_progress),
            reverse_flag=bool(self._read_u8(offsets.reverse_flag)),
            wall_collision_flag=bool(self._read_u8(offsets.wall_collision_flag)),
        )

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

    def _resolve_module_base(self) -> int:
        if self.pid is None:
            raise RuntimeError("Cannot resolve module base without a process id.")

        maps_path = f"/proc/{self.pid}/maps"
        with open(maps_path, "r", encoding="utf-8") as maps_file:
            for line in maps_file:
                if self.offsets.module_name_hint not in line:
                    continue

                address_range = line.split(maxsplit=1)[0]
                start_address, _ = address_range.split("-", maxsplit=1)
                return int(start_address, 16)

        raise RuntimeError(
            f"Unable to resolve a module base containing '{self.offsets.module_name_hint}' "
            f"for pid {self.pid}."
        )

    def _read_f32(self, offset: int) -> float:
        return struct.unpack("<f", self._read_bytes(offset, 4))[0]

    def _read_u8(self, offset: int) -> int:
        return struct.unpack("<B", self._read_bytes(offset, 1))[0]

    def _read_bytes(self, offset: int, size: int) -> bytes:
        if self.mem_fd is None or self.module_base is None:
            raise RuntimeError("Memory reader is not open.")

        absolute_address = self.module_base + offset
        data = os.pread(self.mem_fd, size, absolute_address)
        if len(data) != size:
            raise RuntimeError(
                f"Short read from /proc/{self.pid}/mem at 0x{absolute_address:x}: "
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
