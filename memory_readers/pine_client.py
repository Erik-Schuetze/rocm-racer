"""Pure-Python client for the PCSX2 PINE IPC protocol.

PINE (PCSX2 IPC Network Extension) provides direct read/write access to PS2
EE memory over a Unix domain socket.  No ptrace, no /proc/pid/mem, no kernel
permission tweaks needed.

Enable in PCSX2:  Settings → Advanced → Enable PINE IPC
  or set ``EnablePINE = true`` in ``~/.config/PCSX2/inis/PCSX2.ini``

Protocol reference: ``pcsx2/PINE.cpp`` in the PCSX2 source tree.
"""

from __future__ import annotations

import os
import socket
import struct
from pathlib import Path


# IPC opcodes
_MSG_READ8 = 0
_MSG_READ16 = 1
_MSG_READ32 = 2
_MSG_READ64 = 3
_MSG_WRITE8 = 4
_MSG_WRITE16 = 5
_MSG_WRITE32 = 6
_MSG_WRITE64 = 7
_MSG_VERSION = 8
_MSG_SAVE_STATE = 9
_MSG_LOAD_STATE = 0xA
_MSG_TITLE = 0xB
_MSG_ID = 0xC
_MSG_STATUS = 0xF

_IPC_OK = 0x00

# PINE enforces MAX_IPC_SIZE = 650 000 and MAX_IPC_RETURN_SIZE = 450 000.
# Each Read32 command is 5 bytes (1 opcode + 4 addr), response value is 4 bytes.
# Conservative batch size to stay well within limits.
_MAX_BATCH_READ32 = 40_000  # 40k × 5 = 200 KB request, 40k × 4 + 5 = 160 KB response


def _default_socket_path() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return f"{runtime_dir}/pcsx2.sock"


class PINEClient:
    """Lightweight, zero-dependency PINE IPC client."""

    def __init__(self, socket_path: str | None = None) -> None:
        self.socket_path = socket_path or _default_socket_path()
        self._sock: socket.socket | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(self.socket_path)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def __enter__(self) -> PINEClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── single-value reads ────────────────────────────────────────────────

    def read8(self, address: int) -> int:
        return self._read(_MSG_READ8, address, "<B", 1)

    def read16(self, address: int) -> int:
        return self._read(_MSG_READ16, address, "<H", 2)

    def read32(self, address: int) -> int:
        return self._read(_MSG_READ32, address, "<I", 4)

    def read64(self, address: int) -> int:
        return self._read(_MSG_READ64, address, "<Q", 8)

    def read_f32(self, address: int) -> float:
        raw = self.read32(address)
        return struct.unpack("<f", struct.pack("<I", raw))[0]

    def read_i32(self, address: int) -> int:
        raw = self.read32(address)
        return struct.unpack("<i", struct.pack("<I", raw))[0]

    # ── batched reads ─────────────────────────────────────────────────────

    def batch_read32(self, addresses: list[int]) -> list[int]:
        """Read multiple 32-bit values in a single IPC round-trip."""
        if not addresses:
            return []

        payload = bytearray()
        for addr in addresses:
            payload += struct.pack("<BI", _MSG_READ32, addr)

        resp = self._transact(bytes(payload))
        values: list[int] = []
        off = 0
        for _ in addresses:
            values.append(struct.unpack_from("<I", resp, off)[0])
            off += 4
        return values

    def batch_read_f32(self, addresses: list[int]) -> list[float]:
        raw = self.batch_read32(addresses)
        return [struct.unpack("<f", struct.pack("<I", v))[0] for v in raw]

    # ── bulk memory read ──────────────────────────────────────────────────

    def read_bulk(self, start_address: int, size: int) -> bytes:
        """Read *size* bytes starting at *start_address* via batched Read32.

        *size* must be a multiple of 4.  Handles the batching automatically.
        """
        if size % 4 != 0:
            raise ValueError("read_bulk size must be a multiple of 4")

        n_reads = size // 4
        result = bytearray()

        for batch_start in range(0, n_reads, _MAX_BATCH_READ32):
            batch_count = min(_MAX_BATCH_READ32, n_reads - batch_start)
            addrs = [
                start_address + (batch_start + i) * 4
                for i in range(batch_count)
            ]
            values = self.batch_read32(addrs)
            for v in values:
                result += struct.pack("<I", v)

        return bytes(result)

    # ── writes ────────────────────────────────────────────────────────────

    def write32(self, address: int, value: int) -> None:
        payload = struct.pack("<BII", _MSG_WRITE32, address, value & 0xFFFFFFFF)
        self._transact(payload)

    # ── emulator commands ─────────────────────────────────────────────────

    def get_version(self) -> str:
        payload = struct.pack("<B", _MSG_VERSION)
        resp = self._transact(payload)
        return resp.rstrip(b"\x00").decode("utf-8", errors="replace")

    def get_status(self) -> int:
        """0 = Running, 1 = Paused, 2 = Shutdown."""
        payload = struct.pack("<B", _MSG_STATUS)
        resp = self._transact(payload)
        return struct.unpack_from("<I", resp, 0)[0]

    def save_state(self, slot: int) -> None:
        payload = struct.pack("<BB", _MSG_SAVE_STATE, slot)
        self._transact(payload)

    def load_state(self, slot: int) -> None:
        payload = struct.pack("<BB", _MSG_LOAD_STATE, slot)
        self._transact(payload)

    def get_game_title(self) -> str:
        payload = struct.pack("<B", _MSG_TITLE)
        resp = self._transact(payload)
        return resp.rstrip(b"\x00").decode("utf-8", errors="replace")

    def get_game_id(self) -> str:
        payload = struct.pack("<B", _MSG_ID)
        resp = self._transact(payload)
        return resp.rstrip(b"\x00").decode("utf-8", errors="replace")

    # ── internals ─────────────────────────────────────────────────────────

    def _read(self, opcode: int, address: int, fmt: str, rsize: int) -> int:
        payload = struct.pack("<BI", opcode, address)
        resp = self._transact(payload)
        return struct.unpack_from(fmt, resp, 0)[0]

    def _transact(self, payload: bytes) -> bytes:
        """Send a PINE request and return the response data (after result code)."""
        # Wire request: [total_size u32 LE][payload]
        total = 4 + len(payload)
        self._send(struct.pack("<I", total) + payload)

        # Wire response: [total_size u32 LE][result_code u8][data...]
        hdr = self._recv(4)
        resp_size = struct.unpack("<I", hdr)[0]
        body = self._recv(resp_size - 4)

        if body[0] != _IPC_OK:
            raise RuntimeError("PINE IPC command failed (result=0xFF)")

        return body[1:]  # strip result code

    def _send(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("PINE client not connected")
        self._sock.sendall(data)

    def _recv(self, size: int) -> bytes:
        if self._sock is None:
            raise RuntimeError("PINE client not connected")
        buf = bytearray()
        while len(buf) < size:
            chunk = self._sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("PINE socket closed unexpectedly")
            buf.extend(chunk)
        return bytes(buf)


__all__ = ["PINEClient"]
