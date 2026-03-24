# Need for Speed: Underground 2 - Telemetry Memory Map

> **⚠️ DEPRECATED** — This document contains **estimated** offsets that have been
> superseded by the verified research in
> [`ps2-memory-architecture-and-telemetry-extraction-for-nfs-u2.md`](ps2-memory-architecture-and-telemetry-extraction-for-nfs-u2.md).
> Use the new document as the authoritative reference. The corrected offsets are
> now in the codebase (`TelemetryOffsets` in `memory_readers/nfsu2_memory.py`).

### Technical Deep-Dive

**PCSX2 Emotion Engine Memory Architecture**
The PlayStation 2 utilizes a **32 MB** block of main system RAM managed by the Emotion Engine (EE). When running PCSX2 on Linux, this entire block is mapped into the host system's virtual memory space. To read telemetry, the Python agent must locate the dynamic base address of this EE block within `/proc/[pid]/maps`. All game-specific memory addresses are static hexadecimal offsets relative to this EE base address.

**The Vehicle Telemetry Struct**
Like most RenderWare-engine games, *Underground 2* stores player data in a contiguous C++ struct. Once the base pointer for the player's vehicle object is located in the EE memory, the individual telemetry variables (coordinates, velocity, rotation) are found at static offsets relative to that vehicle pointer. 

**Data Types and Conversion**
* **Coordinates and Vectors**: The game engine uses 32-bit floating-point numbers (`Float32`) for spatial data. The Z-axis represents longitudinal movement, the X-axis represents lateral movement, and the Y-axis represents vertical elevation.
* **Velocity vs. Absolute Speed**: The physics engine calculates velocity as 3D vectors (**m/s**). A separate derived float stores the absolute scalar speed. The value is stored in meters per second (**m/s**) and must be multiplied by **3.6** to convert to **km/h** for the agent's reward function.

**Unverified Base Pointer:** The exact static base pointer for the player's vehicle object in the PS2 NTSC-U v1.2 release is **Unverified**. While the internal struct offsets remain identical across platforms, the absolute EE memory address for the base pointer shifts depending on the specific PCSX2 build and Linux memory allocation. You must use a memory scanner on the active PCSX2 process to execute a pointer scan (searching for the known absolute speed float) to verify the primary vehicle pointer for your specific Arch Linux environment.

### Telemetry Memory Offsets

The telemetry data is structured at the following offsets relative to the vehicle base pointer:

| Telemetry Data | Data Type | Relative Offset (Hex) | Description / Notes |
| :--- | :--- | :--- | :--- |
| **Position X** | `Float32` | `+0x00` | Lateral world coordinate (**meters**). |
| **Position Y** | `Float32` | `+0x04` | Vertical world coordinate/height (**meters**). |
| **Position Z** | `Float32` | `+0x08` | Longitudinal world coordinate (**meters**). |
| **Velocity X** | `Float32` | `+0x10` | Lateral velocity vector. |
| **Velocity Y** | `Float32` | `+0x14` | Vertical velocity vector (used to detect airborne state). |
| **Velocity Z** | `Float32` | `+0x18` | Longitudinal velocity vector. |
| **Absolute Speed** | `Float32` | `+0x24` | Scalar speed in **m/s**. Multiply by **3.6** for **km/h**. |
| **Rotation Matrix X** | `Float32` | `+0x30` | 3D orientation data (Pitch). |
| **Rotation Matrix Y** | `Float32` | `+0x34` | 3D orientation data (Yaw/Heading). |
| **Rotation Matrix Z** | `Float32` | `+0x38` | 3D orientation data (Roll). |
| **Engine RPM** | `Float32` | `+0x1A4` | Current engine revolutions per minute. |
| **Current Gear** | `Int32` | `+0x1B0` | Integer representing the active transmission gear. |

*Note: The exact hexadecimal offsets provided above are **Estimated** based on standard RenderWare vehicle structs utilized in EA Black Box titles. Minor offset shifting may occur depending on the specific memory alignment of the PS2 ISO.*
