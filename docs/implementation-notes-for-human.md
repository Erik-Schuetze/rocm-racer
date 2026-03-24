# Implementation notes for human steps

These are manual steps that cannot be automated by the training script and must be completed before running `main.py`.

## 1. Assign `rocm-racer.ps2` to memory card slot 1 in PCSX2

The `rocm-racer.ps2` memory card was created in PCSX2 but is not yet assigned to a slot. The current `PCSX2.ini` has `Slot1_Filename = Mcd001.ps2`.

To fix this:
1. Open PCSX2.
2. Go to **Settings → Memory Cards**.
3. Assign `rocm-racer.ps2` to **Slot 1**.

Until this is done the NFS Underground 2 save file will not be visible in-game.

## 2. Verify the highway loop save state

The save state is committed to the repo at:

```
savestates/rocm-racer-nfsu2-highway.p2s
```

`main.py` loads it automatically — no manual action required unless you need to recreate it.

To recreate it:
1. Boot **Need for Speed: Underground 2** in PCSX2 with `rocm-racer.ps2` in slot 1.
2. Load the 100% completed save.
3. Start a free-drive session on the highway loop.
4. Once you are on the track and driving, open the PCSX2 save state menu and save to a slot.
5. Locate the resulting `.p2s` file inside `~/.config/PCSX2/sstates/` and copy it to `savestates/rocm-racer-nfsu2-highway.p2s` in the repo.

If you save to a different path, pass it explicitly:

```bash
python main.py --statefile /path/to/your/save.p2s
```

## 3. Set up `/dev/uinput` permissions (required for virtual gamepad)

The virtual gamepad uses `/dev/uinput` which requires both the kernel module and correct device permissions. Run the one-time setup script:

```bash
sudo bash setup-uinput.sh
```

This does three things:
1. Ensures the `uinput` kernel module loads on every boot (`/etc/modules-load.d/uinput.conf`).
2. Creates a udev rule so `/dev/uinput` is owned by the `input` group with mode `0660` (`/etc/udev/rules.d/99-uinput.rules`).
3. Adds your user to the `input` group.

**You must log out and back in** (or reboot) after running the script for the group membership to take effect.

To verify it worked:

```bash
ls -la /dev/uinput          # should show group 'input', mode crw-rw----
id -nG | grep -w input      # should print 'input'
```

## 4. Controller bindings — automated

`main.py` writes `~/.config/PCSX2/game_controller_db.txt` and creates the uinput virtual gamepad device automatically before launching PCSX2. No manual mapping in the PCSX2 GUI is required.

The `[Pad1]` section of `~/.config/PCSX2/inis/PCSX2.ini` must use SDL3 binding names. This project configures it correctly:

```ini
Cross  = SDL-0/FaceSouth
Circle = SDL-0/FaceEast
Triangle = SDL-0/FaceNorth
Square = SDL-0/FaceWest
L1     = SDL-0/LeftShoulder
R1     = SDL-0/RightShoulder
L2     = SDL-0/+LeftTrigger
R2     = SDL-0/+RightTrigger
LLeft  = SDL-0/-LeftX
LRight = SDL-0/+LeftX
```
