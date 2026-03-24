#!/usr/bin/env bash
# setup-uinput.sh — One-time system setup for /dev/uinput access.
#
# Run this once after a fresh install or OS upgrade:
#   sudo bash setup-uinput.sh
#
# What it does:
#   1. Ensures the uinput kernel module loads on every boot.
#   2. Creates a udev rule so /dev/uinput is owned by the 'input' group
#      with mode 0660 (group read-write).
#   3. Adds the current user (or $SUDO_USER) to the 'input' group.
#   4. Loads the module and triggers the udev rule immediately.
#
# After running, LOG OUT AND BACK IN (or reboot) for the group change
# to take effect.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"

# 1. Persist the uinput module across reboots.
MODULES_CONF="/etc/modules-load.d/uinput.conf"
if [[ ! -f "$MODULES_CONF" ]]; then
    echo "uinput" > "$MODULES_CONF"
    echo "[✓] Created $MODULES_CONF"
else
    echo "[·] $MODULES_CONF already exists"
fi

# 2. udev rule: set /dev/uinput group to 'input', mode 0660.
UDEV_RULE="/etc/udev/rules.d/99-uinput.rules"
RULE_CONTENT='KERNEL=="uinput", SUBSYSTEM=="misc", GROUP="input", MODE="0660"'
if [[ ! -f "$UDEV_RULE" ]] || ! grep -qF "$RULE_CONTENT" "$UDEV_RULE"; then
    echo "$RULE_CONTENT" > "$UDEV_RULE"
    echo "[✓] Created $UDEV_RULE"
else
    echo "[·] $UDEV_RULE already exists"
fi

# 3. Add user to 'input' group (idempotent).
if id -nG "$TARGET_USER" | grep -qw input; then
    echo "[·] User '$TARGET_USER' is already in the 'input' group"
else
    usermod -aG input "$TARGET_USER"
    echo "[✓] Added '$TARGET_USER' to the 'input' group"
fi

# 4. Try to load the module and apply the udev rule now.
if modprobe uinput 2>/dev/null; then
    echo "[✓] uinput module loaded"
else
    if [[ -e /dev/uinput ]]; then
        echo "[·] modprobe failed (kernel module dir may be out of sync after"
        echo "    a pacman upgrade — this is normal on Arch). /dev/uinput already"
        echo "    exists so uinput is available. A reboot will fix modprobe."
    else
        echo "[!] WARNING: modprobe uinput failed and /dev/uinput does not exist."
        echo "    Reboot to load the updated kernel, then re-run this script."
    fi
fi

# Apply udev rule to the existing device node (if present).
udevadm control --reload-rules
if [[ -e /dev/uinput ]]; then
    udevadm trigger /dev/uinput
    # Give udev a moment to apply the rule.
    sleep 0.5
    echo ""
    echo "Done. /dev/uinput status:"
    ls -la /dev/uinput
else
    echo ""
    echo "/dev/uinput does not exist yet. Reboot to load the uinput module."
fi

echo ""
echo "IMPORTANT: Log out and back in (or reboot) for the 'input' group"
echo "membership to take effect, then run:  python main.py"
