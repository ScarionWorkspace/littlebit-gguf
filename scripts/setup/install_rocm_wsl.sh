#!/usr/bin/env bash
# ROCm 7.2.4 user space + librocdxg 1.2.0 for WSL2 / Ubuntu 24.04 / Radeon RX 9070 XT (gfx1201).
#
# Sources (checked 2026-11-20):
#  - ROCm 7.2.4 quick start (Ubuntu 24.04 commands):
#    https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/quick-start.html
#  - librocdxg README, WSL compatibility matrix: RX 9070 XT listed for librocdxg 1.2.0 + ROCm 7.2.x;
#    HSA_ENABLE_DXG_DETECTION=1 required for ROCm < 7.13: https://github.com/ROCm/librocdxg
#  - Known issue (gfx1201 hang with ROCm 10.1, iGPU next to dGPU):
#    https://github.com/ROCm/rocm-systems/issues/13291
#
# Deliberate deviation from the native-Linux quick start: NO amdgpu-dkms / linux-headers.
# WSL runs Microsoft's kernel; the GPU is driven by the Windows Adrenalin driver through /dev/dxg,
# and librocdxg connects the ROCm runtime to it (README: "does not require installation of
# Radeon Software for Linux (RSL) packages").
#
# Prerequisites on Windows: recent AMD Adrenalin driver (AMD names 26.2.2 for WSL2 + ROCm 7.2.1).
# If the first GPU kernel hangs: disable the CPU's iGPU ("AMD Radeon(TM) Graphics") in Device
# Manager, then `wsl --shutdown` in PowerShell.
#
# Run as your normal user (it calls sudo itself):  bash scripts/setup/install_rocm_wsl.sh
set -euo pipefail

ROCM_DEB=amdgpu-install_7.2.4.70204-1_all.deb
ROCM_URL=https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/$ROCM_DEB
DXG_DEB=rocdxg-roct_1.2.0_amd64.deb
DXG_URL=https://github.com/ROCm/librocdxg/releases/download/v1.2.0/$DXG_DEB

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

[ -e /dev/dxg ] || { echo "/dev/dxg missing: not WSL2 or no GPU passthrough"; exit 1; }

echo "== 1/4 ROCm 7.2.4 repository (amdgpu-install package only sets up apt sources)"
wget -q "$ROCM_URL"
sudo apt install -y "./$ROCM_DEB"
sudo apt update

echo "== 2/4 ROCm user space (no kernel driver under WSL)"
sudo apt install -y python3-setuptools python3-wheel
sudo apt install -y rocm
sudo usermod -a -G render,video "$LOGNAME"

echo "== 3/4 librocdxg 1.2.0 (WSL bridge)"
wget -q "$DXG_URL"
sudo dpkg -i "./$DXG_DEB"

echo "== 4/4 environment"
if ! grep -q HSA_ENABLE_DXG_DETECTION ~/.bashrc; then
  echo 'export HSA_ENABLE_DXG_DETECTION=1   # librocdxg, required for ROCm < 7.13' >> ~/.bashrc
fi
echo 'export PATH=$PATH:/opt/rocm/bin' | grep -qxFf - ~/.bashrc || echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc

echo
echo "Done. Open a new shell (or: source ~/.bashrc), then check:"
echo "  rocminfo | grep -E 'Marketing Name|gfx'"
echo "  hipcc --version"
