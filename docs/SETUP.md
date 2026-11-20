# Setup

## Ist-Zustand WSL2 (geprüft 2026-11-20, in WSL)

| Punkt | Ergebnis |
|---|---|
| OS | WSL2, Kernel 6.18.58.1-microsoft-standard-WSL2, Ubuntu 24.04 |
| gh | angemeldet als `ScarionWorkspace` (HTTPS, Scopes `repo`, `workflow`, `gist`, `read:org`) |
| git-Identität | gesetzt, E-Mail ist die GitHub-No-Reply-Adresse |
| gcc / cmake / ninja | 13.3.0 / 3.28.3 / 1.11.1 |
| Python | 3.12.3, venv unter `~/littlebit-work/.venv` |
| Go | 1.26.10 |
| CPU | Ryzen 7 7800X3D, 16 Threads. Flags: `avx2 avx512f avx512bw avx512dq avx512vl avx512cd avx512_vnni avx512_bf16 avx512_vbmi avx512_bitalg avx512_vpopcntdq avx512ifma` |
| RAM in WSL | ursprünglich 15 GiB (+4 GiB Swap, Standard = halber Windows-RAM). **Seit 2026-11-20: 24 GB + 16 GB Swap** über `.wslconfig` (geprüft mit `free -h`: 23 GiB / 16 GiB) |
| Platte | 954 GB frei |
| ROCm / HIP | **nicht installiert** (`/opt/rocm` fehlt). In `/usr/lib/wsl/lib` nur D3D12/DXCore, kein `librocdxg` |
| NVIDIA | entfällt (GTX 1080 steckt in einem anderen PC) |

### Python-Umgebung

```bash
python3 -m venv ~/littlebit-work/.venv
~/littlebit-work/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/littlebit-work/.venv/bin/pip install -r littlebit-gguf/requirements.txt
```

Installiert: torch 2.15.0+cpu, transformers 4.57.6, numpy 2.5.5.
**transformers bleibt auf 4.x**: LittleBit ist für 4.51 geschrieben, mit 5.x bricht schon das Laden
(`device_map` braucht dort `accelerate`, `torch_dtype` ist veraltet).

### Mehr RAM für WSL (umgesetzt 2026-11-20)

Für Qwen3-8B-Vergleichsquants (`convert_hf_to_gguf.py` auf 16 GB BF16) und den 8B-Vergleich
gegen LittleBits eigene Inferenz reichen 15 GiB knapp oder gar nicht. 24 GB lassen Windows (31 GB) genug Luft. In Windows
`%UserProfile%\.wslconfig` anlegen bzw. ergänzen, danach in PowerShell `wsl --shutdown`:

```ini
[wsl2]
memory=24GB
swap=16GB
```

### ROCm für die RX 9070 XT unter WSL2 (Recherche 2026-11-20, noch nicht installiert)

Nichts davon ist ausgeführt. Die Installation braucht `sudo` und ggf. ein Windows-Treiberupdate, deshalb erst nach Rückfrage.

Stand der Quellen:
- **Offizieller Weg:** Adrenalin-Treiber für WSL2 (AMD nennt 26.2.2), dazu ROCm **7.2.x** und `librocdxg` 1.2.0. Die Kompatibilitätsmatrix von librocdxg führt RX 9070 / 9070 XT / 9070 GRE bei librocdxg 1.2.0 + ROCm 7.2.x. Ubuntu 24.04 wird unterstützt. Für ROCm vor 7.13 muss `HSA_ENABLE_DXG_DETECTION=1` gesetzt sein. Das Repo `ROCm/librocdxg` ist inzwischen veraltet, der Code liegt in `ROCm/rocm-systems`. Die Releases (`.deb`) gibt es aber weiter dort. librocdxg braucht nach eigener Aussage keine „Radeon Software for Linux“-Pakete.
- **Bekannte Probleme:** Die ROCm-10.1-Wheels hängen auf gfx1201 unter WSL2 beim ersten Kernel (Windows-TDR). ROCm 7.2 läuft auf derselben Maschine (Adrenalin 26.8.1, ROCm 7.2.3, librocdxg 1.2.0). Der Fix ist in einem Nightly (10.3.0a20261105), ein stabiles Release ist nicht genannt.
- **Wichtig für uns:** Der Hänger tritt auf, wenn die **iGPU des 7800X3D** („AMD Radeon Graphics“) neben der 9070 XT sichtbar ist. Workaround laut Issue: iGPU im Windows-Geräte-Manager deaktivieren, dann `wsl --shutdown`. Alternativ `ROCR_VISIBLE_DEVICES`.
- **Konkreter Weg (2026-11-20 festgelegt):** ROCm **7.2.4** (aktuelle 7.2.x) und librocdxg **1.2.0** (die Matrixzeile mit RX 9070 XT). Aus der nativen Linux-Anleitung werden bewusst **nicht** übernommen: `amdgpu-dkms` und `linux-headers`. WSL läuft mit Microsofts eigenem Kernel (6.18-microsoft), die GPU bedient der Windows-Treiber über `/dev/dxg` (vorhanden, geprüft). Neuere librocdxg-Versionen (1.2.1/1.2.2) sind laut Matrix auf ROCm 7.13/7.14 ausgerichtet. Download-URLs geprüft (HTTP 200).
- **Skript:** `scripts/setup/install_rocm_wsl.sh` (braucht `sudo`, vom Menschen auszuführen: `! bash scripts/setup/install_rocm_wsl.sh`). Danach neue Shell, dann `rocminfo | grep -E 'Marketing Name|gfx'` und `hipcc --version`.

Danach:
1. llama.cpp mit `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201` in `build-hip` bauen, `test-backend-ops` für `Q1_0`.
2. Fallback, wenn ROCm unter WSL nicht läuft: Vulkan-Backend (Mesa `dzn`/D3D12 unter WSL prüfen) oder natives Ubuntu.

Quellen:
[AMD: ROCm on Radeon, WSL How-to](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html) ·
[ROCm/librocdxg (README, Kompatibilitätsmatrix)](https://github.com/ROCm/librocdxg) ·
[librocdxg #74: RX 9070 XT läuft unter WSL2](https://github.com/ROCm/librocdxg/issues/74) ·
[rocm-systems #13291: Hänger auf gfx1201 mit ROCm 10.1, iGPU-Workaround](https://github.com/ROCm/rocm-systems/issues/13291) ·
[ROCm CLI unter WSL2](https://rocm.docs.amd.com/projects/rocm-cli/en/latest/install/wsl.html)

---

## Historie: Ist-Zustand von Windows aus (Phase 0)

| Punkt | Ergebnis |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| **WSL2** | **nicht installiert** („Der Windows-Subsystem für Linux ist nicht installiert“) |
| Virtualisierung im BIOS | aktiv (`VirtualizationFirmwareEnabled: True`) |
| CPU | AMD Ryzen 7 7800X3D, 8 Kerne (AVX2/AVX-512-Flags erst unter Linux mit `lscpu` prüfbar) |
| GPU | AMD Radeon RX 9070 XT (+ iGPU „AMD Radeon Graphics“) |
| **GTX 1080** | **nicht in diesem Rechner gefunden** (`Win32_VideoController`). Wo steckt sie? |
| RAM | 31,1 GB |
| Freier Platz C: | nicht ermittelt (Abfrage lieferte nichts), unter WSL mit `df -h ~` prüfen |
| git (Windows) | 2.55.0, **`user.name`/`user.email` global nicht gesetzt** |
| gh | **nicht installiert** |
| Python (Windows) | 3.12 |
| gcc / cmake / go / nvcc / hipcc | **nicht vorhanden** (unter Windows; ohnehin für WSL vorgesehen) |
| Ollama (Windows) | installiert (`%LOCALAPPDATA%\Programs\Ollama`), für Vergleiche nutzbar |

Phase 0 wurde trotzdem durchgeführt. Sie braucht nur Lesezugriff (shallow clones unter `%USERPROFILE%\littlebit-work\ref\`). Ab Phase 1 brauchen wir die Linux-Umgebung.

## Was du tun musst (Admin-Rechte, ich führe das nicht selbst aus)

In einer **Administrator-PowerShell**:

```powershell
wsl --install -d Ubuntu-24.04
# danach Neustart; beim ersten Start Linux-Benutzer + Passwort anlegen
```

Danach in Ubuntu (die Schritte 2–7 aus `CLAUDE.md`):

```bash
sudo apt update && sudo apt install -y git gh build-essential cmake python3-venv ninja-build
gh auth login            # GitHub.com → HTTPS → Login with a web browser
gh auth setup-git
git config --global user.name "<Name>"
git config --global user.email "<ID+user>@users.noreply.github.com"
mkdir -p ~/littlebit-work && cd ~/littlebit-work
cp -r /mnt/c/Users/<Windows-Benutzer>/littlebit-work/docs .   # bisherige Docs übernehmen
# Projektbeschreibung hier als CLAUDE.md speichern (Schritt 7)
```

Go (für Ollama, die Version aus `ollama/go.mod` nehmen; ich prüfe sie dann):

```bash
# Beispiel – Version anpassen:
curl -LO https://go.dev/dl/go1.XX.Y.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.XX.Y.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
```

Dann Claude Code in WSL installieren, anmelden und in `~/littlebit-work` starten. Ich prüfe dort Folgendes und trage die Ergebnisse hier nach: `gh auth status`, Compiler-Versionen, `lscpu` (AVX-512), `df -h ~`, `free -h`.

## GPU (erst vor Phase 3 nötig)

- RX 9070 XT (gfx1201) unter WSL2: ROCm für WSL plus `librocdxg`. Ich prüfe dann die passende ROCm-Version für gfx1201 und gebe die genauen Befehle. Fallback: Vulkan-Backend (llama.cpp hat Q1_0-Shader).
- GTX 1080: steckt in einem anderen PC, **außerhalb des Projektumfangs** (Entscheidung 2026-11-20).
