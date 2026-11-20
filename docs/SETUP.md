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

### ROCm (vor Phase 3.3)

Noch offen. Wird vor dem GPU-Teil geprüft: passende ROCm-Version für gfx1201 unter WSL samt
`librocdxg`. Fallback ist das Vulkan-Backend.

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
