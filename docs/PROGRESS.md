# Fortschritt

## Stand 2026-11-20 – Phase 0 (Analyse) abgeschlossen und freigegeben

Erledigt:
- Umgebung von Windows aus geprüft → `docs/SETUP.md`. **WSL2 fehlt**, ebenso git-Identität, `gh` und die Toolchain.
- LittleBit, littlebit-bench, llama.cpp und Ollama gelesen → `docs/ANALYSIS.md`.

Entscheidungen (freigegeben am 2026-11-20):
1. **Format: Variante A.** Faktoren als bestehender `GGML_TYPE_Q1_0`. `h` steckt in den Blockskalen von U, `l` in denen von V, `g` liegt als eigener Tensor vor, der Rank von U ist auf ein Vielfaches von 128 aufgefüllt (Skala 0). Ein eigener Typ oder fusionierter Kernel kommt erst nach Messungen.
2. **Modelle:** Korrektheit mit Qwen3-0.6B (LittleBit-2-Init ohne QAT, selbst erzeugt). Benchmarks mit `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`, Vergleich gegen Q2_K/IQ2_XXS/IQ1_S/TQ1_0 von Qwen3-8B.
3. **GTX 1080 / CUDA entfällt.** Die Karte steckt in einem anderen PC. GPU-Ziel ist nur noch die RX 9070 XT (HIP, Fallback Vulkan).
4. **Repo-Name:** `littlebit-gguf`.

Offene Punkte (Setup, blockiert Phase 1):
- WSL2/Ubuntu 24.04 installieren (Admin, Neustart), dann Toolchain, `gh`, git-Identität, Go. Siehe `SETUP.md`.
- ROCm für gfx1201 unter WSL (erst vor Phase 3).

Nächster Schritt: In WSL das Setup prüfen, Repo `littlebit-gguf` und Forks anlegen, dann Phase 1 starten, also Python-Referenz und Golden Files.
