# Fortschritt

## Stand 2026-11-20 – Phase 1 (Python-Referenz) abgeschlossen

### Erledigt
- **Setup in WSL2 geprüft** → `SETUP.md`. Toolchain, `gh`, git-Identität und Go 1.26 sind da. AVX-512 inkl. VNNI/BF16 vorhanden. WSL hat nur 15 GiB RAM (Empfehlung `.wslconfig` in `SETUP.md`). ROCm fehlt, wird erst vor Phase 3.3 gebraucht.
- **Repos:** `littlebit-gguf` (vorerst privat). Forks `ScarionWorkspace/llama.cpp` und `ScarionWorkspace/ollama`, jeweils mit Branch `littlebit`. Der llama.cpp-Branch setzt auf Tag `b12257` auf (= `LLAMA_CPP_VERSION` von Ollama).
  Die Forks sind **verlinkt, keine Submodule** (Begründung im README). `scripts/setup_workspace.sh` klont alles.
- **Korrektheitsmodelle** (`scripts/make_littlebit_init.py`, ruft LittleBits Code auf, kopiert nichts):
  Qwen3-0.6B, LittleBit-2-Init (SVD + Joint-ITQ, Residual, 0 QAT-Schritte) mit 0.55 bpw (Ranks k/v 120, q/o 168, MLP 192) und mit 1.0 bpw (232/320/360). Erneute Erzeugung liefert eine byte-identische Datei.
- **Referenz** `littlebit_ref/` (eigener Code, numpy/torch, kein `transformers` im Forward-Pass):
  Bit-Entpacken, Checkpoint-Loader, Qwen3-Forward mit faktorisierten Schichten, Metriken.
- **Golden Files** (`scripts/make_golden.py`, gerechnet in FP64, gespeichert als FP32, nicht in Git):
  Logits und alle Block-Ausgaben für 4 feste Prompts, Ein-/Ausgaben aller 7 Projektionen in Block 0/Mitte/letzter, dazu Zufallseingaben. Erzeugt für 0.6B-Init-0.55, 0.6B-Init-1.0 und **den trainierten 8B-Checkpoint** (`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`). Größe 46/46/84 MB.
- **Tests:** 19 Unit-Tests (Packing, auch gegen LittleBits `binary_packer`; Faktorisierung = dichte Rekonstruktion; Metriken). Dazu Golden-Tests für alle drei Modelle. Die CI (GitHub Actions) führt die Unit-Tests ohne Modelle aus.

### Vergleichsergebnisse (`results/`)

| Vergleich | rel. max. Abweichung Logits | max. KL | Ergebnis |
|---|---|---|---|
| Unsere Referenz (dichtes Qwen3-0.6B) vs. HF transformers, FP32 | 2,6e-6 | 2,7e-11 | bestanden |
| Unsere vs. LittleBit `load_quantized_model`, 0.6B-Init-0.55, FP32, **BF16-Emulation von `l`** | 4,9e-5 | 2,2e-8 | bestanden |
| dto. FP64 | 1,8e-5 | 2,6e-9 | bestanden |
| dto. **ohne** Emulation (exaktes `l`) | 0,20 | 0,37 | erwartete Abweichung, siehe unten |

**Befund: LittleBit rundet `l` auf BF16.** `load_quantized_model` wandelt nur die Vorzeichen in den gewünschten dtype um. `u1, u2, v1, v2` bleiben BF16, deshalb wird `l = v1·u2` im Forward-Pass noch einmal auf BF16 gerundet (≈0,2 % pro Element). Alle Faktoren und Vorzeichen sind bit-identisch, die Formel ist bestätigt. Als Referenz gilt bei uns das **exakte** Produkt der gespeicherten Werte. Im GGUF wird `l` eine FP16-Blockskala und ist damit genauer als im Original (rel. ≤4,8e-4). Der Rest von ~1e-5 auch in FP64 stammt aus HF selbst: HF rechnet RoPE in FP32, auch das dichte Modell weicht in FP64 um 3e-7 ab. Das untrainierte Init verstärkt diese Abweichung.

**8B-Checkpoint:** Die Referenz läuft (6,9 GB RAM, 35 s für 4 Prompts). „The capital of France is“ → „ Paris“. Ein direkter Vergleich gegen LittleBits eigene Inferenz ist mit 15 GiB WSL-RAM **nicht** möglich, weil deren Loader ±1 in FP32 auspackt (≈14 GB). Die Formel ist über das 0.6B mit identischem Codepfad bestätigt. Der Vergleich ist optional nach einer RAM-Erhöhung.

### Toleranzen

- **Bewertet wird nur über Logits** (max. abs./rel. Abweichung, KL-Divergenz), nie über generierten Text oder Argmax. Untrainierte Inits haben fast gleichauf liegende Top-Tokens: Der kleinste gemessene Abstand zwischen Top-1 und Top-2 betrug 0,03 Logits. Argmax-Übereinstimmung wird nur informativ ausgegeben (`littlebit_ref/metrics.py`).
- **Aktuelle Stufe `TOL_FP_REFERENCE`: rel_max ≤ 1e-4, KL_max ≤ 1e-6.** Sie gilt **nur für Python gegen Python** (gleiche Mathematik, FP32/FP64, verschiedene Implementierungen).
- **Für llama.cpp mit `Q1_0` wird in Phase 3 eine eigene Toleranzstufe aus einer Messung abgeleitet.** `Q1_0` quantisiert die Aktivierungen vor dem Skalarprodukt auf Q8_0 (CPU) bzw. Q8_1 (CUDA/HIP), auch im latenten Raum (Rank 120–824). Das ist eine gewollte Näherung und passt nicht zur FP-Stufe. Vorgehen: Den Q8-Effekt zuerst in Python beziffern (Aktivierungen mit Q8_0-Rundung simulieren, Abweichung pro Schicht sowie Logits/KL messen). Daraus mit Sicherheitsabstand die llama.cpp-Stufe festlegen. Zusätzlich am trainierten 8B messen, weil das untrainierte Init Abweichungen verstärkt.

### Erkenntnisse für Phase 2 (Format)
- Wertebereiche der Skalen (`scripts/scale_stats.py`): `l` ∈ [1,6e-2; 0,12] (8B), in FP16 unproblematisch (rel. Rundung ≤4,8e-4).
  **Im trainierten 8B sind ~0,1 % von `h` und `g` winzig** (bis 2e-11) und werden in FP16 zu 0 oder subnormal. Beim 0.6B-Init kommt das nicht vor. Der absolute Beitrag ist vernachlässigbar, wird aber in Phase 2 gemessen. Bei Bedarf `g` als F32 ablegen. `h` sitzt in der Q1_0-Blockskala und muss deshalb FP16 sein.
- Der WSL-RAM ist für das 8B knapp. Die Referenz kommt mit 6,9 GB aus, die BF16-Konvertierung von Qwen3-8B für die Vergleichsquants braucht mehr.

### Entscheidungen (freigegeben am 2026-11-20, unverändert)
1. Format-Variante A: Faktoren als `GGML_TYPE_Q1_0`, `h` und `l` in den Blockskalen, `g` separat.
2. Modelle: Korrektheit mit dem Qwen3-0.6B-Init, Benchmark mit `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`.
3. GTX 1080 / CUDA entfällt. GPU ist die RX 9070 XT (HIP, Fallback Vulkan).
4. Repo-Name `littlebit-gguf`. Wird nach dem Phase-1-Merge und der Historienprüfung öffentlich (freigegeben am 2026-11-20).

### Offene Punkte
- ROCm für gfx1201 unter WSL (vor Phase 3.3).
- Optional: `.wslconfig` mit mehr RAM (siehe `SETUP.md`).
- Verhalten von `llama-quantize` bei fremden Q1_0-Tensoren (Phase 2/3).

### Nächster Schritt
Phase 2: `convert_littlebit_to_gguf.py` (Q1_0-Faktoren, F32-`g`, Metadaten `littlebit.*`), Roundtrip-Test GGUF → Python-Referenz, effektive bpw der ganzen Datei. Vorher die Python-Simulation der Q8-Aktivierungen als Grundlage für die Toleranz in Phase 3.
