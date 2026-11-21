# Fortschritt

## Stand 2026-11-20 – Phase 3 (llama.cpp): CPU und GPU (HIP) korrekt, PPL bestätigt

### GPU (RX 9070 XT, ROCm 7.2.4 unter WSL2)
- ROCm installiert und geprüft (`SETUP.md`). llama.cpp-Build `build-hip` (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201`).
- `test-backend-ops -o MUL_MAT -p type_a=q1_0`: **85/85 OK** auf ROCm0.
- Block 0 schrittweise (`results/llamacpp/steps_block0_8b_hip.txt`): Der faktorisierte `v_proj` (gleicher Eingang) stimmt auf der GPU bis auf **5,9e-8** mit der Simulation überein, auf der CPU bis auf 6,4e-8. Der LittleBit-Pfad im HIP-Backend ist damit korrekt. Ab der Attention weicht die GPU ab, weil K·Q dort in F16 läuft. Das ist kein LittleBit-Thema.
- Ganzes Modell, 2108 Positionen (`results/llamacpp/qwen3-8b-lnone-f16_hip.json`): **Stufe 1 und 2 bestanden.** vs. Simulation KL mittel 2,40e-4 (Rauschboden 2,38e-4), vs. exakt KL mittel 8,81e-4, ΔPPL −0,14 %.

### Perplexity WikiText-2 (LittleBit-Methode: seqlen 2048, alle Positionen, 146 Fenster, 299 078 Tokens, GPU)

| Variante | PPL |
|---|---|
| exaktes `l` (`--l-round none`) | **24,4932** |
| BF16-gerundetes `l` (`--l-round bf16`, wie LittleBits Loader) | **24,4896** |
| Modellkarte (LittleBit `eval.py`, BF16 auf A40) | 24,4754 |

- Die BF16-Variante ist der vergleichbare Wert. Sie liegt **0,06 %** über dem veröffentlichten Wert, das passt zu den Q8-Aktivierungen (simuliert −0,1 % bis +0,1 %) und liegt deutlich innerhalb von Stufe 2 (0,5 %).
- Die BF16-Rundung von `l` kostet praktisch nichts (0,015 %).
- Gegencheck CPU vs. GPU auf den ersten 26 Fenstern: 23,8961 vs. 23,8953 (0,003 %). Der CPU-Teillauf ist in `results/llamacpp/ppl_wikitext2_s2048_qwen3-8b-lnone-f16_cpu.json` gesichert.
- Ergebnisse: `results/llamacpp/ppl_wikitext2_s2048_qwen3-8b-l{none,bf16}-f16_hip.json`.

### Geschwindigkeit RX 9070 XT (llama-bench, ngl 99)

| Datei | Größe | pp512 | tg128 |
|---|---|---|---|
| 8B LittleBit, Embeddings Q4_K, lm_head Q6_K | 1,33 GiB | 3093 ± 170 t/s | 70,95 t/s |
| 8B LittleBit, Embeddings/lm_head F16 | 2,84 GiB | 1281 ± 38 t/s | 61,75 t/s |

Vermutung, noch ungeprüft: Bei 1,33 GiB sollte die Generierung deutlich schneller sein können. Die vielen kleinen Kernels pro Schicht (7 Ops × 7 Projektionen × 36 Blöcke) dürften die Grenze sein. Vor jeder Optimierung profilieren und gegen dichtes Qwen3-8B auf derselben Karte messen.

### Entscheidungen 2026-11-21
- **Richtung nach dem Baseline-Ergebnis** (LittleBit 8B verliert bei gleicher Größe gegen Qwen3-1.7B Q4_K_M): **A, danach C.** A = Infrastruktur fertigstellen (Profiling → Optionen Geschwindigkeit → Ollama → ehrliches README, Kontakt zu Parallelprojekten). C = danach große Modelle (27B/32B), sobald ein trainierter Checkpoint verfügbar ist. Kein Cloud-Training für das 8B (laut Paper-Verhältnis nicht konkurrenzfähig).
- **Parallelprojekte** (`docs/RELATED.md`): Kontakt wegen eines gemeinsamen GGUF-Formats erst **nach** den Profiling-Ergebnissen (Option b). Text vorher dem Nutzer zur Freigabe vorlegen.
- **Geschwindigkeit:** Nach dem Profiling Optionen (fusionierte Op / eigener Typ ohne Blockskalen / beides) mit Messwerten und Empfehlung vorlegen. Das hat Vorrang vor Phase 4 (Ollama).
- **Eigenes QAT:** vorerst **kein Geld ausgeben**. Reihenfolge: (1) nach Profiling und Vergleich auf der RX 9070 XT testen, ob Training mit PyTorch/ROCm unter WSL stabil läuft (Rezept `mgiachetti/little-bit` für Qwen3-0.6B, laut README 8 GB). Wenn ja, das 0.6B lokal trainieren (geschätzt 4–8 Tage). (2) Für das 8B im Kontakt mit den Parallelprojekten bzw. dem Autor von `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` nach Checkpoints oder Weitertraining fragen. (3) Erst danach Credits (TPU Research Cloud, AWS/Azure Research) oder Cloud-Miete (8B, 1 Epoche, grob 200–500 USD, vorher 1 h Testlauf zur Durchsatzmessung).

### Nächste Schritte
1. PR `phase3-llamacpp` → `main` (erledigt, PR #4).
2. Vergleichsquants aus Qwen3-8B (Download lief bei Stand noch: `../models/Qwen3-8B`, bei Abbruch `hf download Qwen/Qwen3-8B --local-dir ../models/Qwen3-8B` erneut): Q4_K_M, Q2_K, IQ2_XXS, IQ1_S, TQ1_0. Größe, PPL (dieselbe Methode) und Geschwindigkeit auf der GPU messen, Vergleich bei gleicher Dateigröße.
3. Profiling der Generierung auf der GPU, danach Optionen (Fusion der Kette / eigener Typ ohne Blockskalen) mit Empfehlung zur Entscheidung vorlegen.
4. Phase 4: Ollama gegen den Fork bauen.

## Stand 2026-11-20 – Phase 3.1 (llama.cpp, CPU)

### Erledigt
- **llama.cpp-Patch** (Fork `ScarionWorkspace/llama.cpp`, Branch `littlebit`, Commit `3735b3a7f` auf Basis `b12257`), 6 Dateien, +118 Zeilen, **kein neuer ggml-Typ und keine neue Op**:
  - `llama_model_base::create_tensor_lb()`: lädt statt eines dichten Gewichts die Faktoren `lb_u/lb_v/lb_g[_r]`, sofern vorhanden. Der Rank kommt aus der Datei. Die Faktoren werden in `model.lb` unter einem Stellvertreter-Tensor (`lb_u`) registriert. Verwendet in `create_tensor_qkv` (generisch für alle Architekturen mit separaten Q/K/V) und in `qwen3.cpp` für `wo` und die FFN.
  - `build_lora_mm()`: Für registrierte Gewichte entsteht die Kette `mul(g) → mul_mat(v) → pad → mul_mat(u)` (+ Residualpfad, `add`). Nach dem Muster der vorhandenen Hadamard-Faltung.
  - Loader: `lb_g` wird für `GGML_OP_MUL` platziert (wichtig für die GPU-Zuordnung).
- **Erster Lauf (CPU):** „The capital of France is“ → „ Paris, and the city is the capital of the French Republic.“
- **Werkzeuge:** `tools/lb_dump_logits.cpp` (Logits, Zwischentensoren über den eval-Callback, NLL-Modus), `scripts/check_llamacpp.py` (Toleranzstufen 1/2 inkl. Rauschboden), `scripts/check_llamacpp_steps.py` (Block 0 Schritt für Schritt), `scripts/ppl_littlebit_method.py` (PPL exakt wie LittleBits `eval.py`).

### Korrektheit (8B, CPU, 4 Prompts + 4×512 WikiText-2, 2108 Positionen, `results/llamacpp/`)

| | KL mittel | KL max | PPL |
|---|---|---|---|
| llama.cpp vs. Simulation (Flash-Attention auto = an) | 2,38e-4 | 5,2e-3 | |
| llama.cpp vs. Simulation (Flash-Attention aus) | 2,36e-4 | 3,3e-3 | |
| Rauschboden: Simulation FP32 vs. FP64 | 2,38e-4 | 5,6e-3 | |
| llama.cpp vs. exakt (FA an) | 8,78e-4 | 5,3e-2 | −0,18 % |
| Simulation vs. exakt | 8,89e-4 | 5,8e-2 | −0,10 % |

**Stufe 1 und Stufe 2 bestanden** (mit FA an und aus).

**Schrittweise, Block 0** (`results/llamacpp/steps_block0_cpu.txt`): Mit Flash-Attention aus stimmt llama.cpp an jedem Zwischenschritt bis auf ~1e-7 mit der Simulation überein. Beim 0.6B gilt das bis zur Blockausgabe, beim 8B bis zur FFN-Norm. Danach weicht das 8B um 1e-3 ab, weil der CPU-„repack“-Pfad für Matrizen mit mehreren Zeilen einen x86-Q8-Quantisierer nutzt (`127/max`, Rundung half-to-even) und nicht die Referenz. Welcher Pfad greift, hängt von der Shape ab. Die Implementierung der LittleBit-Kette ist damit bestätigt. Mit Flash-Attention an weicht die Attention-Ausgabe um 2e-4 ab (FA rechnet teils in F16). Das kippt einzelne Q8-Rundungen, ist also eine Eigenschaft des FA-Pfads und kein LittleBit-Fehler.

**Korrektur der Toleranzstufe 1:** Die ursprüngliche Ableitung („nur die Summationsreihenfolge weicht ab → 10 % des Q8-Effekts reichen“) war falsch. Die Q8-Rundung ist unstetig: Jeder FP32-Unterschied kippt einzelne Quantisierungsstufen, und das pflanzt sich fort. Der gemessene Rauschboden (dieselbe Simulation in FP32 vs. FP64) liegt bei KL mittel 2,38e-4, also bei ~27 % des Q8-Effekts. llama.cpp liegt genau darauf. **Neue Stufe 1 = 2× Rauschboden: KL mittel ≤ 5e-4, KL max ≤ 2e-2, |ΔPPL| ≤ 0,25 %.** Feine Fehler findet die schrittweise Prüfung schärfer. Stufe 2 bleibt unverändert.

### Geschwindigkeit (CPU, 8 Threads, erste Messung, unoptimiert)

| Datei | Größe | pp512 | tg128 |
|---|---|---|---|
| 8B LittleBit, Embeddings Q4_K, lm_head Q6_K | 1,33 GiB | 53 ± 8 t/s | 27,4 t/s |
| 8B LittleBit, Embeddings/lm_head F16 | 2,84 GiB | 53 ± 9 t/s | 18,0 t/s |

Die Generierung ist schnell, die Prompt-Verarbeitung langsam. Mögliche Ursachen: viele kleine Ops (7 statt 1 pro Schicht) und `Q1_0` ohne AVX-512-Pfad. Vergleich mit dichtem Qwen3-8B folgt (Download läuft).

### Läuft gerade (Hintergrund, bei Pause gestartet)
- WikiText-2-PPL nach LittleBit-Methode (seqlen 2048, ~146 Fenster, ~46 s/Fenster auf der CPU) für exaktes `l` und für BF16-`l`. Referenzwert der Modellkarte: 24,48. Der wurde in BF16 auf einer A40 mit LittleBits `eval.py` gemessen, also mit BF16-gerundetem `l`. Zwischenstand nach 2 von 146 Fenstern: 22,4. Das ist noch nicht aussagekräftig.
- Download Qwen3-8B für die Vergleichsquants.

### Offen in Phase 3
- PPL-Ergebnisse eintragen, Vergleichstabelle bei gleicher Dateigröße.
- Prompt-Verarbeitung profilieren (welche Ops kosten Zeit?), AVX-512-Pfad für `Q1_0` prüfen.
- GPU: ROCm für gfx1201 unter WSL einrichten, HIP-Build, `test-backend-ops` für `Q1_0` mit unseren Shapes, dieselben Prüfungen auf der GPU.
- `llama-perplexity` (Standardmethode) zusätzlich, für den Vergleich mit anderen llama.cpp-Quants.

## Stand 2026-11-20 – Phase 2 (GGUF-Format) abgeschlossen

### Erledigt
- **Simulation der llama.cpp-Numerik** (`littlebit_ref/numerics.py`), im Quellcode von ggml `b12257` nachgelesen: `Q1_0`-Matmul quantisiert die Aktivierung auf `Q8_0` (Blöcke à 32, `d = amax/127` als FP16, `roundf`). Die Simulation ist **bit-exakt** zu gguf-py. `h` und `l` werden FP16-Blockskalen, der KV-Cache ist standardmäßig F16, F16-Gewichte lassen die Aktivierungen in F32.
- **Messung des Effekts** (`scripts/measure_numerics.py`, `results/numerics/`), trainiertes 8B, 4 Prompts + 4×512 WikiText-2-Tokens:

  | Variante (FP32) vs. exakt FP64 | KL max | KL mittel | PPL |
  |---|---|---|---|
  | FP32 exakt | 2,5e-11 | 1,4e-12 | ±0,000 % |
  | + FP16-Skalen `h`, `l` | 3,3e-6 | 1,2e-7 | +0,001 % |
  | + F16-KV-Cache | 5,6e-6 | 6,9e-8 | −0,001 % |
  | + Q8_0-Aktivierungen | 5,0e-2 | 8,9e-4 | −0,096 % |
  | alles (= llama.cpp) | 5,8e-2 | 8,9e-4 | −0,099 % |

  Die Q8-Aktivierungen machen praktisch den ganzen Effekt aus und sind für die Qualität unkritisch. Pro einzelner Schicht beträgt der Q8-Fehler 0,4–1,0 % (rel. L2), beim 0.6B-Init und beim 8B gleich. Beim 0.6B-Init explodiert er im ganzen Modell (KL mittel 0,28), weil das untrainierte Modell chaotisch ist. Ganzmodell-Vergleiche mit Toleranz deshalb **nur am 8B**, das 0.6B nur schichtweise.
- **Konverter** `convert_littlebit_to_gguf.py` → Format in `docs/FORMAT.md`. Nutzt den Qwen3-Konverter von llama.cpp für Metadaten, Tokenizer und dichte Tensoren. Die LittleBit-Tensoren werden selbst erzeugt, weil der normale Pfad alles nach FP32 wandelt und damit die int32-Bitwörter zerstören würde. Optionen `--outtype` (Embeddings/lm_head) und `--l-round none|bf16`. 8B in 38 s, 12 GB RAM.
  Abweichung zu `ANALYSIS.md`: `V` wird **nicht** aufgefüllt (behält Rank `r`), nur `U` auf Vielfache von 128. Im Graph kommt dafür ein `ggml_pad` hinzu. Spart beim 0.6B 11 % der Faktorgröße.
- **Roundtrip-Test** `tests/test_gguf_roundtrip.py`: GGUF → Python. Vorzeichen bit-exakt, `g` exakt, `h`/`l` exakt FP16-gerundet, Füllzeilen 0, Logits gleich der Referenz mit FP16-Skalen (rel. ≤ 1e-6). Grün für 0.6B-Init und 8B (`l_round` none und bf16).
- **`llama-quantize`** (gebaut aus dem Branch `littlebit`, CPU) lässt die `lb_*`-Tensoren unverändert und quantisiert nur Embeddings und lm_head. Geprüft mit Q4_K_M und Q2_K.
- **Effektive bpw** (`scripts/gguf_size_report.py`, `results/size_*.json`), Bezug ist immer die Parameterzahl des dichten Modells:

  | Modell | Datei | bpw ganze Datei | bpw Linear |
  |---|---|---|---|
  | 0.6B-Init, Embeddings F16 | 340 MiB | 4,78 | 0,70 (Embeddings = 87 % der Datei) |
  | 8B, Embeddings/lm_head F16 | 2917 MiB | 2,99 | 0,648 |
  | 8B, `llama-quantize Q4_K_M` | 1364 MiB | 1,40 | 0,648 |
  | 8B, `llama-quantize Q2_K` | 1225 MiB | 1,25 | 0,648 |

  Linear 0,648 statt 0,548 bpw: Die `Q1_0`-Blockskalen kosten 12,5 %, das Auffüllen von `U` (Ranks 544/208/824 → 640/256/896) den Rest. Zum Vergleich: das originale Qwen3-8B in Q2_K hat etwa 3,3 GB (in Phase 3 selbst nachmessen).
- llama.cpp-Branch `littlebit` (= `b12257`, unverändert) baut auf der CPU (`build-cpu`, 3 min).

### Toleranzen für Phase 3 (aus der Messung abgeleitet, `littlebit_ref/metrics.py`)

| Stufe | Vergleich | Kriterium |
|---|---|---|
| 0 `TOL_FP_REFERENCE` | Python gegen Python | rel_max ≤ 1e-4, KL_max ≤ 1e-6 |
| 1 `TOL_LLAMACPP_VS_SIM` | llama.cpp gegen Simulation (`Numerics.llamacpp_q1_0`), 8B | KL mittel ≤ 1e-4, KL max ≤ 1e-2, \|ΔPPL\| ≤ 0,05 % |
| 2 `TOL_LLAMACPP_VS_EXACT` | llama.cpp gegen exakte Referenz, 8B | KL mittel ≤ 2e-3, KL max ≤ 0,1, \|ΔPPL\| ≤ 0,5 % |
| Schicht | einzelne Schicht mit Golden-Eingaben, gegen exakt | rel. L2 ≤ 2 % |

Begründung: Stufe 2 ist etwa das Doppelte des gemessenen Q8-Effekts. Stufe 1 erlaubt 10 % davon, weil dieselbe Quantisierung simuliert wird und nur Summationsreihenfolge und Implementierungsdetails abweichen dürfen. Eine knapp gekippte Q8-Rundung kann allerdings eine Stufe springen, deshalb nicht enger. Wenn llama.cpp Stufe 1 knapp verfehlt, Stufe 2 aber klar einhält, zuerst die Ursache suchen (z. B. Flash-Attention oder andere RoPE-Variante), bevor die Grenze gelockert wird.

### Nächster Schritt
Phase 3.1: In llama.cpp (Branch `littlebit`) die Tensoren `lb_u/lb_v/lb_g(_r)` für Qwen3 als optionale Tensoren laden und `build_lora_mm` für Projektionen mit LittleBit-Faktoren auf die Kette `mul → mul_mat → pad → mul_mat` (+ Residual) umleiten. Dann Logits auf der CPU gegen die Golden Files (8B, Stufen 1 und 2) und schichtweise gegen das 0.6B-Init prüfen. Danach `test-backend-ops` für `Q1_0` mit unseren Shapes, PPL (exaktes und BF16-`l`) und `llama-bench`.

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
  **Im trainierten 8B sind ~0,1 % von `h` und `g` winzig** (bis 2e-11) und werden in FP16 zu 0 oder subnormal. Beim 0.6B-Init kommt das nicht vor. **Entscheidung (2026-11-20): `g` wird im GGUF als F32 gespeichert**, damit ist `g` verlustfrei (BF16 → F32 exakt). Die FP16-Unterlaufprüfung betrifft damit nur `h` und `l` (beide Q1_0-Blockskala). Bei `l` gibt es keinen Unterlauf (min. 1,6e-2). Bei `h` den Effekt der ~0,1 % winzigen Werte in Phase 2 messen.
- WSL-RAM auf 24 GB + 16 GB Swap erhöht (2026-11-20, aktiv). Damit sind die BF16-Konvertierung von Qwen3-8B für die Vergleichsquants und der direkte 8B-Vergleich gegen LittleBit machbar.

### Entscheidungen (freigegeben am 2026-11-20, unverändert)
1. Format-Variante A: Faktoren als `GGML_TYPE_Q1_0`, `h` und `l` in den Blockskalen, `g` separat **als F32** (festgelegt 2026-11-20).
2. Modelle: Korrektheit mit dem Qwen3-0.6B-Init, Benchmark mit `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`.
3. GTX 1080 / CUDA entfällt. GPU ist die RX 9070 XT (HIP, Fallback Vulkan).
4. Repo-Name `littlebit-gguf`. Wird nach dem Phase-1-Merge und der Historienprüfung öffentlich (freigegeben am 2026-11-20).

### Vorgaben für Phase 3 (Qualität)
- **PPL des 8B zweimal messen und beide dokumentieren:** mit exaktem `l = v1·u2` (unsere Referenz bzw. FP16-Blockskala im GGUF) und mit auf BF16 gerundetem `l` (wie LittleBits Loader). Der veröffentlichte Wert von 24,48 (WikiText-2) stammt aus LittleBits Inferenz, also mit BF16-gerundetem `l`. Vergleichbar mit 24,48 ist nur die BF16-Variante. Die Differenz der beiden zeigt den Effekt der Rundung selbst. Die BF16-Variante lässt sich im GGUF erzeugen, wenn der Konverter `l` vor dem Schreiben auf BF16 rundet (Option `--l-round bf16`). In Python gibt es dafür bereits `load_checkpoint(..., l_round=torch.bfloat16)`.

### Offene Punkte
- ROCm für gfx1201 unter WSL (vor Phase 3.3).

### Nächster Schritt (Stand Phase 1, erledigt)
Phase 2, siehe oben.
