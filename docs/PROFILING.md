# Profiling: Generierung auf der RX 9070 XT (2026-11-21)

Modell: LittleBit Qwen3-8B 0,55 bpw, Embeddings Q4_K, lm_head Q6_K (1364 MiB). Gemessen: tg128 ≈ 69 t/s → **≈ 14,5 ms pro Token**.
Zum Vergleich: dichtes Qwen3-8B Q4_K_M (4,8 GB) 102 t/s (9,8 ms), Qwen3-1.7B Q4_K_M (1,2 GB) 241 t/s.

Unter WSL laufen die ROCm-Profiler nicht. Werkzeug ist deshalb `test-backend-ops perf` mit den echten 8B-Shapes. Diese Fälle sind in unserem llama.cpp-Fork ergänzt, ebenso Korrektheitsfälle: 101/101 `Q1_0`-Tests grün auf ROCm.

## Messung pro Matmul (n = 1, also ein Token)

| Projektion | V-Stufe (Q1_0) | U-Stufe (Q1_0) | dichtes Q4_K |
|---|---:|---:|---:|
| q / o (4096→4096, r 544) | 5,9 µs | 6,3 µs | 21,0 µs |
| k / v (4096→1024, r 208) | 5,6 µs | 4,9 µs | 9,1 µs |
| gate / up (4096→12288, r 824) | 6,2 µs | 11,0 µs | 61,0 µs |
| down (12288→4096, r 824) | 8,9 µs | 6,8 µs | 45,9 µs |

- Pro Block: 28 `Q1_0`-Matmuls = **191 µs**, dicht Q4_K: 7 Matmuls = 228 µs. Die Faktoren sind 6× kleiner, die Zeit aber nur 16 % kürzer.
- Effektive Bandbreite der `Q1_0`-Matmuls: **80 GB/s** bei ~640 GB/s möglich. Jede Matmul ist so klein (z. B. 313 KB für die V-Stufe von `q`), dass sie praktisch nur aus der festen Mindestdauer eines Kernels (~5 µs) besteht.

## Aufteilung pro Token (Schätzung aus den Messungen)

| Teil | Zeit |
|---|---:|
| 28 × 36 = 1008 `Q1_0`-Matmuls | 6,9 ms |
| zusätzliche kleine Ops: `mul(g)`, `pad`, `add` (Residual), ca. 35 pro Block = 1260 Kernels | ≈ 6 ms (Rest) |
| Attention, Norms, lm_head usw. (wie beim dichten Modell) | ≈ 1,6 ms |
| **gemessen** | **14,5 ms** |

**Ergebnis: Die Generierung ist durch die Anzahl der Kernels begrenzt, nicht durch die Speicherbandbreite.** Pro Token laufen ~2300 Kernels, beim dichten Modell ~400. Die Theoretische Grenze aus der Bandbreite (~1 GB pro Token) läge bei ~1,6 ms.

## Optionen

| | Was | Erwartung (grob) | Aufwand | Upstream / Ollama |
|---|---|---|---|---|
| **1. Fusion im CUDA/HIP-Backend** | Das Muster `mul(g) → mul_mat(Q1_0) → pad → mul_mat(Q1_0)` (+ Residual, `add`) erkennt der Backend-Graph-Optimierer (wie die bestehenden Fusionen, z. B. `rms_norm+mul`) und führt es als zwei eigene Kernels aus: Stufe A rechnet beide latenten Vektoren mit `g`/`g_R` direkt im Skalarprodukt, Stufe B `[U | U_R]·[t; t_R]`. Kein neuer ggml-Typ, keine neue Op, keine Formatänderung | ~14 statt ~63 Kernels pro Block, also eher ~5 ms pro Token → **~150–200 t/s** | mittel: zwei HIP/CUDA-Kernels + Mustererkennung, Tests in `test-backend-ops` | gut: rein backend-intern, Dateien unverändert, Ollama profitiert automatisch |
| 2. Zusätzlich Projektionen mit gleichem Eingang bündeln | q/k/v (und gate/up) haben denselben Eingang. Stufe A für alle drei in einem Kernel | weitere ~2× weniger Kernels → **~250 t/s** möglich | höher (gruppierter Kernel, variable Ranks) | gut, gleiches Prinzip |
| 3. Eigener Typ ohne Blockskalen | spart 11 % der Faktorgröße (0,648 → ~0,58 bpw linear) | **keine** Beschleunigung, da nicht bandbreitenbegrenzt | hoch (~25 Dateien, alle Backends) | schlecht |
| 4. Nur Graph-Umbau mit bestehenden Ops | z. B. `pad` vermeiden durch aufgefüllte `V` (wie GeniusDevam), Residual-Matmuls zusammenlegen | ~10–20 % | klein | trivial |

## Empfehlung

**Option 1, danach Option 2.** Option 3 bringt für die Geschwindigkeit nichts und kommt erst in Frage, wenn die Dateigröße das Problem ist. Option 4 als schneller Vorab-Gewinn ist möglich, ändert aber das Datei-Layout von `V` (aufgefüllt). Weil die Fusion ohnehin intern alles abdeckt, würde ich Option 4 überspringen.

Vorgehen bei Option 1:
1. Fused-Kernel zuerst als Referenz in Python (Simulation mit `Numerics.llamacpp_q1_0`) und als `test-backend-ops`-Fall gegen die ungefusionierte Kette (Toleranz: Q8-Rauschboden).
2. HIP/CUDA-Kernel (ein Code für beide, RDNA4 Wave32), Mustererkennung im Backend.
3. Messung: `test-backend-ops perf`, `llama-bench`, die Prüfungen Stufe 1/2 und Block 0 auf der GPU.

Für die Prompt-Verarbeitung sind die Matmuls selbst schon schneller als dicht (n = 512: 1,1 ms gegen 3,1 ms pro Block und Pfad). Die gemessenen pp512-Werte schwanken aber stark (1836–3093 t/s). Das wird vor jeder Aussage mit mehr Wiederholungen nachgemessen.
