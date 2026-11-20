# Phase 0 – Analyse

Stand: 2026-11-20. Gelesene Stände (shallow clones unter `%USERPROFILE%\littlebit-work\ref\`, nur gelesen, nicht verändert):

| Repo | Commit | Datum |
|---|---|---|
| SamsungLabs/LittleBit | `42d658b` | 2026-10-09 |
| christopherrobertbrooks-tech/littlebit-bench | `da8b063` | 2026-10-19 |
| ggml-org/llama.cpp | `374d90a` | 2026-11-20 |
| ollama/ollama | `0b243a1` (pinnt llama.cpp `b12257`) | 2026-11-20 |

Kurzfassung:

1. Die LittleBit-Schicht ist exakt `y = ((((x ⊙ v2) · Vᵀ) ⊙ (v1 ⊙ u2)) · Uᵀ) ⊙ u1` plus ein zweiter, gleich gebauter Residual-Pfad. Komprimiert werden alle `nn.Linear` außer `lm_head`. Embeddings und Norms bleiben BF16.
2. **Es gibt jetzt einen trainierten 8B-Checkpoint**: `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` (Community, kein Samsung-Release, nur 0,5 Epochen). Samsung selbst veröffentlicht keine Checkpoints (Issue #1, „internal policies“).
3. **llama.cpp hat inzwischen `GGML_TYPE_Q1_0`**, einen 1-Bit-Vorzeichentyp mit Kernels für CPU (AVX2), CUDA/HIP, Vulkan, Metal und weitere Backends. Damit lassen sich die LittleBit-Faktoren **ohne neuen ggml-Typ** speichern. Die Skalen `h` und `l` stecken dabei verlustfrei in den Blockskalen.
4. **Ollama hat seine eigene Go/ggml-Engine aufgegeben.** Es baut das upstream-`llama-server` aus einer gepinnten llama.cpp-Version (CMake FetchContent plus Kompatibilitäts-Patch). Die Go-Seite kennt `Q1_0` schon. Die Ollama-Integration schrumpft damit auf „Ollama gegen unseren llama.cpp-Fork bauen“.

Das ändert den Plan aus `CLAUDE.md` deutlich. Abschnitt 5 nennt die Entscheidungen, die dein OK brauchen.

---

## 1. LittleBit: exakte Parametrisierung

Quelle: `quantization/modules/littlebit.py`, `quantization/utils/quant_util.py`, `quantization/utils/binary_packer.py`, `quantization/hub.py`.

### 1.1 Forward-Pass (`LittleBitLinear._compute_forward`)

```python
Vq = sign(V); Uq = sign(U)          # sign(0) := +1
y  = ((((x * v2) @ Vq.T) * (v1 * u2)) @ Uq.T) * u1
if residual:
    y += ((((x * v2_R) @ V_Rq.T) * (v1_R * u2_R)) @ U_Rq.T) * u1_R
if bias is not None: y += bias
```

Abgleich mit der Notation in `CLAUDE.md` und den Papers (LittleBit-2 Eq. 1: `Ŵ = diag(h) U_b diag(l) V_bᵀ diag(g)`):

| Paper | Code | Shape | Rolle |
|---|---|---|---|
| `g` | `v2` | `[1, in]` | Eingangsskala |
| `V_b` | `V` (→ `V_packed`) | `[r, in]` | Vorzeichen ±1 |
| `l` | `v1 * u2` | `[1, r]` | Skala im latenten Raum (Produkt zweier gespeicherter Vektoren) |
| `U_b` | `U` (→ `U_packed`) | `[out, r]` | Vorzeichen ±1 |
| `h` | `u1` | `[1, out]` | Ausgangsskala |

Die Vermutung in `CLAUDE.md` (`y = h ⊙ (U_sign · (l ⊙ (V_signᵀ · (g ⊙ x))))` plus Residual mit eigenen Faktoren) stimmt. Zu beachten:

- `l` wird im Checkpoint als **zwei** Vektoren `v1` und `u2` gespeichert. Für die Inferenz zählt nur ihr Produkt (siehe 1.3).
- Der Residual-Pfad hat **eigene** `g_R`, `l_R`, `h_R`, also nicht nur eigene U/V.
- Bias wird unterstützt. Bei Qwen3 und Llama gibt es keinen.
- LittleBit-2 (`use_itq`) ändert nur die Initialisierung. Das Inferenzformat ist identisch.

### 1.2 Rank (`split_dim`) pro Schicht

Der Rank wird **pro Schicht** aus `eff_bit` berechnet (`_estimate_split_dim`, mit Residual):

```
r = floor( (in·out·eff_bit − 32·(in+out)) / (2·(in+out+16)) )   → auf Vielfaches von 8 abgerundet, min. min_split_dim
```

`k_proj` und `v_proj` werden zusätzlich mit `kv_factor` multipliziert. Der Rank ist also je nach Projektion verschieden. Bei littlebit-bench (`--eff_bit_layers`) ist er sogar je nach Block verschieden. **Das GGUF-Format darf den Rank deshalb nicht global annehmen.** Er ergibt sich aus der Tensor-Shape. Gemessen im 8B-Checkpoint: q/o = 544, k/v = 208, gate/up/down = 824.

Die Bitrechnung zählt die FP16-Skalen mit: `eff = (2r(in+out+16) + 32(in+out)) / (in·out)`. Das ist identisch mit LittleBit-2 Appendix H.3, Eq. 25.

### 1.3 Checkpoint-Format (gepackt, `state_dict()`)

Pro komprimierter Schicht, Präfix z. B. `model.layers.0.mlp.down_proj.`:

| Name | dtype | Shape | Inhalt |
|---|---|---|---|
| `U_packed`, `U_R_packed` | I32 | `[out, ceil(r/32)]` | Vorzeichen, zeilenweise, **LSB first**, **Bit 1 = −1**, Bit 0 = +1, Padding mit +1 |
| `V_packed`, `V_R_packed` | I32 | `[r, ceil(in/32)]` | dto. |
| `U_shape`, `V_shape`, `*_R_shape` | I64 | `[2]` | Originalshape |
| `u1, u2, v1, v2` (+ `_R`) | BF16 | `[1, n]` | Skalen |
| `_split_dim_final` | I64 | `[]` | Rank |
| `_eff_bit_target`, `_eff_bit_actual` | BF16 | `[]` | Buchhaltung |

Unkomprimiert bleiben `model.embed_tokens.weight`, `lm_head.weight`, alle Norms (bei Qwen3 auch `q_norm` und `k_norm`). `lm_head` wird ausdrücklich ausgeschlossen (`exclude_names=["lm_head"]`). Embeddings sind kein `nn.Linear` und daher nie betroffen.

`littlebit_config.json` (Beispiel 8B-Checkpoint):
`{"quant_func": "SmoothSign", "eff_bit": 0.55, "split_dim": 1024, "residual": true, "kv_factor": 1.0, "min_split_dim": 8, "quant_mod": "LittleBitLinear", "use_itq": true, "itq_n_iter": 50}`.
Dieselben Schlüssel stehen zusätzlich in `config.json`.

Bekannter Bug: `binary_unpacker` hat bis 2026-10-08 in die falsche Richtung geschoben (Issue #21, Fix in PR #22, in `42d658b` enthalten). Der **Packer** war immer korrekt. Unser Konverter liest die Bits selbst und hängt nicht vom Unpacker ab.

### 1.4 Numerik (aus littlebit-bench, Issue #24)

- In FP16 können die Zwischensummen der Binär-Matmuls überlaufen (`down_proj`, bis zu `in` Terme vor der Skalierung). Für uns heißt das: **in FP32 akkumulieren.** Auf der GTX 1080 ohnehin vorgesehen.
- BF16→FP16-Konvertierung der Skalen ist unkritisch (keine Werte außerhalb des FP16-Bereichs gemessen).

### 1.5 Paper-Angaben (geprüft, Werte aus Table 1 noch gegen das PDF abzugleichen)

WikiText-2-PPL, Llama-3-8B (FP16: 6,10): LittleBit-2 11,53 @ 1,0 bpw · **14,01 @ 0,55** · **23,74 @ 0,1**. LittleBit (v1): 16,30 / 18,47 / 26,11. Die Community-Angabe „~14 / ~24“ stimmt also für LittleBit-2.
Zero-Shot bei 0,1 bpw (5 Tasks): Llama-3-8B 41,80 (v1) gegen 41,64 (v2), hier also leicht schlechter. Llama-2 verbessert sich leicht.

Speed: 11,6× ist die **Kernel**-Latenz (GEMV, eigener CUDA-Kernel, Llama-2-70B-MLP, A100, 0,1 bpw). 2,46× ist **End-to-End**-Decoding (Llama-2-7B, 0,1 bpw, Batch 1, 82,6 → 203,2 tok/s). Den Kernel beschreibt keines der Papers (kein Bit-Layout, nichts zu XNOR/Popcount oder Fusion). Er ist nicht im Repo (bestätigt in Issue #18). Unser Kernel ist also Neuland. Das Risiko aus `CLAUDE.md` stimmt.

---

## 2. Verfügbare Checkpoints (Stand 2026-11-20)

| Repo | Basis | bpw | QAT? | Format | Bewertung |
|---|---|---|---|---|---|
| `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` | Qwen3-8B | 0,55, Residual, LittleBit-2 | **ja**, 0,5 Epochen, 1× H200 | genau das gepackte LittleBit-Format, 2,97 GB | **Einziger trainierter Kandidat.** PPL WikiText-2 24,48 (Basis 9,73). ARC-C auf Zufallsniveau. CC BY-NC 4.0. Community-Release, nicht von Samsung |
| `Haverbex/Qwen3.8-27B-LittleBit-{0.1,0.55,1}bit` | Qwen3.8-27B-Uncensored | 0,1–1 | **nein** (0 Schritte) | eigenes Layout, nur MLP | Unbrauchbar: wählt bei 346/346 Fragen „A“. Hybrid-Architektur |
| `dellathefella/Qwen3.8-27B-LittleBit-2-0.55bpw` | Qwen3.8-27B | 0,55 | **nein** (nur Init) | ±1 als bf16, fusionierte QKV, vLLM-Plugin `littlebit-vllm` | Unbrauchbar für Qualität. Hinweis: es existiert ein vLLM-Plugin, mögliche Vorarbeit für Kernels, noch nicht angesehen |
| SamsungLabs | – | – | – | – | Issue #1 geschlossen: „unable to share the trained checkpoints publicly due to internal policies“ |
| littlebit-bench (Qwen3-0.6B/1.7B) | – | – | ja | – | Gewichte **nicht** veröffentlicht |

Folgen:

- **Benchmark-Modell:** Llama-3-8B (wie in `CLAUDE.md` gewünscht) hat keinen Checkpoint. Der Qwen3-8B-Checkpoint ist die realistische Wahl. Vorteil: gleiche Architektur wie das Korrektheitsmodell Qwen3-0.6B. Nachteil: kaum mit Paper-Zahlen vergleichbar (anderes Modell, nur 0,5 statt 5 Epochen). Vergleichen müssen wir deshalb selbst gegen Q2_K, IQ1_S und TQ1_0 von Qwen3-8B.
- **Korrektheitsmodell Qwen3-0.6B:** Dafür gibt es keinen Checkpoint. Für Korrektheit braucht es aber **keinen trainierten** Checkpoint: Eine LittleBit-Initialisierung ohne QAT (SVD + Joint-ITQ, `main.py` mit 0 Trainingsschritten) hat exakt dasselbe Format. Der Abgleich „unsere Inferenz = PyTorch-Referenz“ funktioniert damit genauso. Aufwand: Minuten statt Tage. QAT für 0.6B wäre nur für Qualitätsaussagen nötig, und die liefert bei 0.6B ohnehin nichts (Embeddings sind 92 % der Datei).
- QAT auf der RX 9070 XT (16 GB): Laut littlebit-bench braucht 0.6B ca. 2.450 tok/s auf einer V100. Das volle Rezept (~850M Tokens) würde etwa 4 Tage dauern. Bei 8B: Lehrer plus Schüler plus Optimizer passen nicht in 16 GB. **Nicht machbar.** Eigenes QAT nur, falls wir später Qualität bei 0.6B/1.7B messen wollen. ROCm unter WSL ist dafür noch ungeprüft.

---

## 3. llama.cpp

### 3.1 Neuer ggml-Typ: wo es ihn bräuchte

Muster `TQ1_0`/`Q1_0` (Typ-ID 41, der jüngste Neuzugang):
`ggml/include/ggml.h` (Enum, `GGML_TYPE_COUNT`) · `ggml/src/ggml-common.h` (Block-Struct plus `static_assert`) · `ggml/src/ggml.c` (`type_traits`: Name, Blockgröße, Typgröße, `to_float`, `from_float_ref`) · `ggml/src/ggml-quants.{c,h}` (Referenz-(De-)Quantisierung) · `ggml/src/ggml-cpu/{ggml-cpu.c,quants.c,quants.h,arch/x86/quants.c,arch-fallback.h,ops.cpp,repack.*}` (`type_traits_cpu`: `vec_dot`, `vec_dot_type`) · CUDA: `convert.cu`, `dequantize.cuh`, `vecdotq.cuh`, `mmvq.cu`, `mmq.cu/.cuh` plus `mmq-config-*.cuh` und Template-Instanzen · Vulkan-Shader · `gguf-py/gguf/constants.py` plus `quants.py` · `src/llama-model-loader.cpp`, `src/llama-quant.cpp` · `tests/test-backend-ops.cpp`, `tests/test-quantize-fns.cpp`. Das sind rund 25 Dateien pro Backend-Satz. Ein eigener Typ ist also ein großer, schwer upstream-fähiger Patch.

### 3.2 Der vorhandene Typ `Q1_0`

```c
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;   // 18 Byte / 128 Werte = 1,125 bpw
// dequant: Bit (LSB first) == 1 → +d, == 0 → −d
```

Abdeckung: CPU (generisch plus AVX2, **kein eigener AVX-512-Pfad**), CUDA `mmvq` und `mmq` (damit auch HIP/ROCm, mit RDNA4-Config), Vulkan, Metal, SYCL, OpenCL, WebGPU, `gguf-py` (Quantisierer vorhanden), `test-backend-ops`. **Aktivierungen werden für das Skalarprodukt nach Q8_0 (CPU) bzw. Q8_1 (CUDA) quantisiert.** Das ist eine Abweichung zur FP-Referenz, die wir in Phase 1/3 messen müssen.

Abbildung von LittleBit auf `Q1_0` (exakt):

- Bit invertieren (LittleBit: 1 = −1, Q1_0: 1 = +d).
- `V` als ggml-Tensor `ne = [in, r]` (Zeile = Rank-Index, Länge `in`). `in` ist bei allen Zielmodellen ein Vielfaches von 128. **Blockskala `d` = `l[i]` = `v1[i]·u2[i]`** für alle Blöcke der Zeile `i`.
- `U` als `ne = [r_pad, out]` mit `r_pad` = `r` auf ein Vielfaches von 128 aufgerundet. **Blockskala `d` = `h[o]` = `u1[o]`.** Die Padding-Spalten von U treffen auf Padding-Zeilen von V. Deren Skala wird 0 gesetzt, sie tragen also exakt 0 bei.
- Übrig bleibt **nur `g`** (Eingangsskala) als F32/F16-Tensor.

Pro Pfad ergibt das `ggml_mul(x, g) → ggml_mul_mat(V, ·) → ggml_mul_mat(U, ·)`. Dazu kommen der Residual-Pfad und ein `ggml_add`: insgesamt 7 Ops statt einer Matmul, alle mit bestehenden, optimierten Kernels.

Kosten: 12,5 % Skalen-Overhead auf die Vorzeichen plus Padding von U. Für den 8B-Checkpoint: Vorzeichen 442 MiB roh, in Q1_0 grob 500–550 MiB. Lineare Schichten statt 0,548 effektiv ca. 0,6–0,67 bpw. Ein eigener skalenloser Typ würde das zurückholen (siehe 5).

### 3.3 Integrationspunkt im Graph

- Alle Gewichts-Matmuls der Qwen3-/Llama-Familie laufen über **`llm_graph_context::build_lora_mm(w, cur, w_s)`** (`src/llama-graph.cpp:1808`). Aufrufer: `build_qkv` (wq/wk/wv bzw. wqkv), `build_attn` (wo), `build_ffn` (up/gate/down), lm_head (`src/models/qwen3.cpp`).
- `build_lora_mm` hat schon zwei passende Haken: `w_s` (Ausgangsskala pro Zeile, z. B. `layer.wq_s`) und `build_hadamard_input(w, cur)` (Eingangstransformation). LittleBit ist dazu verwandt („Eingangs-Transform + Matmul + Ausgangsskala“).
- Modelldefinition Qwen3: `src/models/qwen3.cpp` (`load_arch_tensors` mit `create_tensor_qkv`, `create_tensor`). Tensor-Namen kommen aus `src/llama-arch.{h,cpp}`.
- Vorschlag: eine Struktur `llama_lb_weight { U, V, g, U_r, V_r, g_r }` pro Projektion im `llama_layer`, geladen als optionale Tensoren (`TENSOR_NOT_REQUIRED`), und eine Hilfsfunktion `build_lb_mm(lb, cur)`. In `build_lora_mm` wird abgezweigt, wenn für `w` LittleBit-Faktoren existieren. Offen ist, wie ein „fehlendes“ dichtes `w` signalisiert wird. Das klären wir beim Implementieren im Code (z. B. Lookup über den Tensor-Namen wie bei LoRA `get_weight(w)`, oder `wq == nullptr` plus `wq_lb`).

### 3.4 Konverter

`convert_hf_to_gguf.py` hat eine Qwen3-Klasse. Ein eigener `convert_littlebit_to_gguf.py` im Haupt-Repo kann gguf-py nutzen (`GGUFWriter`, `quants.Q1_0`) und Metadaten, Tokenizer usw. aus der Qwen3-Klasse übernehmen. Embeddings und lm_head lassen sich danach mit `llama-quantize` auf Q4_K/Q6_K bringen. Dafür muss `llama-quant.cpp` die LittleBit-Tensoren unangetastet lassen (prüfen).

---

## 4. Ollama

- **Die Go/ggml-Engine ist weg.** `ml/` enthält nur noch Gerätelogik, `model/` nur Parser und Renderer. Ollama startet für GGUF-Modelle das upstream-**`llama-server`** (`llm/llama_server.go`).
- llama.cpp wird **nicht mehr vendored**. `llama/server/CMakeLists.txt` holt es per `FetchContent` von `github.com/ggml-org/llama.cpp` mit dem Tag aus `LLAMA_CPP_VERSION` (aktuell `b12257`) und wendet `llama/compat/*.patch` an (Hooks für Altformate).
- Eigene llama.cpp-Quelle einbinden:
  - `OLLAMA_LLAMA_CPP_SOURCE=<pfad>`: lokaler Baum, Compat-Patch wird übersprungen. Ideal für die Entwicklung.
  - `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`: lokaler Baum, Compat-Patch wird angewendet.
  - Für einen sauberen Fork: `GIT_REPOSITORY`/`LLAMA_CPP_VERSION` auf unseren Fork und Branch zeigen lassen (ein kleiner Patch), oder unsere Änderung als `llama/compat/002-littlebit.patch` beilegen.
- Go-GGUF-Parser (`fs/gguf`): kennt `Q1_0` schon (`blockSize 128`, `typeSize 18`). Speicherabschätzung und Anzeige funktionieren also. Eine Architektur-Whitelist gibt es nur im MLX-Runner (`mlxrunner/model/model.go`), nicht im llama-server-Pfad.
- **Folge:** Mit Variante A (Q1_0) sind auf der Go-Seite vermutlich **keine** Änderungen nötig. Phase 4 besteht dann aus Ollama bauen mit unserem llama.cpp, `ollama create` und Vergleichen. Die Basis muss zum Pin passen: Unser llama.cpp-Branch muss auf `b12257` (oder dem dann aktuellen Pin) aufsetzen, nicht auf `master`, damit die Compat-Patches greifen.

---

## 5. Empfehlung und Entscheidungen

### 5.1 Format (braucht dein OK)

| | **A: bestehendes `Q1_0`** (empfohlen) | B: neuer Typ `B1` (wie in `CLAUDE.md`) |
|---|---|---|
| Neue ggml-Typen | keine | einer, ~25 Dateien, alle Backends |
| Backends sofort | CPU (AVX2), CUDA/HIP, Vulkan, Metal … | nur was wir schreiben |
| Ollama Go-Seite | keine Änderung | Typ nachtragen |
| Upstream-Chance | gut (nur Graph plus Konverter) | gering |
| Größe Faktoren | +12,5 % Skalen + U-Padding (8B: ~0,6–0,67 statt 0,55 bpw linear) | exakt LittleBit |
| Numerik | Aktivierungen Q8 (Abweichung messen) | frei wählbar (F32) |
| Geschwindigkeit | 7 Einzel-Ops pro Schicht | später fusionierbar |

Empfehlung: **A für den MVP und den ersten Ollama-Lauf.** Danach messen. Wenn Größe oder Speed es rechtfertigen, als Phase-3b einen fusionierten Op bzw. einen skalenlosen Typ ergänzen. Das Datei-Layout (U mit Zeilenskala h, V mit Zeilenskala l, g separat) bleibt dabei kompatibel. Phase 3.1 aus `CLAUDE.md` („Faktoren als F16 ±1“) fällt so praktisch mit 3.2 zusammen.

Tensor-Namen (Vorschlag): `blk.N.attn_q.lb_u`, `.lb_v`, `.lb_g`, Residual mit `.lb_u_r`, `.lb_v_r`, `.lb_g_r`. `h` und `l` stecken in den Blockskalen. Für die Rückverfolgbarkeit legen wir zusätzlich `littlebit.version`, `littlebit.residual`, `littlebit.eff_bit` als Metadaten ab. Der Rank kommt aus den Shapes.

### 5.2 Testmodelle (braucht dein OK)

- Korrektheit: **Qwen3-0.6B, LittleBit-2-Init ohne QAT** (selbst erzeugt, CPU/GPU, Minuten). Optional zusätzlich ein Init mit `residual=False`, um beide Pfade abzudecken.
- Benchmark: **`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`** statt Llama-3-8B. Vergleich gegen Q2_K, IQ2_XXS, IQ1_S und TQ1_0 von Qwen3-8B sowie gegen Qwen3-1.7B/4B (Q4_K_M) bei gleicher Dateigröße.

### 5.3 Minimaler Pfad zum ersten lauffähigen Modell

1. Umgebung: WSL2/Ubuntu 24.04 plus Toolchain (siehe `SETUP.md`, **fehlt komplett**).
2. Phase 1: Python-Referenz (numpy/torch) liest den gepackten Checkpoint direkt, erzeugt Golden Files (0.6B-Init und 8B, jeweils eine Schicht plus Logits für feste Prompts) und gleicht gegen `LittleBit/eval.py`-Inferenz ab.
3. Phase 2: `convert_littlebit_to_gguf.py` → GGUF (Q1_0-Faktoren, F32-`g`, Embeddings F16) und Roundtrip-Test gegen die Referenz.
4. Phase 3: llama.cpp-Branch `littlebit` (Basis = Ollama-Pin): `build_lb_mm` plus Laden in `qwen3.cpp`, Logits gegen Golden Files, `test-backend-ops` für Q1_0 bei unseren Shapes (Rank nicht durch 128 teilbar → Padding) auf CPU und HIP.
5. Phase 4: Ollama mit `OLLAMA_LLAMA_CPP_SOURCE` bauen → `ollama create` → `ollama run`.

### 5.4 Offene Fragen und Risiken

- Q8-Aktivierungsquantisierung im latenten Raum (`r` = 208–824) kann die Abweichung zur FP-Referenz vergrößern. Toleranz in Phase 1/3 festlegen. Notfalls die Matmul für `B1`/F16 erzwingen.
- `llama-quantize`: Behandelt es fremde Q1_0-Tensoren korrekt (unangetastet lassen)? Im Code prüfen.
- Kein AVX-512-Pfad für `Q1_0` auf x86. Wäre ein guter, upstream-fähiger Beitrag (Zen 4).
- Das vLLM-Plugin `littlebit-vllm` (aus der dellathefella-Modellkarte) ansehen: möglicherweise vorhandene GPU-Kernels, Lizenz prüfen.
- Lizenz: LittleBit-Code CC BY-NC 4.0. Wir übernehmen **keinen** Code von dort. Der Konverter liest nur das Datenformat (Bit-Layout aus Dokumentation und Messung). Die littlebit-bench-Patches betreffen nur das Training und sind für uns derzeit nicht nötig.
