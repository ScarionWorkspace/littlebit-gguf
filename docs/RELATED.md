# Verwandte Projekte (Recherche 2026-11-21)

Gesucht auf GitHub nach „littlebit“, „littlebit-2“, „littlebit gguf/llama.cpp/vllm/kernel“, „sub-1-bit llm“, „littlebit.cpp“. Nur gelesen, nichts übernommen.

## Direkte Parallelprojekte (llama.cpp + `Q1_0`)

| Projekt | Seit | Lizenz | Stand laut README | Format |
|---|---|---|---|---|
| [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) | 2026-11-20 | MIT | CPU-Patch für Llama/Qwen2/Qwen3, Konverter mit Embedding-Fallbacks, QAT-Notebook (Kaggle), Patch-Datei auf llama.cpp `6aee508` | `lb_u/lb_v/lb_g` + `_r`, `h`/`l` in den Q1_0-Blockskalen, **`V` auf `r_pad` aufgefüllt**, Metadaten `littlebit.format_version`, `littlebit.eff_bits` |
| [Cybertiron/llama.cpp-littlebit](https://github.com/Cybertiron/llama.cpp-littlebit) | 2026-11-20 | MIT | CUDA + CPU (RTX 3090), Qwen3 und Qwen3.5/3.8 (DeltaNet). QAT eines Qwen3.8-27B läuft. Konverter mit Index-Probes durch `convert_hf_to_gguf.py` (überträgt Permutationen exakt) | `lb_u/lb_v` mit Blockskala 1,0, **Skalen als drei eigene F32-Vektoren** (`lb_s_v2/lb_s_mid/lb_s_u1`), Residual `lb_vr/lb_ur` |
| **dieses Projekt** | 2026-11-20 | MIT | CPU + HIP (RX 9070 XT) geprüft gegen Referenz (Toleranzstufen), PPL reproduziert die Modellkarte, Baseline-Vergleich | `lb_u/lb_v/lb_g` + `_r`, `h`/`l` in den Blockskalen, `V` **nicht** aufgefüllt (`ggml_pad` im Graph), Metadaten `littlebit.version`, `littlebit.l_round` |

Beobachtungen:
- Alle drei setzen am selben Punkt an (`build_lora_mm`, Stellvertreter-Tensor, `Q1_0`, kein neuer ggml-Typ). Die Formate sind aber **nicht austauschbar**: gleiche Namen bei GeniusDevam und uns, aber unterschiedliche Shapes von `lb_v` und andere Metadatenschlüssel. Cybertiron hat andere Namen und eine andere Skalenablage. Unser Loader würde GeniusDevam-Dateien vermutlich laden (dort ist `r = r_pad`), umgekehrt nicht.
- Cybertiron bestätigt unabhängig unser Geschwindigkeitsproblem: „each linear launches about 11 small kernels“. Bei uns sind es 7 pro Pfad, weil `h`/`l` in den Blockskalen stecken. Das spricht für eine fusionierte Op.
- Für einen Upstream-Vorschlag an llama.cpp sollte es **ein** Format geben. Abstimmung mit den beiden anderen Projekten wäre sinnvoll (Issue/Discussion), braucht aber die Zustimmung des Nutzers, weil es nach außen geht.

## Stand 2026-11-21 abends

- **Drittes Parallelprojekt:** [luben93/littlebit-llamacpp](https://github.com/luben93/littlebit-llamacpp) (64-Zeilen-Patch, CUDA ohne Backend-Änderung, PPL mit llama-perplexity-Methode 22,31 bei 1457 MB). luben93 hat die Tensor-Namen von GeniusDevam übernommen. Beide stimmen sich in [GeniusDevam/littlebit-gguf#1](https://github.com/GeniusDevam/littlebit-gguf/issues/1) ab.
- **Upstream:** luben93 hat [ggml-org/llama.cpp Discussion #32671](https://github.com/ggml-org/llama.cpp/discussions/32671) eröffnet („Show and tell“). Bisher keine Antwort der Maintainer. Dort **keine KI-geschriebenen Beiträge** (`AGENTS.md` von llama.cpp). Wenn, dann schreibt der Nutzer selbst.
- **Kompatibilität geprüft:** Eine GGUF aus GeniusDevams Konverter (Commit `893866f`, `--embd-type f16`, separater venv mit transformers 5) lädt in unserem llama.cpp-Build und erzeugt denselben Text. Die Fusion greift auch bei deren Layout (aufgefülltes `lb_v`, kein PAD-Knoten): 76 → 110 t/s (unsere F16-Datei: 72 → 110). KL gegen die exakte Referenz 4,0e-7, Stufe 1 und 2 bestanden (60 Positionen, `results/llamacpp/check_geniusdevam-format_hip_fused.json`). Deren Datei hat 3086 MB, unsere 3059 MB (unaufgefülltes `lb_v`).
- **Größere Modelle:** LittleBitLLM plant [Qwen3-14B @ 0,55 bpw mit 5 Epochen](https://huggingface.co/LittleBitLLM/littlebit) nach der 8B-Auswertung. Das 8B brauchte für 0,5 Epochen ~8,5 h auf 1× H200, eine volle Epoche also ~17 GPU-Stunden.

## Nützliche Nachbarprojekte

| Projekt | Lizenz | Nutzen für uns |
|---|---|---|
| [GeniusDevam/littlebit.cpp](https://github.com/GeniusDevam/littlebit.cpp) | MIT | CPU-Kernels direkt auf den gepackten Bits (AVX2/AVX-512F, Lookup-Table), Prefill in FP32. Vorlage bzw. Vergleich für einen eigenen skalenlosen Typ oder eine fusionierte CPU-Op |
| [mgiachetti/little-bit](https://github.com/mgiachetti/little-bit) | MIT | Eigene PyTorch-Implementierung mit „sink-aware“ Verlust: Qwen3-0.6B @ 0,55 bpw → PPL 30,64 nach ~1 Mrd. Tokens (72 h, eine L40S). Rezept für ein eigenes QAT eines kleinen Modells. Checkpoints sind nicht veröffentlicht (keine Releases, kein HF-Link gefunden) |
| [bhubbard/littlebit-rs](https://github.com/bhubbard/littlebit-rs) | „other“ (vermutlich CC BY-NC, Port) | Rust-Port mit gepacktem Matmul, Streaming-Quantisierung großer Modelle. Nur zum Lesen, Lizenz vor jeder Übernahme klären |
| [donicrosby/quantibbler](https://github.com/donicrosby/quantibbler) | Apache-2.0 | Verwandte Methode (binäre latente Faktorisierung, PTQ + QAT) mit Rust-Kernels und OpenAI-kompatiblem Server, verweist auch auf NanoQuant (arXiv 2602.06694) |
| [christopherrobertbrooks-tech/littlebit-bench](https://github.com/christopherrobertbrooks-tech/littlebit-bench) | CC BY-NC 4.0 | Bereits in Phase 0 ausgewertet |

Nicht gefunden: das in einer HF-Modellkarte erwähnte vLLM-Plugin `littlebit-vllm` (weder Repo noch Code-Treffer).
