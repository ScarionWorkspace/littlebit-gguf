# GGUF-Format für LittleBit (`littlebit.format_version` 1 und 2)

Erzeugt von `convert_littlebit_to_gguf.py`. Gelesen von `littlebit_ref/gguf_loader.py` und unserem llama.cpp-Branch `littlebit`.
Die gemeinsame Spezifikation der drei Implementierungen ist [GeniusDevam/littlebit-gguf docs/FORMAT.md](https://github.com/GeniusDevam/littlebit-gguf/blob/main/docs/FORMAT.md#gguf-layout). Diese Seite beschreibt, wie unser Konverter sie umsetzt.
Es gibt **keinen neuen ggml-Typ**: Die Vorzeichenmatrizen sind `GGML_TYPE_Q1_0` (llama.cpp ab `b12257`).

## Mathematik

Pro komprimierter Projektion (z. B. `attn_q`) gibt es zwei Pfade, Haupt- und Residualpfad:

```
y = Σ_S  h_S ⊙ (U_S · (l_S ⊙ (V_S · (g_S ⊙ x))))         S ∈ {"", "_r"}
    U ∈ {±1}^(out×r),  V ∈ {±1}^(r×in),  h ∈ R^out,  l ∈ R^r,  g ∈ R^in
```

Bezug zum LittleBit-Checkpoint: `h = u1`, `l = v1·u2`, `g = v2` (Residual: `*_R`).

## Tensoren

Für jede Projektion `P` ∈ {`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`} in Block `N` und jeden Pfad `S` ∈ {``, `_r`}:

| Name | Typ | ggml `ne` | Inhalt |
|---|---|---|---|
| `blk.N.P.lb_v{S}` | Q1_0 | `[in, r]` | Zeile `i`: Vorzeichen `V[i,:]`, Blockskala `d = fp16(l[i])` (mit Vorzeichen) in allen Blöcken der Zeile |
| `blk.N.P.lb_u{S}` | Q1_0 | `[r_pad, out]` | Zeile `o`: Vorzeichen `U[o,:]`, Blockskala `d = fp16(h[o])` (mit Vorzeichen). Spalten `r … r_pad−1` sind Füllwerte mit gelöschtem Bit (0) |
| `blk.N.P.lb_g{S}` | F32 | `[in]` | `g`, verlustfrei (BF16 → F32) |

- `r_pad` = `r` auf ein Vielfaches von 128 aufgerundet (`QK1_0`). Für `Q1_0` muss die Eingangsdimension (`ne0`) durch 128 teilbar sein. Bei `V` ist das die Hidden-Größe, `V` behält also den echten Rank `r`. Bei `U` ist es der Rank, `U` wird deshalb aufgefüllt.
- Der Rank ergibt sich aus den Shapes (`ne[1]` von `lb_v`). Er darf je Projektion und Block verschieden sein.
- Bit-Konvention von `Q1_0`: LSB zuerst, Bit 1 = `+d`, Bit 0 = `−d`. Bei LittleBit ist es umgekehrt (Bit 1 = −1). Der Konverter schreibt die `Q1_0`-Blöcke direkt (nicht über den gguf-py-Quantisierer, der immer `d ≥ 0` erzeugt). Eine Skala, die in FP16 auf null fällt, bleibt eine Null mit Vorzeichen, wie bei `numpy.astype(float16)`. Damit sind unsere `lb_*`-Tensoren byte-gleich mit denen der beiden anderen Konverter.
- Dateien unseres Konverters vor 2026-11-24 haben `d = fp16(|h|)` bzw. `fp16(|l|)` und das Vorzeichen in den Bits, die Füllspalten von `U` haben Bit 1. Sie ergeben dieselben Werte und bleiben gültig.
- Ein dichtes `blk.N.P.weight` gibt es für komprimierte Projektionen nicht.

## Forward-Pass im ggml-Graph (Phase 3)

```
t = ggml_mul(x, lb_g)                 // F32
t = ggml_mul_mat(lb_v, t)             // [r]: enthält bereits l
t = ggml_pad(t, r_pad - r)            // Nullen → Füllspalten von U tragen 0 bei
y = ggml_mul_mat(lb_u, t)             // [out]: enthält bereits h
(+ dasselbe mit _r, dann ggml_add)
```

## Unverändert aus dem Basismodell

`token_embd.weight`, `output.weight` (fehlt bei gebundenen Embeddings), alle Norms (`attn_norm`, `ffn_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm`), Tokenizer und Hyperparameter. Diese kommen aus dem Qwen3-Konverter von llama.cpp. `--outtype` (f16/bf16/f32) betrifft nur Embeddings und lm_head. `llama-quantize` lässt die `lb_*`-Tensoren unverändert, weil ihre Namen nicht auf `weight` enden. Es quantisiert nur Embeddings und lm_head (geprüft mit Q4_K_M und Q2_K).

## Metadaten

Die ersten vier Schlüssel sind das gemeinsame Format mit [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) und [luben93/littlebit-llamacpp](https://github.com/luben93/littlebit-llamacpp) (abgestimmt in GeniusDevam/littlebit-gguf#1). Die letzten beiden gibt es nur bei uns. Sie liegen unter `littlebit.x.*`, das alle Loader laut Spezifikation ignorieren.

| Schlüssel | Typ | Bedeutung |
|---|---|---|
| `littlebit.format_version` | u32 | `1`: `lb_v` auf `r_pad` aufgefüllt. `2`: `lb_v` mit echtem Rank `r` (Loader füllen den Latent-Vektor auf). Dieser Konverter füllt `lb_v` nie auf und schreibt `1` nur, wenn alle Ranks schon Vielfache von 128 sind (dann sind beide Layouts identisch), sonst `2`. Ältere Loader, die nur `1` kennen, lehnen `2` mit einer klaren Meldung ab |
| `littlebit.residual` | bool | Residualpfad vorhanden (`*_r`-Tensoren) |
| `littlebit.eff_bits` | f32 | Ziel-bpw aus `littlebit_config.json` (`eff_bit`) oder `--eff-bits`; muss endlich und > 0 sein (nur zur Information) |
| `littlebit.use_itq` | bool | Mit LittleBit-2-ITQ initialisiert (`use_itq` aus `littlebit_config.json`, sonst `false`; nur zur Information) |
| `littlebit.x.l_round` | string | Nur bei uns. `none`: `l = v1·u2` exakt, nur FP16-Blockskala. `bf16`: `l` wurde vor dem Schreiben auf BF16 gerundet, wie bei LittleBits eigener Inferenz |
| `littlebit.x.quant_func` | string | Nur bei uns. Trainings-Vorzeichenfunktion (nur zur Information) |

Ältere Dateien dieses Projekts (vor dem gemeinsamen Format) haben stattdessen `littlebit.version = 1` und `littlebit.eff_bit` und kein `use_itq`. Ihr Tensor-Layout entspricht `format_version` 2. Unser llama.cpp-Loader liest beide. Dateien vom 2026-11-23 haben `l_round` und `quant_func` noch ohne `x.`; `scripts/check_llamacpp.py` liest beide Schreibweisen.

## Genauigkeit

- Vorzeichen: bit-exakt. `g`: exakt. `h` und `l`: FP16-gerundet (`l` rel. ≤ 4,8e-4). Im 8B werden 56 von 2,8 Mio. `h`-Werten zu 0 und 3130 subnormal. Gemessener Effekt aller FP16-Skalen auf die PPL: +0,001 %.
- Geprüft durch `tests/test_gguf_roundtrip.py` (0.6B-Init sowie 8B mit `l_round=none` und `bf16`).

## Größe (Qwen3-8B, LittleBit 0.55 bpw)

| Variante | Datei | bpw ganze Datei | bpw Linear-Schichten |
|---|---|---|---|
| Embeddings/lm_head F16 | 2917 MiB | 2,99 | 0,648 |
| `llama-quantize Q4_K_M` (Embeddings Q4_K, lm_head Q6_K) | 1364 MiB | 1,40 | 0,648 |
| `llama-quantize Q2_K` (Embeddings Q2_K, lm_head Q6_K) | 1225 MiB | 1,25 | 0,648 |

Die Linear-Schichten liegen bei 0,648 statt LittleBits 0,548 bpw. Der Grund sind die FP16-Blockskalen von `Q1_0` (+12,5 % auf die Vorzeichen) und das Auffüllen von `U` auf Vielfache von 128 (Ranks 544/208/824 → 640/256/896). Ein eigener, skalenloser Typ würde das zurückholen (mögliche Phase 3b).
