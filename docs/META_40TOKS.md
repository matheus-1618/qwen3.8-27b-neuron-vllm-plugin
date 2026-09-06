# META — Deixar o Qwen3.8-27B "firmão": ~40 tok/s, concorrência real, competitivo

Anotado em 2026-09-05. Base: números medidos em RESULTADOS.md e limitações da §7 do CONTEXT.md.

## Onde estamos → onde queremos chegar

| Métrica | Hoje | Meta |
|---|---|---|
| Decode (1 stream) | 20.7 tok/s | ~40 tok/s |
| TTFT (prompt curto) | ~7 s (bucket único 4096) | ~1-2 s |
| Concorrência | 1 (MNS=1, mais que isso OOM/corrupção) | 4+ requests paralelas |
| Contexto | 4096 (teto do compilador) | 8K-16K (desejável p/ competitivo) |

## A aritmética (por que 40 tok/s é possível)

Decode é bound por leitura de peso da HBM. Hoje cada rank lê 21.79 GB por token
(DeltaNet replicado). A conta de bandwidth:

| Mudança | Bytes/rank | tok/s estimado |
|---|---:|---:|
| Hoje | 21.79 GB | 20.7 (medido) |
| + shardar DeltaNet | 13.45 GB | ~27 (1.3×) |
| + FP8 nos pesos | ~6.7 GB | ~37 (1.8×) |

**Stream único satura em ~37 tok/s** — os 40 "de verdade" vêm de batching:
com MNS>1 o custo de leitura de peso é amortizado entre sequências → agregado
estimado 60-80 tok/s. Ou seja: **a meta exige as duas frentes** (peso menor E concorrência).

## Plano de trabalho (ordem de ataque)

### Fase 1 — grátis, só config (1 dia)
1. **Buckets múltiplos de prefill**: `BUCKETS=512,1024,2048,4096 ./serve.sh`
   → TTFT ~7s → ~1-2s em prompt curto. Custo: 1 NEFF por bucket no 1º boot.
2. **`--enable-prefix-caching`** no 04_serve.sh → grande ganho em uso agent/multi-turno.
3. Re-medir com AIPerf (07_loadtest.sh) pra ter baseline pós-config.

### Fase 2 — shardar o DeltaNet (1-2 dias, risco baixo) ← MELHOR CUSTO/BENEFÍCIO
Os 48 v-heads do DeltaNet estão **replicados** nos 4 ranks ("functional, not optimal"
na docstring do porte original). Shardar (12 heads/rank + all-gather):
- Ganho duplo: ~1.3× decode (20.7 → ~27) **e** +8.34 GB de HBM por core.
- A HBM liberada é o que hoje sufoca o KV cache → pré-requisito prático da Fase 3.
- O padrão de sharding já existe no próprio arquivo (GQA é shardado).
- Validação: suite 5/5 + contagem 1..15 (detector do bug de estado).

### Fase 3 — concorrência de verdade: MNS>1 (1-2 semanas) ← O SALTO GRANDE
Dois bloqueios conhecidos (§7 itens 1-2):
- **Correção**: estado do DeltaNet é indexado por posição no batch (`buffer[:batch_size]`),
  sem mapear para sequence IDs do vLLM → com continuous batching, estado vai pra
  sequência errada (falha silenciosa) + o `copy_` faz broadcast.
  Fix: indexar estado por slot de sequência (mesmo padrão do k_cache com `index_put_`).
- **Memória**: sem a Fase 2 não há HBM pra KV de múltiplas sequências (MNS=4 deu OOM).
Resultado esperado: 4 usuários simultâneos sem serialização; agregado 60-80 tok/s.
Validação: AIPerf CONC="1 2 4" — hoje 5 usuários = 5× espera, 0× throughput.

### Fase 4 — contexto >4096 (3-5 dias) — necessário pra ser "competitivo"
O compilador falha em 8K (status 70): prefill single-shot explode o grafo
(~2300 invocações NKI, chunks dobram com o contexto).
Fix: **mudar o kernel NKI** `deltanet_fused_chunked_fwd` para aceitar estado inicial
→ permite prefill segmentado com grafo pequeno. Exige validação de parity numérica.

### Fase 5 — FP8 per-tensor (2-3 dias, risco médio-alto) — otimização final
Requantizar BF16 → fp8_static_per_tensor offline (o checkpoint FP8 do HF é block-wise,
incompatível). Adaptar caminho FP8 do llama3 (`model_static_fp8.py`) pras nossas camadas.
~1.45× decode. Só vale DEPOIS das fases 2-3 (custa mais e entrega menos que elas).
NÃO fazer: dequantizar FP8→BF16 (inútil) · kv-cache fp8 (só 16/64 camadas têm KV, ganho ~nulo).

## Definição de "competitivo" (critério de saída)
- [ ] ≥ 35-40 tok/s por stream (Fases 2+5) ou ≥ 60 tok/s agregado (Fases 2+3)
- [ ] TTFT ≤ 2 s em prompt curto (Fase 1)
- [ ] 4 requests paralelas sem degradar ITL (Fase 3)
- [ ] contexto ≥ 8K (Fase 4)
- [ ] suite 5/5 + parity após cada fase; AIPerf re-medido a cada passo

## Armadilhas (não repetir)
- Não subir MNS antes das Fases 2+3 → OOM ruidoso ou corrupção silenciosa.
- Não baixar GMU < 0.9 → KV cache não fecha nem o mínimo.
- Compile abortado → SEMPRE `sudo docker restart vllm_qwen38` antes de relançar (zumbis).
- Benchmark MNS=1 não estima concorrência (mede fila).
