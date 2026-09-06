# APRENDIZADOS — Portando Qwen3.8-27B pro Trainium2 (vLLM-Neuron público)

Material bruto da palestra + lições operacionais. Estado final: **estavel-v2**
(2026-09-06) — 4 fases de otimização validadas em 1 dia de sessão.

## O placar (medido, não estimado)

| Métrica | Porte inicial (17/08) | estavel-v2 (06/09) | Ganho |
|---|---:|---:|---:|
| Decode single-stream | 20.7 tok/s | 31.3 tok/s (MODE=chat) | 1.5× |
| TTFT prompt curto | 6.96 s | 0.92 s | 7.5× |
| TTFT 512 tokens | 6.96 s | 1.75 s | 4× |
| Contexto máximo | 4.096 | 8.192 (needle test 5.681 tok) | 2× |
| Requests paralelas | 1 (fila) | 4 (estado correto, 4/4) | 4× |
| Throughput agregado @4 | 22 tok/s (plano) | 43.6 tok/s | 2.2× |
| TTFT p99 @ 4 users | 45 s | 7.0 s | 6.5× |
| Custo/1M tokens | ~$30 | ~$14 (@conc 4) | 2.1× |

Hardware: 1× trn2.3xlarge ($2.24/h), 1 chip Trainium2, 96 GB HBM, TP=4, BF16.

## Lições técnicas (cada uma custou horas de debugging)

### 1. Estado recorrente × functionalization (o fix mais importante do porte)
Buffers de estado do DeltaNet mutados via `.data.copy_()` são DESCARTADOS pela
functionalization do capture backend — o modelo vira stateless e degenera em loop
após 2-3 tokens. `copy_()`/`index_put_` DIRETOS no buffer são capturados como input
mutation. Buffers zero-init também são constant-folded: init com eps 1e-30 (float)
ou valores distintos (int).

### 2. Sequence Parallel é contrato de TODAS as camadas
O residual stream do prefill é sequence-scattered ([T/ws, hidden]). Camada custom
que não faz all_gather na entrada processa 1/ws da sequência — e FUNCIONA POR
ACASO enquanto os tokens reais couberem no pedaço do rank 0 (prompt ≤ bucket/ws).
Bug latente por semanas; explodiu no bucket 512 com prompts de ~300 tokens.

### 3. Padding não é inofensivo em modelos com estado
Saída de token de padding é descartada pelo sampler, mas NÃO pelo estado recorrente
do DeltaNet (que processa a sequência padded inteira). Quando o prefill segmentado
passou a LER o KV cache (fallback PyTorch do segmented_attention), queries de
padding atendendo sobre memória não-inicializada envenenaram o estado → EOS
imediato, dependente do conteúdo. Fix: máscara `pad_valid` zerando a saída de
queries padded na fonte.

### 4. Sharding paga duas vezes
Shardar os 48 v-heads do DeltaNet (12/rank @ TP=4) deu 1.5× no decode (bandwidth
de pesos) E liberou 7.8 GB/core de HBM — que é o que destravou MNS=4 (a mesma
config dava OOM antes). Bônus: o all_gather do sharding CORRIGIU o bug do item 2.

### 5. Estado por sequência sem tocar o runner (design da Fase 3)
Continuous batching move rows (condense) → estado indexado por posição corrompe
silenciosamente. Solução 100% in-graph: anchor = primeiro bloco do KV (estável por
vida da sequência), staging rotativo pra prefills consecutivos, decode resolve
estado por anchor-matching com self-healing pós-condense. Restrições que moldaram
o design: prefill não pode consumir block_table (shape varia → recompile); decode
roda com shapes estáticos [MNS].

### 6. O compilador é parte do sistema (e falha)
`neuronx-cc` produziu NEFF quebrado pro MESMO grafo em recompiles (prefill 4096
bucket único): código da madrugada validado 5/5 às 08:46 falhou ao recompilar à
tarde, até pós-reboot. Multi-bucket imune (2/2 builds bons). Lição: NEFF cache no
S3 não é só velocidade — é REPRODUTIBILIDADE. Grafos validados são artefatos
preciosos; recompilar é risco.

### 7. Fallbacks silenciosos
`NF.segmented_attention` cai em fallback PyTorch sem avisar quando o kernel NKI
não suporta head_dim 256. O fallback funciona, mas com semântica sutilmente
diferente (lê a janela padded inteira). Sempre verificar QUAL caminho compilou.

### 8. Operação de instâncias efêmeras
- Capacity block/spot trocam instance ID → descoberta automática + auto-fix de
  SSM (agente que boota sem credencial entra em backoff longo; tem que reiniciar).
- Compile abortado deixa zumbis que seguram os cores → `docker restart` SEMPRE
  antes de relançar.
- NEFF + modelo no S3 mesma região: redeploy de ~60 min → ~15 min.

## Runbook: REDEPLOY RÁPIDO (instância nova do zero)

```bash
cd sp/            # ou melbourne/
./bootstrap.sh    # driver + docker + DLC + modelo (S3, ~5min) + plugin + NEFFs (S3)
./serve.sh        # MODE=prod default: MNS=4 multi-bucket — SEM COMPILE (cache hit)
./chat.sh suite   # 5/5 esperado
```
Tempo total estimado: **~15-20 min** (vs ~60-90 min sem os caches S3).

Modos do serve.sh:
- `./serve.sh` — produção: 4 paralelas, 43.6 tok/s agregado
- `MODE=chat ./serve.sh` — single-stream máximo (31.3 tok/s)
- `MODE=8k ./serve.sh` — contexto 8192 (prefill segmentado)

⚠️ NUNCA `BUCKETS=4096` sozinho (known-issue item 6).

Artefatos S3 (bucket `YOUR_S3_BUCKET`, sa-east-1, ~$3/mês):
- `neff_cache/` — todos os grafos validados (estavel-v1 e v2)
- `models/Qwen3.8-27B/` — checkpoint 52 GB (hash-verificado vs HF)

Rollback de código: `git checkout estavel-v2 -- qwen38-27b-trn2/serving_pkg/`
depois `./sync.sh` + `03_install_plugin.sh` + serve.

## Próximos passos (opcionais, priorizados)

1. **MNS=8** — HBM sobra; só compile + medir (meta: ~60-70 tok/s agregado)
2. **16K de contexto** — infra de segmentos pronta; custo: decode lê janela inteira
3. **Plugin v0.24 / SDK 2.32** — accuracy debugger + possível fix do miscompile
4. **FP8 per-tensor** — ~1.45× decode (requantização offline, 2-3 dias)
5. **Repro mínimo do miscompile** pro time do Neuron SDK
6. **Publicação** — sanitizar (IDs/IPs/conta), repo limpo, Holmes scan, open-source approval
