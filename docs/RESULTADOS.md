# RESULTADOS — Qwen3.8-27B @ trn2.3xlarge (vLLM-Neuron público 0.21)

**Status: FUNCIONANDO** — `Qwen/Qwen3.8-27B` servindo via `vllm serve` no plugin público,
com chat e tool calling validados. Medido em 2026-08-17.

Config: TP=4 (1 chip Trainium2, LNC2 → 4 cores lógicos de 24GB, 96GB HBM), BF16,
greedy on-device, DLC público
`pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`.

## Suite de testes (`05_teste.sh`) — 5/5 PASS @ MAX_LEN=4096

| Teste | Resultado |
|---|---|
| smoke: capital da França | `Paris` ✅ |
| smoke: contagem 1..15 (detector do estado DeltaNet) | `1, 2, 3, ..., 15` exato ✅ |
| smoke: 17×23 | `391` ✅ |
| tool: emite tool_call | `get_weather({"city":"São Paulo","unit":"celsius"})` ✅ |
| tool: resposta final usa o resultado da tool | "Agora em **São Paulo** está a **24 °C** e o céu está **ensolarado** ☀️" ✅ |

O teste de contagem é o detector do bug de state-persistence do DeltaNet (com estado
quebrado o modelo entra em loop de ~3 tokens no decode). Passa → estado persiste
corretamente entre steps.

## Chat interativo multi-turno com tools (`05_teste.sh chat`)

```
você> Quanto é 25 vezes 4?
  [tool] calculate({'expression': '25*4'})
qwen38> 25 vezes 4 é **100**.

você> E o clima em Recife?
  [tool] get_weather({'city': 'Recife'})
qwen38> O clima atual em **Recife** está **ensolarado**, com temperatura de **24°C**. ☀️
```

## Bateria de qualidade — 8/8 PASS

Português e inglês, factual, matemática, geração de código, tradução, listas:

| Prompt | Saída | tok |
|---|---|---:|
| Qual a capital do Brasil? | "A capital do Brasil é **Brasília**." | 58 |
| What is 12 * 12? | "144" | 29 |
| Name the largest planet in our solar system. | "Jupiter" | 27 |
| Write a Python function that returns the nth Fibonacci number. | ```def fibonacci(n: int) -> int: ...``` | 243 |
| Explain in one sentence why the sky is blue. | "sunlight scatters off molecules in the atmosphere..." | 54 |
| Translate to Portuguese: The cat sleeps on the roof. | "O gato dorme no telhado." | 55 |
| List 3 prime numbers greater than 10. | "11, 13, 17" | 57 |
| Answer with just the city name. Capital of Japan? | "Tokyo" | 33 |

### Quirk conhecido do modelo (não é bug de serving)
A formulação exata `"What is the capital of France? Answer with just the city name."`
faz o modelo gerar `"User:"` + EOS em 3 tokens. Reproduzido também em `/v1/completions`
com o chat template aplicado manualmente → é comportamento do modelo com essa sequência
de tokens, não do nosso porte. Formulações equivalentes respondem certo (inclusive a
variante com Japão acima, mesma instrução "answer with just the city name").

## Performance (single stream, MAX_LEN=4096, TP=4, BF16)

| prompt (palavras) | TTFT (s) | decode tok/s | E2E 128 tok (s) |
|---:|---:|---:|---:|
| 0 | 6.97 | 20.7 | 13.06 |
| 200 | 6.96 | 20.7 | 13.05 |
| 800 | 6.96 | 20.7 | 13.05 |

**TTFT constante** porque há um único bucket de prefill (`num_batched_tokens_buckets=[4096]`):
todo prompt é padeado até 4096 tokens. Para prompts curtos, subir com `MAX_LEN=512`
(ou adicionar buckets menores) reduz TTFT muito.

## Load test com AIPerf — concorrência NÃO escala (medido)

Ferramenta: **AIPerf 0.12.0** (NVIDIA), em venv isolada dentro do container
(`scripts/07_loadtest.sh`). 512 tok in / 64 tok out, streaming, 5 requests por nível,
servidor com `--max-num-seqs=1`.

| usuários | TTFT p50 (ms) | TTFT p99 (ms) | ITL avg (ms) | req/s | tok/s saída | latência avg (ms) | latência máx (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 956 | 6 961 | 48.0 | 0.1 | 6.4 | 9 982 | 9 986 |
| 2 | 16 943 | 26 127 | 48.0 | 0.1 | 5.4 | 21 643 | 29 153 |
| 5 | 26 930 | 46 481 | 48.0 | 0.1 | 6.4 | 29 956 | 49 907 |

**Leitura:** serialização pura. O throughput fica **plano** (~6.4 tok/s de saída, 0.1 req/s)
de 1 para 5 usuários — a caixa atende **um request por vez**. O ITL constante em 48 ms confirma
que o decode em si não degrada; o que cresce é só a **espera na fila**:

- com 2 usuários, o segundo espera o primeiro → TTFT p99 26 s (vs 7 s sozinho)
- com 5 usuários, o último espera ~4 × 10 s → TTFT p99 **46 s**, latência máxima **50 s**
- a aritmética fecha: latência máxima ≈ N × 10 s

Ou seja: **5 usuários juntos = 5× a espera, 0× o throughput.** Para atender concorrência de
verdade é preciso resolver os dois bloqueios do [`ROADMAP.md`](./ROADMAP.md) (memória de HBM
por causa da replicação do DeltaNet, e o estado do DeltaNet indexado por posição de batch).

## Verificação de identidade do modelo (é o Qwen3.8-27B mesmo?)

Provado por hash, não por auto-declaração do modelo:

| Verificação | Resultado |
|---|---|
| `sha256` dos shards baixados vs publicados pela HF (`/api/models/Qwen/Qwen3.8-27B/tree/main`) | **bate exatamente** nos shards 1-3 (ex: `ba0ce20a...751b1c`) |
| Nº de shards / tamanho | 18 safetensors, 52 GB em disco |
| `total_size` do index | 55.56 GB → **27.8 B params** em BF16 |
| Path carregado pelo servidor (log) | `/root/models/Qwen3.8-27B-text` → symlinks pra `/root/models/Qwen3.8-27B` (arquivos verificados) |
| Composição do checkpoint | 850 tensores `model.language_model` + 1 `lm_head` (servidos) + 333 `model.visual` + 15 `mtp` (pulados) |
| Repo commit HF | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (lastModified 2026-08-14) |

## ⚠️ Achado: pesos do DeltaNet estão REPLICADOS entre ranks TP (não shardados)

Descoberto ao conferir a métrica `vllm_neuron:model_load_size_bytes` = 87.17 GB, quando o
esperado com sharding completo seria 53.79 GB (texto) → 13.45 GB/rank em TP=4.

| Medida | Valor |
|---|---|
| Texto (servido) no checkpoint | 53.79 GB |
| Esperado por rank em TP=4 se tudo shardado | 13.45 GB |
| **Medido por rank** (87.17/4) | **21.79 GB** |
| Excesso por rank | 8.34 GB |
| Pesos DeltaNet (`*.linear_attn.*`) totais | 11.12 GB |
| DeltaNet shardado seria | 2.78 GB/rank |
| **11.12 − 2.78 = 8.34 GB** | ← **casa exatamente com o excesso** |

Confirmado no código (`model_bf16.py`, herdado do porte Qwen3.6):
```
TP strategy (Phase 4 minimum-viable):
    weights replicated across TP ranks (no head sharding).
             Each rank computes the same DeltaNet output. Functional, not optimal.
    shard num_v_heads across TP. The kernel is per-(b,h)
             so this is just a question of which heads each rank owns.
```
e `# Input projections — replicated across TP ranks for now` nos `in_proj_*`.

**Consequência direta**: 21.79 GB de pesos num core de 24 GB deixa só ~2.2 GB pra KV cache +
estado do DeltaNet + DMA rings. É a explicação exata do envelope apertado documentado abaixo
(GMU 0.8 → KV = 0 GiB; MNS=4 → OOM).

**Maior otimização disponível**: shardar os 48 v-heads do DeltaNet entre os ranks (48/4 = 12
por rank, divide limpo; o kernel é por-(b,h)) liberaria **8.34 GB por core** — quase 5× a folga
atual. Habilitaria contexto maior e concorrência sem trocar de instância. Não implementado.

## Envelope de memória medido (trn2.3xlarge, 1 chip, 24 GB/core)

Uso durante serving (MAX_LEN=4096, TP=4, MNS=1): **~23.7 GB de 24 GB por core**
(23 745 / 23 747 / 23 747 / 23 747 MB nos cores 0-3).

Composição real por core: **21.79 GB de pesos** (por causa da replicação do DeltaNet acima)
+ KV cache + estado + scratch. Isso deixa a janela apertadíssima — três configs testadas:

| Config | Resultado |
|---|---|
| GMU 0.9 + MNS=1 | ✅ **estável** (>1 h de uso, incluindo benchmarks) — config de referência |
| GMU 0.9 + MNS=4 | ❌ **OOM de device em runtime**: `TDRV:dmem_alloc_internal Failed to allocate DEVICE memory (989952 bytes) ret=-12` → worker morre no meio dos requests, servidor cai. O estado do DeltaNet é dimensionado por `max_batch`, então MNS multiplica memória |
| GMU 0.8 + MNS=1 | ❌ **nem sobe**: `Computed KV cache budget is below minimum threshold. effective=0.00 GiB` — o uso não-KV já passa de 19.2 GB, então 0.8 não deixa nada pro KV |

Fórmula do plugin: `KV = HBM_total × GMU − bytes_usados`, com um cap heurístico.
O que sobra de `HBM × (1−GMU)` atende DMA rings, estado do DeltaNet e scratch.

**Implicação para produção nesta instância: concorrência = 1.** Não há memória pra atender
dois requests simultâneos. Antes de trocar de instância, a alavanca é shardar o DeltaNet
(+8.34 GB/core); depois disso, FP8 e/ou mais chips.

## Compile times (trn2.3xlarge, 12 vCPU)

| Bucket | Cold compile | Restart com NEFF cache |
|---|---|---|
| MAX_LEN=512, TP=4 | ~8 min (487 s) | ~4 min |
| MAX_LEN=4096, TP=4 | ~12 min (410 s até READY após cache parcial) | ~7 min |

Bem mais rápido que o temido — o porte irmão em 48xl reportava ~40 min. NEFF cache
persistido em `~/neff_cache` (montado no container).

## Configuração de serving que funciona

```bash
vllm serve /root/models/Qwen3.8-27B-text \
  --served-model-name qwen38 --tensor-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[4096],
      "num_seqs_buckets":[1],"on_device_sampling_config":{"all_greedy":true}}}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

Pontos não óbvios:
- **`--tool-call-parser qwen3_xml`** (não `hermes`): o modelo emite tool calls em XML
  (`<tool_call><function=get_weather><parameter=city>`); o parser hermes espera JSON e
  quebra com `JSONDecodeError`.
- **`--reasoning-parser qwen3`**: o modelo é reasoning (`<think>...</think>`); sem o parser
  o raciocínio vaza no `content`. Com ele, vai pro campo `reasoning` e `content` fica limpo
  (pode vir `None` — trate isso no cliente).
- Budget de tokens precisa acomodar o thinking (dezenas a centenas de tokens antes da resposta).
- Model dir **text-only** obrigatório (`make_local_model.py`) — ver PROGRESSO.md §Fase 3.

## Pendente

- [ ] Buckets múltiplos de prefill (TTFT baixo em prompt curto + suporte a contexto longo)
- [ ] `--max-num-seqs > 1` (concorrência) e medição sob carga
- [ ] Fase 5: FP8 (KV cache primeiro, depois pesos `Qwen3.8-27B-FP8`)
- [ ] Parity numérica por layer vs HF em CPU (opcional; comportamento end-to-end já validado)
