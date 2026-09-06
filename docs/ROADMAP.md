# ROADMAP — Qwen3.8-27B em Trainium2

Estado atual medido, limitações reais e caminhos de melhoria em ordem de custo/benefício.
Números medidos estão em [`RESULTADOS.md`](./RESULTADOS.md). Onde é estimativa, está marcado.

## 1. Onde estamos (medido em 2026-08-17)

| Métrica | Valor | Como foi medido |
|---|---|---|
| Correção | suite 5/5, bateria 8/8 | `05_teste.sh`, prompts diversos PT/EN |
| TTFT | 6.96 s (constante 0-800 palavras de prompt) | `06_perf.sh` (`vllm bench serve`) |
| Decode | 20.7 tok/s · TPOT 47.9 ms | idem |
| E2E (512 in / 64 out) | 9.98 s | idem |
| Concorrência suportada | **1** | MNS=4 → OOM de device |
| Contexto | 4096 | config atual |
| HBM | 23.7 / 24 GB por core | `neuron-monitor` |
| Pesos por rank | 21.79 GB (esperado 13.45) | métrica `model_load_size_bytes` |

Benchmark: `bash 06_perf.sh` dentro do container (env `IN`, `OUT`, `CONC`, `PROMPTS`).
Usa o harness oficial `vllm bench serve` (TTFT/TPOT/ITL/E2EL + throughput), salva JSON por
ponto em `results/` e imprime tabela consolidada.

## 2. Suporta múltiplas sessões?

Precisa separar dois sentidos:

**Múltiplas conversas ao longo do tempo: SIM.** O servidor é stateless — o histórico vive no
cliente e vai inteiro em cada request. Vários usuários podem usar em sequência sem interferência.

**Múltiplos requests simultâneos: NÃO — medido com AIPerf.** Com 5 usuários o throughput não
sobe (fica plano em ~6.4 tok/s / 0.1 req/s) e a espera multiplica: TTFT p99 vai de 7 s para
**46 s** e a latência máxima para **50 s** (tabela completa em `RESULTADOS.md`). Há **dois**
bloqueios independentes, não só um:

### Bloqueio A — memória (medido)
`MNS=4` → `TDRV:dmem_alloc_internal Failed to allocate DEVICE memory (ret=-12)`: o servidor
morre em runtime. Causa: 21.79 GB de pesos por core de 24 GB deixam ~2.2 GB para KV cache +
estado do DeltaNet + DMA rings, e o estado do DeltaNet é dimensionado por `max_batch`.

### Bloqueio B — correção do estado do DeltaNet (encontrado por leitura de código)
Mais grave, porque **falha silenciosamente** em vez de crashar. O estado recorrente é lido e
escrito por **posição no batch**, sem nenhum mapeamento para os IDs de sequência do vLLM:

```python
conv_state      = self.conv_state_buffer[:batch_size]        # decode
recurrent_state = self.recurrent_state_buffer[:batch_size]
...
self.recurrent_state_buffer.copy_(new_state.to(self.dtype))  # escreve o buffer inteiro
```

Problemas com `MNS > 1`:
1. **Slots não são estáveis.** No continuous batching do vLLM a composição do batch muda entre
   steps (sequências terminam, novas entram, a ordem muda). O slot `i` no step `t` pode ser
   outra sequência no step `t+1` → o estado recorrente é aplicado à sequência errada.
   Resultado: saída corrompida, sem erro.
2. **O `copy_` escreve o buffer todo.** Buffer é `[max_batch, H, k, v]`, `new_state` é
   `[batch_size, H, k, v]`. Com `batch_size=1` e `max_batch=4`, o `copy_` faz *broadcast* e
   grava o estado de uma sequência em **todos** os 4 slots. Com `batch_size=2` e `max_batch=4`,
   levanta erro de shape.

Isso vem do porte original (padrão "side-channel buffer" da PR #152, escrito para batch=1) e
foi herdado. **Correção obrigatória antes de qualquer concorrência**: indexar o estado por slot
real (via `slot_mapping`/`index_put_`, como o KV cache faz) em vez de fatia posicional.

## 3. O throughput está adequado?

Depende do caso de uso. Com TTFT 7 s e 20.7 tok/s:

| Caso de uso | Veredito |
|---|---|
| Chat exploratório, 1 pessoa | **OK.** Resposta curta em ~10 s, aceitável com streaming ligado |
| Prototipagem / validação de porte | **OK.** É o propósito atual desta caixa |
| Coding agent interativo | **Inadequado.** Um diff de 500 tokens leva ~25 s + 7 s de TTFT. Agents fazem várias chamadas por tarefa → minutos por iteração. E tool calls em paralelo não rodam (concorrência 1) |
| API multiusuário / produção | **Inadequado.** Concorrência 1 e o bloqueio B acima |
| Batch offline (throughput-bound) | **Marginal.** 20.7 tok/s numa caixa de $2.23/h dá custo por token alto sem batching |

Para referência de ordem de grandeza: o TTFT de 7 s é **constante** entre prompt de 0 e 800
palavras — porque há um único bucket de prefill (4096) e todo prompt é padeado até lá. Ou seja,
hoje pagamos o preço de 4096 tokens mesmo para um "oi".

## 4. Como aumentar — em ordem de custo/benefício

### P0 — Buckets múltiplos de prefill
**Ganho: TTFT de ~7 s para ~1-2 s em prompts curtos (estimado).** Esforço: baixo (config).
Hoje `num_batched_tokens_buckets=[4096]`. Passar `[512, 1024, 2048, 4096]` faz o runtime
escolher o menor bucket que serve o prompt. Custo: um NEFF por bucket (compile mais longo na
primeira vez; cache depois). Risco: baixo. **Não testado ainda.**

### P0 — Prefix caching (`--enable-prefix-caching`)
**Ganho: elimina o reprefill do contexto repetido.** Esforço: baixo (flag; suportado pelo
plugin e usado no exemplo gemma4 público). Para coding agent é o maior ganho isolado, porque
o system prompt + contexto do repo se repetem a cada turno — só o delta é prefillado.
Risco: baixo, mas consome KV cache (que hoje é escasso — depende do P1). **Não testado.**

### P1 — Shardar os pesos do DeltaNet entre ranks TP
**Ganho: +8.34 GB por core (medido).** Esforço: médio (mudar loaders + forward das 48 camadas
GDN). Hoje os pesos do DeltaNet são replicados em todos os 4 ranks — a docstring do porte
original admite: *"weights replicated across TP ranks (no head sharding). Functional, not
optimal."* Os 48 v-heads dividem limpo por TP=4 (12/rank) e o kernel é por-(b,h), então é
questão de escolher quais heads cada rank possui. Libera quase 5× a folga atual de HBM →
habilita contexto maior **e** é pré-requisito prático pra concorrência.

### P1 — Estado do DeltaNet indexado por sequência
**Ganho: destrava concorrência corretamente (bloqueio B).** Esforço: médio-alto. Trocar a
fatia posicional por escrita/leitura indexada pelo slot da sequência. Sem isso, `MNS>1`
produz saída corrompida silenciosamente — pior que o OOM, que ao menos é ruidoso.
**Fazer junto com o P1 acima**, e validar com um teste que roda N conversas simultâneas e
compara cada uma com a resposta obtida sequencialmente.

### P2 — FP8
**Ganho: ~2× no orçamento de memória.** Esforço: médio-alto, risco alto no stack público.
Duas alavancas independentes:
- **KV cache FP8** (`--kv-cache-dtype fp8_e4m3`): mais barato de tentar, não toca nos pesos.
  Usado no exemplo gemma4 público em contextos ≥16k.
- **Pesos FP8** (`Qwen/Qwen3.8-27B-FP8`, e4m3 block `[128,128]`): halving dos 21.79 GB/rank.
  Exige suporte a fp8 block-wise no nosso pacote — o plugin suporta FP8 no framework, mas
  num modelo custom é trabalho novo.
Fora de escopo por decisão atual.

### P2 — Mais chips (trocar de instância)
**Ganho: o maior de todos, mas custa dinheiro.** A trn2.3xlarge tem **1 chip**; a
trn2.48xlarge tem **16**. Com TP=8/16/32 os pesos se espalham, libera HBM para KV e batch,
e o prefill é shardado (TTFT cai proporcionalmente). É a configuração dos benchmarks de
referência do gemma4 que empatam/batem H100 em contexto longo. Para produção real é o
caminho; para validar o porte, a 3xl basta.

### P3 — Kernels NKI de decode
**Ganho: 1.5-2× no decode (estimado pelo porte irmão).** Esforço: alto. O porte do Qwen3.5-4B
tentou um kernel NKI de decode-attention para `head_dim=256` e a **v1 ficou 20% mais lenta**
que o caminho eager que o `neuronx-cc` auto-fusiona (63.7 vs 79.6 tok/s) — está documentado
em `<internal reference port: qwen3.5-4b>`. Só vale depois de esgotar
P0/P1, e com A/B rigoroso.


## 4.5 Esforço estimado dos dois P1 (e prior art investigado)

### Prior art: NÃO existe no repo de referência interno para nenhum dos dois
- **Ambos os portes** (`qwen3.5-4b` e `qwen3.6-27b`) têm a **mesma docstring** admitindo
  `weights replicated across TP ranks (no head sharding). Functional, not optimal.`
- **Ambos** têm `MAX_NUM_SEQS=1` como default no `serve.sh` — são projetos batch=1 por design.
- A abordagem "estado no KV cache" do 4B (`_write_recurrent_state_to_cache`) resolve
  **persistência**, não concorrência: escreve com `torch.arange(N)` a partir do índice 0, ou
  seja, **também é posicional**. E foi **abandonada** pelo próprio autor: *"generated graphs
  with 4M+ instructions, blowing up compile time at MAX_LEN=4096 to over 90 minutes per HLO"*.

### Prior art fora do repo: vLLM core tem o design, o plugin Neuron não tem a infra
- vLLM core traz `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (**GDN = Gated Delta
  Net**, exatamente nossa arquitetura, usada pelo `Qwen3NextForCausalLM`) e
  `MambaSpec(KVCacheSpec)` em `v1/kv_cache_interface.py` — a abstração correta de estado
  indexado por sequência. **Excelente referência de design.**
- Mas no `vllm_neuron` os **únicos** arquivos que mencionam `mamba`/`linear_attention` são
  **os nossos** (`model/qwen38/*`). O `model/kv_cache.py` do plugin só tem
  `LayerSpec(name, num_kv_heads, head_size, dtype, sliding_window_size, chunk_size)` —
  **nenhum conceito de estado recorrente**. Ou seja: não há como "só usar a infra do vLLM";
  o gerenciador de KV do plugin não sabe o que é estado de atenção linear.

### P1a — Shardar pesos do DeltaNet: **~1-2 dias**, risco baixo-médio
Bem delimitado, e o padrão existe **no próprio arquivo**:
- As camadas GQA já shardam heads (`num_attention_heads_per_rank = // world_size`, linha ~310)
- `sharding_weight_loader_with_padding` + `set_weight_loader` já são usados
- O all-reduce pós-projeção já existe (`self.tp_group.all_reduce(...)`, linhas 774 e 1409)
- **Divide limpo**: 48 v-heads / TP=4 = 12; 16 k-heads / 4 = 4
- **Kernel não muda** — é per-(b,h), cada rank processa os heads dele
- Muda: `in_proj_*` (column-parallel), `out_proj` (row-parallel + all_reduce),
  `conv1d_weight` (por canal), `norm_weight`/`A_log`/`dt_bias` (por head), e os reshapes do
  forward passam a usar contagens por rank (2 métodos: prefill e decode)
- **Validação fácil**: mesmo prompt, greedy, saída deve bater com a atual
- Ganho: **+8.34 GB/core** (medido). Vale por si só, mesmo sem concorrência — libera contexto.

### P1b — Estado indexado por sequência: **~1-2 semanas**, risco alto
Sem prior art aplicável; nós seríamos donos da correção:
- Precisa mapear slot de estado ↔ sequência do scheduler, e cobrir o **ciclo de vida**:
  início de sequência (estado zerado), término/eviction (slot liberado e reusado → resetar),
  preempção/recompute
- A escrita tem de ser **scatter no slot certo**, não `copy_` no buffer inteiro — e tem de
  sobreviver à functionalization (aprendemos que `.data.copy_` é descartado)
- **Risco de compile-time é precedentado neste mesmo código**: a tentativa do 4B com
  `index_put_` gerou grafos de 4M+ instruções e 90+ min de compile. Nossa versão (scatter em
  ~N slots) é menor que a deles (flat sobre 524k floats), mas são 48 camadas × 2 estados
- **Questão de design em aberto**: interação com **prefix caching**. Blocos de KV se reusam
  entre requests; estado recorrente de atenção linear **não** — o estado do prefixo teria de
  ser cacheado ou recomputado. Isso é decisão de arquitetura, não só código

### Recomendação de ordem
**P1a primeiro**, sempre. Além de ser barato e ganho garantido, ele é o que torna o P1b
**testável**: hoje `MNS>1` dá OOM antes de dar pra observar o bug de estado.

## 5. Sequência recomendada

1. **P0 (buckets + prefix caching)** — dias, só config, derruba TTFT. Medir com `06_perf.sh`.
2. **P1 (shardar DeltaNet)** — libera 8.34 GB/core; re-medir HBM e contexto máximo.
3. **P1 (estado por sequência)** — destrava `MNS>1`; validar com teste de concorrência.
4. Reavaliar throughput. Se ainda insuficiente para o caso de uso → **P2 (mais chips ou FP8)**.
5. **P3 (NKI)** só com evidência de que o eager é o gargalo.

## 6. O que NÃO fazer

- Subir `MNS` sem resolver os bloqueios A e B: OOM (ruidoso) ou corrupção de estado (silencioso).
- Mexer em `GMU` para ganhar KV: 0.8 já não sobe (`KV = 0.00 GiB`). A janela é estreita porque
  os pesos ocupam 21.79 GB — resolver a replicação primeiro.
- Confiar em benchmark com `MNS=1` para estimar concorrência: mede fila, não paralelismo.
