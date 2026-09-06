# FASE 3 — Design: estado do DeltaNet por sequência (MNS>1)

Anotado em 2026-09-06 após leitura do source do plugin (`refs/vllm_neuron/`).
Objetivo: destravar `MNS=4` com correção — hoje o estado é indexado por posição
no batch e o `copy_` faz broadcast (§7 do CONTEXT.md, itens 1-2).

## Fatos levantados no source (com linha)

1. **Decode batch = rows do `input_batch`** (persistent batch do vLLM v1).
   O runner **condensa**: `self.input_batch.condense()` em
   `neuron_model_runner.py:981` e `:2027` — quando uma request termina, a última
   row é movida pro buraco. **Row NÃO é identidade estável.**
2. **O plugin já resolve esse exato problema** pra outro estado por-row no
   device: `_remap_prev_sampled_by_req_id` (`neuron_model_runner.py:2085+`)
   reordena o tensor de sampled tokens do passo anterior via permutação
   construída de `prev_req_ids_ordered` vs `input_batch.req_ids` atual.
   **É o gabarito do nosso hook.**
3. **Prefill é sempre batch de 1 request** (`_compute_cached_seq_len`:
   "Neuron currently only supports a prefill batch size of 1"). Mas a request
   prefillada pode ocupar QUALQUER row do input_batch (preenche buraco).
4. `attn_metadata[layer_name]` (por grupo de KV cache, `:3872-4045`) dá ao
   DeltaNet dummy: `block_table_tensor` [padded_num_reqs, max_blocks],
   `slot_mapping` [tokens], `block_size`. Block tables por **grupo** —
   as camadas DeltaNet (dummy 1-head) formam grupo próprio.
5. Não existe `MambaSpec` no plugin (só FullAttention/SlidingWindow) → não dá
   pra usar o padrão upstream "estado dentro do KV paginado" sem mexer fundo.

## Design

### A. In-graph (model_bf16.py) — corrige broadcast + identidade no prefill

- **Decode (escrita):** trocar `self.recurrent_state_buffer.copy_(new_state)`
  (broadcast!) por `index_copy_`/`index_put_` com `arange(B)`:
  só as rows vivas são escritas, cada uma no seu slot. Mesmo padrão de input
  mutation do `index_put_` do k_cache (que a functionalization captura).
  Idem `conv_state_buffer`. Leitura `buffer[:B]` continua posicional.
- **Prefill (escrita/leitura):** descobrir a row da request DENTRO do grafo,
  sem tocar o runner:
  ```
  first_block = slot_mapping[0] // block_size          # bloco 0 da request
  hits = (block_table_tensor[:, 0] == first_block)     # [padded_num_reqs]
  row  = hits.float().argmax()                         # índice da row
  ```
  (bloco 0 da request é estável e único no grupo dummy; o table row é copiado
  junto no condense.) Escrever estado com `index_put_((row,), final_state)`;
  no prefill segmentado (Fase 4), LER o estado inicial de `buffer[row]` em vez
  de `buffer[:1]`.
- **Máscara de continuação** (Fase 4) continua igual — `positions[0] > 0`.

### B. Host-side hook (patch no runner via 03_install_plugin) — estado segue a row

Depois de `input_batch.condense()` a permutação de rows precisa ser aplicada
aos buffers de estado (espelho do item 2 acima):

- Guardar `prev_req_ids_ordered`; ao detectar reordenação, construir o index
  de gather e aplicar `buffer = buffer.index_select(0, perm)` (ou index_copy
  in-place) em `recurrent_state_buffer` + `conv_state_buffer` das 48 camadas.
- Implementação: helper no nosso pacote que recebe o modelo e a permutação;
  chamado de um monkeypatch pós-condense (03_install_plugin já patcha registry;
  adicionar patch pontual no runner). 96 index_selects pequenos por
  reordenação — só acontece quando request termina, não por passo.
- Alternativa mais simples e robusta: manter os buffers como estão e fazer o
  hook copiar row-a-row apenas os movimentos que o condense fez (a lista de
  moves é conhecida no host; normalmente 1 move por request finalizada).

### C. Memória (por que agora cabe)

Estado por slot (TP=4, por rank): recurrent [12,128,128] fp32→bf16 + conv
[2560,3] ≈ 0.4 MB × 48 layers ≈ 19 MB/slot. MNS=4 ≈ 76 MB/rank — nada.
O gargalo real era KV cache: Fase 2 liberou 7.8 GB/core (medido 12.56 GiB
usados) → KV pra 4 sequências de 4096 cabe com folga.

### D. Validação

1. Parity CPU: simular 2 sequências intercaladas (prefill A, prefill B,
   decode A, decode B, término A + condense, decode B) vs execução isolada.
2. On-device: `MNS=4` + suite 5/5 por conexão; contagem 1..15 em 4 conexões
   PARALELAS com prompts diferentes (detector de estado cruzado).
3. AIPerf `CONC="1 2 4"` — critério: ITL estável e tok/s agregado escalando
   (hoje: plano em ~22 tok/s de 1→4).

### Riscos

- `argmax` do prefill assume bloco 0 único por request no grupo dummy —
  verificar se prefix caching (se ligado) compartilha blocos no grupo dummy
  (se sim, desligar prefix caching pro grupo dummy ou usar outro anchor).
- O decode graph é compilado com shape fixo [MNS_bucket]; buckets de seqs
  `num_seqs_buckets=[4]` → padding de rows mortas escreve lixo no próprio
  slot (isolado, OK), mas conferir que rows padded têm block_table sentinela
  que não colide no argmax do prefill (PAD_SLOT_ID/-1 → hits=false, OK).
- Custo do all_gather de decode com B=4: medir; deve ser desprezível vs
  leitura de pesos.
