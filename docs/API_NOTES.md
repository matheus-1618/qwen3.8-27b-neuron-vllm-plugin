# API_NOTES — Plugin público vLLM-Neuron 0.21.0.1.0.0 (Fase 1 ✅)

Catálogo da API pública e diffs vs a beta v5 (base do porte qwen3.6).
Fontes: source extraído do DLC em `refs/vllm_neuron/` (local), diff gemma4 beta vs público,
runner (`vllm/worker/neuron_model_runner.py`).

## Conclusão executiva

**A API de modelo da beta v5 ≅ API pública.** O diff inteiro do gemma4 model.py
beta→público são 15 linhas (toggle de kernel v2 opcional). Todos os primitivos que o
qwen3_6 usa existem no plugin público com os mesmos nomes. O que muda é a **integração**:
- Beta: PYTHONPATH + sitecustomize.py + register.py (force-replace de stub no vllm)
- Público: pacote instalado em `vllm_neuron/model/<nome>` + entrada no `registry.py`
  (nosso `scripts/03_install_plugin.sh`, padrão gemma4)

→ Porte do qwen3_6 = copiar pacote, renomear módulo, aplicar o **fix de state
persistence** (que a cópia local do 3.6 NÃO tem), e instalar via registry.

## Layout do plugin no DLC (verificado)

```
/opt/conda/lib/python3.13/site-packages/vllm_neuron/
  backend.py envs.py metrics.py
  accuracy/ compile/ functional/ fx_passes/ model/ nki/ nn/ overrides/ parallel/ utils/ vllm/
  model/
    interfaces.py     # Protocols de visão/mrope (não precisamos — text-only)
    kv_cache.py       # KVSpec / LayerSpec (dataclasses: name, num_kv_heads, head_size, dtype, sliding_window_size, chunk_size)
    neuron_config.py  # NeuronConfig (480 linhas)
    registry.py       # get_models() -> [(arch_name, cls)] — PATCHÁVEL (âncora `models = [`)
    llama3/ gpt_oss/ qwen3_vl/ synthetic/
  nki/nki_hop.py      # wrap_nki ✅ (usado pelo kernel DeltaNet) + can_run_kernel
  nn/                 # sampler.py, embedding.py (VocabDimShardedEmbedding), gqa.py, cpl.py, rpl.py
  functional/         # namespace NF
```

## Contrato modelo ↔ runner (neuron_model_runner.py ~1167-1230, 7783-7830)

```python
# 1. Instanciação (SOB torch.device("meta")!):
model = model_cls.from_configs(hf_config, neuron_config)
# 2. Pesos:
model.load_weights(model_path, device, download_dir)
model = model.to(device)
# 3. KV cache:
spec = model.get_kv_spec()            # -> KVSpec(layers=[LayerSpec...])
model.bind_kv_cache(kv_caches)        # dict[str, list[Tensor]]
# 4. Forward:
model(input_ids, positions, inputs_embeds=None, attn_metadata=..., sampling_positions=...,
      sampling_params=..., spec_decode_metadata=None, logit_mask=None, rank=...)
```
O qwen3_6 beta implementa exatamente isso (assinaturas idênticas ao gemma4 público,
conferido lado a lado). `get_weight_mappings()` é detalhe interno do load_weights do
próprio pacote (via `SafetensorsCheckpoint`), não parte do contrato do runner.

## registry.py (formato confirmado)

```python
def get_models() -> list[tuple[str, type]]:
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM), ...
    ]
```
Nossa entrada: `("Qwen3_5ForConditionalGeneration", Qwen38ForCausalLM)` — nome de
arquitetura HF do Qwen3.8 → nossa factory. `03_install_plugin.sh` cobre.

## NF.* — primitivos usados vs disponíveis

| Usado pelo qwen3_6 (beta) | No público (`functional/__init__.py`) |
|---|---|
| `NF.qkv_proj` | ✅ `attention/qkv.py` |
| `NF.o_proj` | ✅ `attention/o_proj.py` |
| `NF.flash_attention` | ✅ `attention/attention_cte.py` |
| `NF.attention_decode` | ✅ `attention/attention_decode.py` |
| `NF.mlp` | ✅ `mlp.py` |
| — (gemma4 usa) | `NF.segmented_attention` (attention_segmented_cte.py) — disponível p/ prefix caching futuro |

NKI: `vllm_neuron.nki.nki_hop.wrap_nki` ✅ existe no público (mesmo caminho da beta).
O kernel `deltanet_fused.py` do qwen3_6 é NKI puro (independe da API do plugin).

## ⚠️ Fix obrigatório: state persistence do DeltaNet

A cópia local do qwen3_6 (`model_bf16.py` ~linha 907) ainda usa `torch.zeros` nos
buffers `recurrent_state_buffer`/`conv_state_buffer` → **constant-folded pelo compilador
= DeltaNet stateless = loop de 3 tokens**. O README do qwen3.5-4b documenta o fix final
(o 4B do repo usa a variante kv-cache-backed que explode compile time em contexto longo;
não usar):

```python
eps = 1e-30   # = 0 em bf16, mas impede constant folding
torch.full((...), eps, dtype=..., device="cpu")   # em vez de torch.zeros
```
Aplicar nos DOIS buffers ao portar. A técnica evita que buffers all-zero sejam
constant-folded e deve ser validada com teste de estado entre passos de decode.

## Flags de serve que funcionam no público (do gemma4 launch_serve_public.sh)

```bash
vllm serve $MODEL --served-model-name X --tensor-parallel-size TP \
  --max-model-len LEN --max-num-seqs MNS --max-num-batched-tokens SEG \
  [--kv-cache-dtype fp8_e4m3] [--enable-prefix-caching] \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[B],"num_seqs_buckets":[M],"on_device_sampling_config":{"all_greedy":true}}}'
# Envs úteis: VLLM_CACHE_ROOT (NEFF cache), NEURON_SKIP_EFA_AFFINITY=1,
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS/VLLM_ENGINE_ITERATION_TIMEOUT_S/VLLM_RPC_TIMEOUT (compile longo)
```
Obs: `on_device_sampling_config: all_greedy` conflita com sampling variado do chat —
avaliar remover no serving final de chat/tools (sampler on-device suporta top-k/p/temp
segundo o README do repo).

## Riscos em aberto (verificar na Fase 3)

1. **transformers do DLC conhece `model_type: qwen3_5`?** Se não parsear o config nested
   (text_config), fazer `make_local_model.py` nosso (padrão gemma4: gera dir local
   text-only com config achatado + tokenizer). O qwen3_6 config.py já parseia
   `text_config` por conta própria — o risco é só o vllm core (resolução de
   arquitetura/tokenizer).
2. **vllm 0.21 core reconhece a arch `Qwen3_5ForConditionalGeneration`?** Beta usava
   register.py pra force-replace o stub do vllm. Gemma4 público não precisou. Testar;
   se precisar, portar o register.py como parte do pacote/instalação.
3. **TP=4 memória**: ~27GB de pesos/core lógico? NÃO — TP=4 divide os 54GB → 13.5GB/core
   + KV + estado DeltaNet. 24GB/core deve fechar (3.6 mediu 15.34GiB/core em TP=8 no
   48xl; TP=4 tem menos sharding de replicação de KV heads). Medir com neuron-top.
4. **Compile na 3xl (12 vCPU)**: walrus é CPU-bound; 40min do 48xl (192 vCPU) pode virar
   horas. Mitigação: MAX_LEN=256/512 pra iteração; paciência no primeiro boot; NEFF
   cache persistido em ~/neff_cache.
5. **SP (sequence parallel) exige T > world_size e T % world_size == 0** no prefill
   (assert no forward) — prompts mínimos nos testes.

## Decisão de porte (Fase 2)

- Base: `qwen3_6/` completo (config, factory, model_bf16, weight_loaders, nki_kernels)
- Renomes: módulo `qwen3_6` → `qwen38`; classe factory exportada `Qwen38ForCausalLM`
  (mapeada da arch `Qwen3_5ForConditionalGeneration` no registry)
- Aplicar fix eps 1e-30 nos state buffers
- Descartar: register.py, sitecustomize.py, _serve_main.py (substituídos pelo registry
  patch do install), serve.sh da beta (substituído pelo nosso 04_serve.sh)
- Manter testes CPU-only adaptados
