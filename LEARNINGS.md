# Qwen3.8-27B on Trainium2: lessons, validated configuration and limits

This document consolidates what was learned while porting Qwen3.8-27B to the public vLLM-Neuron plugin on a single trn2.3xlarge. It is the only technical document in this repository; the README covers setup.

## Scope

- Instance: trn2.3xlarge, one Trainium2 chip, 96 GiB HBM, 12 vCPU, 128 GiB RAM.
- Stack: public vLLM-Neuron `0.21.0.1.0.0`, Neuron SDK 2.31, public Neuron DLC.
- Model: text-only derivative of Qwen3.8-27B produced by `scripts/make_local_model.py`.
- Precision: BF16 weights and KV cache. TP=4 (LNC2).
- Status: experimental. Functional and stable in the configuration below; not optimized for throughput.

## Integration with the public plugin

The public plugin supports a fixed list of architectures. This repository installs a model package under `vllm_neuron/model/qwen38` and adds an entry to the plugin registry (`scripts/03_install_plugin.sh`). The model follows the same runner contract as the bundled models: `from_configs`, `load_weights`, `get_kv_spec`, `bind_kv_cache` and `forward`.

The architecture combines 48 GatedDeltaNet layers with 16 full-attention layers. Only the 16 attention layers use the paged KV cache. The DeltaNet layers keep recurrent state and causal-convolution state in model buffers.

## Lessons

### Recurrent state is part of the serving contract

Keeping the KV cache is not enough. DeltaNet recurrent state and convolution state must survive segmented prefill and decode, and must follow the request identity when the batch is condensed.

- `.data.copy_()` mutations were dropped by graph capture. Direct `copy_()` and `index_put_` are captured as input mutations.
- Buffers initialized to zero were constant-folded by the compiler, making the layer stateless. Initializing with a negligible epsilon (`1e-30`, zero in BF16) prevents folding.
- A plain `copy_` into the state buffer broadcast one sequence to every slot. Per-row `index_put_` fixes it.
- The runner reorders rows when requests finish (`input_batch.condense()`). State is located inside the graph by matching the first KV block of the request against the block table, so no runner change is required for identity.

### Sequence parallelism is a contract

A custom layer that skips the entry all-gather processes only 1/world_size of the sequence. It appears to work until prompts exceed bucket/world_size tokens. Every custom layer must honor the same sharding contract as the surrounding layers.

### Padding is not neutral in stateful layers

The sampler discards outputs for pad tokens, but pad tokens still update DeltaNet state. Slot mapping, continuation masks and state writes must exclude padding explicitly.

### KV budget must match the architecture

With the default cap (0.30), the runtime reserved about 6.48 GiB per logical core for 419,744 KV tokens. Since only 16 of 64 layers use KV, this allocation was far larger than needed and reduced memory for DeltaNet state and runtime scratch. `KV_CAP=0.05` allocated about 1.08 GiB per core for 69,952 tokens, above the 49,152 tokens required for four full 12K requests.

Recalculate this value when changing `MAX_LEN`, `MNS`, precision or plugin version.

### The compiler is part of the system

- A single `BUCKETS=4096` prefill graph produced a miscompiled artifact on recompile in the 4K configuration (same HLO, incorrect EOS on specific prompts). Multi-bucket configurations were not affected.
- Parallel graph trace produced `NCC_EVRF059`: the HLO referenced NKI temporary files removed before `neuronx-cc` read them. An isolated cache plus `VLLM_NEURON_DISABLE_PARALLEL_TRACE=1` produced a valid artifact in 1,912 s.
- Keep validated compiled caches together with the code revision, DLC version, flags and shapes. Reuse a cache only when all of these match. Compiled artifacts are not stored in this repository; they are rebuilt from the code.

### Keep configurations separate

Earlier 4K experiments and the current 12K configuration use different graphs and memory budgets. Numbers from one must not be used to describe the other.

## Validated 12K configuration

```bash
MODEL=/root/models/Qwen3.8-27B-text \
VLLM_CACHE_ROOT=/root/neff_cache/long_context_12k_mns4 \
VLLM_NEURON_DISABLE_PARALLEL_TRACE=1 \
MAX_LEN=12288 SEG=4096 BUCKETS=4096 KV_SEG=4096 \
MNS=4 GMU=0.90 KV_CAP=0.05 \
bash scripts/04_serve.sh
```

`KV_CAP` maps to `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`.

| Test | Result |
|---|---:|
| KV capacity | 69,952 tokens (5.69 times the window) |
| One synthetic long request | 9,062 input tokens, HTTP 200, 54 output tokens, 27.81 s |
| Four concurrent long requests | 4/4 HTTP 200, server ready afterwards |
| C4 wall time | 104.83 s |
| Peak HBM at C4 | 68.887 GiB per chip, 17.222 GiB per core |
| Functional endpoint suite | 5/5 |

C4 wall time is about four times C1 because segmented prefills are serialized by the current scheduler. This validates capacity and stability, not concurrency scaling. Sanitized raw results are in [`results/`](results/).

## Known limits

- Long prefills are serialized.
- Attention decode gathers the configured window; the available fused kernel limits `head_dim` to 128 and this model uses 256.
- No prefix caching for DeltaNet state.
- The vLLM endpoint has no authentication. `04_serve.sh` binds to loopback; access through an SSH or SSM tunnel.

## Next steps

1. Paged decode for `head_dim=256` that reads only the needed blocks. Gate: same outputs, lower TPOT, no extra HBM.
2. NKI segmented attention for `head_dim=256` (d-tiling in 128 with online softmax). Gate: numerical parity and lower TTFT at 4K, 8K and 12K.
3. Hybrid prefix cache that stores and restores DeltaNet state at the end of the prefix. Gate: identical output to a full prefill.
4. Non-serialized prefills and a robust parallel trace.

## Comparing with GPUs

Results here do not establish parity with H100. A fair comparison uses the same model revision, precision, prompts, output length, schema constraints, cache policy, warm-up and load generator on both platforms, and reports cost per successful request.
