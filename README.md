# Qwen3.8-27B on AWS Trainium2 — custom port for the public vLLM-Neuron plugin

**Experimental** port of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(hybrid GatedDeltaNet + GQA, 27.8B params, BF16) to a single-chip **trn2.3xlarge**
using the public [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
plugin (`0.21.0.1.0.0`, Neuron SDK 2.31). OpenAI-compatible endpoint with streaming,
reasoning parsing and tool calling.

The plugin natively supports only GPT-OSS, Llama3 and Qwen3-VL — hybrid
linear-attention models like this one require a custom model package. This repo
is that package, plus the full replication pipeline and the optimization journey.

## Measured results (TP=4, BF16, 1× Trainium2 chip, 96GB HBM)

| Metric | Initial port | Optimized | Gain |
|---|---:|---:|---:|
| Decode (single stream) | 20.7 tok/s | 31.3 tok/s | 1.5× |
| TTFT (short prompt) | 6.96 s | 0.92 s | 7.5× |
| Max context | 4,096 | 8,192 (needle-test verified) | 2× |
| Concurrent requests | 1 (serialized) | 4 (correct per-sequence state) | 4× |
| Aggregate throughput @4 | 22 tok/s | 43.6 tok/s | 2.2× |
| TTFT p99 @ 4 users | 45 s | 7.0 s | 6.5× |

Achieved through four phases: multi-bucket prefill → DeltaNet TP head-sharding →
per-sequence recurrent state (continuous batching) → segmented prefill (8K context
via an NKI kernel extended with initial-state input).

## Highlights / war stories (see `docs/`)

1. **Recurrent state vs. functionalization** — `.data.copy_()` mutations are
   silently dropped by the capture backend; direct `copy_()`/`index_put_` are
   captured as input mutations. Zero-init buffers get constant-folded.
2. **Sequence-parallel is a contract** — a custom layer that skips the entry
   all_gather silently processes 1/ws of the sequence and *works by accident*
   until prompts exceed bucket/ws tokens.
3. **Padding is not harmless in stateful models** — pad-token outputs are
   discarded by the sampler but NOT by the DeltaNet recurrent state.
4. **Per-sequence state without touching the runner** — block-anchor matching
   + rotating staging slots, fully in-graph, self-healing after batch condense.
5. **The compiler is part of your system** — a documented non-deterministic
   miscompile (same HLO, good NEFF in the morning, bad NEFF on recompile).
   Cache your validated NEFFs: it is reproducibility, not just speed.

## Layout

```
serving_pkg/qwen38/   the model package (installs into vllm_neuron/model/qwen38)
scripts/              00→07: host setup, container, model download, plugin
                      install, serve, tests, perf, load test (all idempotent)
test/                 CPU parity tests (sharding, per-seq state) + endpoint suite
docs/                 port notes, results, roadmap, lessons learned (pt-BR)
```

## Quickstart (trn2.3xlarge, Amazon Linux 2023)

```bash
bash scripts/00_setup_host.sh          # Neuron driver + docker
bash scripts/01_container.sh           # public DLC container
bash scripts/02_download_model.sh      # 52GB from HF (or your S3 mirror)
# inside the container:
bash scripts/03_install_plugin.sh && python3 scripts/make_local_model.py
MODEL=/root/models/Qwen3.8-27B-text MAX_LEN=4096 SEG=4096 \
  BUCKETS=512,1024,2048,4096 MNS=4 bash scripts/04_serve.sh
```

⚠️ Known issue: do NOT serve with a single `BUCKETS=4096` bucket — recompiles of
that prefill graph can miscompile (deterministic garbage EOS on specific prompts).
Multi-bucket configs are unaffected.

## Security note

The vLLM API has no auth. Bind to loopback (default in `04_serve.sh`) and access
through an SSH tunnel. Never expose the port on a public IP.

## License

Apache-2.0 (same as vllm-neuron). This is an independent experimental port, not
an official AWS or Alibaba artifact.
