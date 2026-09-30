# Qwen3.8-27B on AWS Trainium2 — custom port for the public vLLM-Neuron plugin

**Experimental** port of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(hybrid GatedDeltaNet + GQA, 27.8B params, BF16) to a single-chip **trn2.3xlarge**
using the public [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
plugin (`0.21.0.1.0.0`, Neuron SDK 2.31). OpenAI-compatible endpoint with streaming,
reasoning parsing and tool calling.

The plugin natively supports only GPT-OSS, Llama3 and Qwen3-VL — hybrid
linear-attention models like this one require a custom model package. This repo
is that package, plus the full replication pipeline and the optimization journey.

## Validated single-chip results (TP=4, BF16, 96 GiB HBM)

| Metric | Result |
|---|---:|
| Max model length | 12,288 tokens |
| Runtime KV capacity | 69,952 tokens (5.69× request length) |
| Synthetic long request | 9,062 input tokens, HTTP 200 |
| Four admitted long requests | 4/4 HTTP 200; server remained ready |
| Peak HBM at C4 | 68.887 GiB/chip; 17.222 GiB/core |
| Functional endpoint suite | 5/5 |

C4 wall time was approximately four times C1 because segmented prefills were serialized. This is a capacity/stability result, not linear throughput scaling. Sanitized raw JSON and the exact configuration are in [`docs/LONG_CONTEXT_TRN2_3XL.md`](docs/LONG_CONTEXT_TRN2_3XL.md). Historical 4K optimization experiments remain under `docs/` and should not be mixed with this 12K configuration.

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
  BUCKETS=512,1024,2048,4096 MNS=1 bash scripts/04_serve.sh
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

## Single-chip 12K context update

A BF16/TP4 configuration with `MAX_LEN=12288`, `MNS=4` and a workload-sized KV budget has now been validated on the trn2.3xlarge envelope. Four synthetic 9,062-token requests completed without OOM; peak HBM was 17.222 GiB per logical core. Prefills remain serialized, so this is a capacity/stability result rather than linear C4 scaling.

See [`docs/LONG_CONTEXT_TRN2_3XL.md`](docs/LONG_CONTEXT_TRN2_3XL.md) for configuration, measurements, compilation caveats and the scope of any H100 comparison. See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) for upstream references and attribution.
