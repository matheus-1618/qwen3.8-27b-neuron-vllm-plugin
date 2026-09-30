# Qwen3.8-27B on AWS Trainium2: custom port for the public vLLM-Neuron plugin

Experimental port of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(hybrid GatedDeltaNet + GQA, 27.8B params, BF16) to a single-chip **trn2.3xlarge**
using the public [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
plugin (`0.21.0.1.0.0`, Neuron SDK 2.31). OpenAI-compatible endpoint with streaming,
reasoning parsing and tool calling.

The plugin natively supports only GPT-OSS, Llama3 and Qwen3-VL. Hybrid
linear-attention models like this one require a custom model package. This repo
is that package plus the replication scripts.

## When to use this repository

Use it to serve Qwen3.8-27B on one Trainium2 chip for functional evaluation, long-context
capacity tests (up to 12K tokens) and development of the hybrid DeltaNet path on Neuron.
It is not a throughput-optimized deployment: long prefills are serialized and decode
attention for `head_dim=256` is not yet paged.

## Validated single-chip results (TP=4, BF16, 96 GiB HBM)

| Metric | Result |
|---|---:|
| Max model length | 12,288 tokens |
| Runtime KV capacity | 69,952 tokens (5.69 times request length) |
| Synthetic long request | 9,062 input tokens, HTTP 200 |
| Four admitted long requests | 4/4 HTTP 200; server remained ready |
| Peak HBM at C4 | 68.887 GiB/chip; 17.222 GiB/core |
| Functional endpoint suite | 5/5 |

C4 wall time was approximately four times C1 because segmented prefills were serialized.
This is a capacity and stability result, not linear throughput scaling.

Configuration, lessons learned, known limits and next steps are consolidated in
[`LEARNINGS.md`](LEARNINGS.md). Sanitized raw JSON is in [`results/`](results/).

## Layout

```
serving_pkg/qwen38/   the model package (installs into vllm_neuron/model/qwen38)
scripts/              00 to 07: host setup, container, model download, plugin
                      install, serve, tests, perf, load test (all idempotent)
test/                 CPU parity tests (sharding, per-seq state) + endpoint suite
results/              sanitized raw results of the validated 12K configuration
LEARNINGS.md          lessons, validated configuration, limits and roadmap
```

## Quickstart (trn2.3xlarge, Amazon Linux 2023)

```bash
bash scripts/00_setup_host.sh          # Neuron driver + docker
bash scripts/01_container.sh           # public DLC container
bash scripts/02_download_model.sh      # 52GB from HF (or your own mirror)
# inside the container:
bash scripts/03_install_plugin.sh && python3 scripts/make_local_model.py
MODEL=/root/models/Qwen3.8-27B-text MAX_LEN=4096 SEG=4096 \
  BUCKETS=512,1024,2048,4096 MNS=1 bash scripts/04_serve.sh
```

For the 12K/MNS4 configuration use the exact command in [`LEARNINGS.md`](LEARNINGS.md).

Known issue: in the 4K configuration, do not serve with a single `BUCKETS=4096` bucket.
Recompiles of that prefill graph can miscompile. Multi-bucket configs are unaffected.

## Compiled artifacts

Compiled NEFFs are not stored in this repository. They are rebuilt from this code on first
serve and cached under `VLLM_CACHE_ROOT`. Reuse a cache only with the same code revision,
DLC, flags and shapes.

## Security note

The vLLM API has no auth. Bind to loopback (default in `04_serve.sh`) and access
through an SSH or SSM tunnel. Never expose the port on a public IP.

## License

Apache-2.0 (same as vllm-neuron). This is an independent experimental port, not
an official AWS or Alibaba artifact. See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md).
