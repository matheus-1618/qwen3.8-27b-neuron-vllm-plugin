#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create the public vLLM-Neuron container for a single Trainium2 chip.
set -euo pipefail
IMG="${IMG:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04}"
NAME="${NAME:-vllm_qwen38}"
REPO_DIR="${REPO_DIR:-$PWD}"
DEVICE="${DEVICE:-/dev/neuron0}"
MEM="${MEM:-128g}"
CPUSET="${CPUSET:-}"
sudo docker pull "$IMG"
sudo docker rm -f "$NAME" 2>/dev/null || true
mkdir -p "$HOME/models" "$HOME/neff_cache" "$HOME/hf_cache"
args=(
  -d --name "$NAME" --device "$DEVICE" --memory "$MEM"
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --ipc=host --network host
  -v "$HOME/models:/root/models"
  -v "$HOME/hf_cache:/root/hf_cache"
  -v "$HOME/neff_cache:/root/neff_cache"
  -v "$REPO_DIR:/workspace/qwen38-27b-trn2:ro"
  -e HF_HOME=/root/hf_cache
  -e VLLM_CACHE_ROOT=/root/neff_cache
  -e NEURON_SKIP_EFA_AFFINITY=1
)
[[ -n "$CPUSET" ]] && args+=(--cpuset-cpus "$CPUSET")
sudo docker run "${args[@]}" "$IMG" sleep infinity
sudo docker inspect "$NAME" --format 'cpuset={{.HostConfig.CpusetCpus}} mem={{.HostConfig.Memory}} devices={{json .HostConfig.Devices}}'
