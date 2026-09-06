#!/usr/bin/env bash
# [INSTÂNCIA] Fase 0: pull do DLC público vLLM-Neuron 0.21 e sobe o container.
# trn2.3xlarge = 1 device (/dev/neuron0). Idempotente (recria container se existir).
# Uso: bash 01_container.sh
set -euo pipefail

IMG="${IMG:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04}"
NAME="${NAME:-vllm_qwen38}"

echo "=== [1/3] Pull DLC público (grande, ~10+ GB — só na primeira vez) ==="
sudo docker pull "$IMG"

echo "=== [2/3] (Re)cria container ==="
sudo docker rm -f "$NAME" 2>/dev/null || true
mkdir -p ~/models ~/neff_cache ~/hf_cache
sudo docker run -d --name "$NAME" \
  --device /dev/neuron0 \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --ipc=host \
  --network host \
  `# network host: a API fica em 127.0.0.1 do HOST (ver HOST= no 04_serve.sh).` \
  `# NÃO publicar portas (-p): acesso só via túnel SSH-over-SSM.` \
  -v "$HOME/models:/root/models" \
  -v "$HOME/hf_cache:/root/hf_cache" \
  -v "$HOME/neff_cache:/root/neff_cache" \
  -v "$HOME/qwen38-27b-trn2:/workspace/qwen38-27b-trn2" \
  -v "$HOME/<internal reference repo>:/workspace/<internal reference repo>" \
  -e HF_HOME=/root/hf_cache \
  -e VLLM_CACHE_ROOT=/root/neff_cache \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  "$IMG" sleep infinity

echo "=== [3/3] Verificação ==="
sudo docker ps --filter "name=$NAME"
sudo docker exec "$NAME" bash -lc 'python3 -c "import vllm, vllm_neuron; print(\"vllm\", vllm.__version__)" && neuron-ls'
echo "CONTAINER OK — nome: $NAME"
