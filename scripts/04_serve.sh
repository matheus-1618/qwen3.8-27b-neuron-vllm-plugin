#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [CONTAINER] Fases 3-4: sobe o vllm serve do Qwen3.8-27B na trn2.3xlarge (TP=4).
# Primeiro boot compila (potencialmente horas na 3xl/12vCPU; NEFF cache torna
# restarts rápidos). Padrão de flags: gemma4 PublicVLLM launch_serve_public.sh.
#
# Env: MODEL, TP, MAX_LEN, PORT, MNS (max-num-seqs), SEG (max-num-batched-tokens),
#      GREEDY (1 = on-device all_greedy p/ smoke determinístico; 0 = sampling geral),
#      SERVED_NAME, EXTRA_ARGS
# Uso: bash 04_serve.sh                      # TP=4 MAX_LEN=512 MNS=1 (primeiro boot)
#      MAX_LEN=4096 SEG=4096 bash 04_serve.sh
set -euo pipefail

MODEL="${MODEL:-/root/models/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-qwen38}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-512}"
SEG="${SEG:-$MAX_LEN}"
# P0 do ROADMAP: buckets múltiplos de prefill. CSV, ex: BUCKETS=512,1024,2048,4096
# O runtime escolhe o menor bucket que serve o prompt -> TTFT baixo em prompt curto.
# Default = um bucket só (comportamento original). Custo: 1 NEFF por bucket no 1º boot.
BUCKETS="${BUCKETS:-$SEG}"
PORT="${PORT:-8000}"
MNS="${MNS:-1}"
GREEDY="${GREEDY:-1}"
# SEGURANÇA: bind em LOOPBACK por padrão. A instância tem IP público e a API não
# tem autenticação — nunca exponha em 0.0.0.0. Acesso pretendido:
#   - de dentro do container (network host → localhost funciona)
#   - do laptop via um túnel SSH que conecta em 127.0.0.1
# Só mude com HOST=... se souber exatamente o que está fazendo.
HOST="${HOST:-127.0.0.1}"
# Fração da HBM que o vLLM pode usar (pesos + KV cache). O budget de KV é
#   KV = HBM_total × GMU − pesos
# e o que sobra (HBM × (1−GMU)) atende DMA rings do runtime, buffers de estado do
# DeltaNet (dimensionados por max_batch!) e scratch.
# MEDIDO na trn2.3xlarge (24GB/core, 27B BF16, TP=4):
#   GMU=0.9 + MNS=1  -> ESTÁVEL (>1h de uso, incluindo benchmarks)
#   GMU=0.9 + MNS=4  -> OOM de device em runtime ("TDRV:dmem_alloc_internal
#                       Failed to allocate DEVICE memory ret=-12"), derruba o
#                       servidor no meio de requests. O estado do DeltaNet é
#                       dimensionado por max_batch, então MNS multiplica memória.
#   GMU=0.8 + MNS=1  -> nem sobe: "KV cache budget below minimum" (KV=0.00 GiB),
#                       o uso não-KV (pesos + constantes + estado + scratch) já
#                       passa de 19.2GB, então 0.8 não deixa nada pro KV.
# Com o cap KV padrão, GMU=0.9/MNS=1 é o ponto conservador comprovado.
# Para 12K/MNS4, dimensione explicitamente KV_CAP; o valor 0.05 foi validado
# para 4×12.288 tokens e libera scratch sem mudar a precisão BF16.
GMU="${GMU:-0.9}"
# Cap do orçamento KV como fração do budget HBM já escalado por GMU.
# Default do plugin = 0.30. Para Qwen híbrido, KV muito acima do necessário
# reduz a HBM disponível aos buffers/scratch do DeltaNet em MNS>1.
# KV_CAP=0.05 limita KV a ~1.08 GiB (~70K tokens BF16/rank), ainda acima dos
# 49.152 tokens de pior caso e libera ~5.4 GiB frente ao default.
KV_CAP="${KV_CAP:-0.30}"
RUN_TAG="${RUN_TAG:-}"
python3 - "$GMU" "$KV_CAP" <<'PY'
import sys
for name, raw in zip(("GMU", "KV_CAP"), sys.argv[1:]):
    value = float(raw)
    if not 0 < value <= 1:
        raise SystemExit(f"{name} deve estar em (0,1], recebido {raw}")
PY
LOG_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"
LOG="/root/serve_qwen38_len${MAX_LEN}_tp${TP}${LOG_SUFFIX}.log"

# Patches de timeout p/ compile longo (lição do qwen3.5: default 30min é curto).
PYSITE="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')"
sed -i 's|^default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT|default_pg_timeout: timedelta = timedelta(hours=6)|' \
  "$PYSITE/torch/distributed/constants.py" 2>/dev/null || true
grep -rl 'timedelta(seconds=1800)' "$PYSITE/vllm_neuron/" 2>/dev/null | \
  xargs -r sed -i 's|timedelta(seconds=1800)|timedelta(hours=6)|g' || true
find "$PYSITE/torch/distributed" "$PYSITE/vllm_neuron" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/neff_cache}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-21600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-21600}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-21600000}"
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION="$KV_CAP"

if [ "$GREEDY" = "1" ]; then
  ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}],\"on_device_sampling_config\":{\"all_greedy\":true}}}"
else
  ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}]}}"
fi

# Fase 4: prefill segmentado p/ contexto > 4096. KV_SEG=4096 compila grafos
# (bucket, segmento) que atendem sobre o cache dos segmentos anteriores.
# Requisito do kernel: seqlen_q == kv_segment_size, então os BUCKETS devem
# ser iguais ao KV_SEG. Uso: MAX_LEN=8192 SEG=4096 BUCKETS=4096 KV_SEG=4096
if [ -n "${KV_SEG:-}" ]; then
  ADD="$(echo "$ADD" | sed "s/\"num_batched_tokens_buckets\"/\"kv_segment_size_buckets\":[${KV_SEG}],\"num_batched_tokens_buckets\"/")"
fi

echo "[serve] MODEL=$MODEL TP=$TP MAX_LEN=$MAX_LEN SEG=$SEG BUCKETS=[$BUCKETS] MNS=$MNS GMU=$GMU KV_CAP=$KV_CAP GREEDY=$GREEDY BIND=$HOST:$PORT"
echo "[serve] log: $LOG"
# Cada RUN_TAG representa um experimento; não misture falhas antigas no parser.
: > "$LOG"

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f multiproc_executor 2>/dev/null || true
sleep 5

# Tool calling: Qwen3.8 emite tool calls em XML (<tool_call><function=...>) →
# parser qwen3_xml (hermes espera JSON e quebra, visto na prática). Reasoning
# parser qwen3 separa o <think>...</think> em `reasoning`, deixando `content` limpo.
nohup vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "$SEG" \
  --gpu-memory-utilization "$GMU" \
  --additional-config "$ADD" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --port "$PORT" --host "$HOST" \
  ${EXTRA_ARGS:-} \
  >> "$LOG" 2>&1 < /dev/null &
disown

echo "[serve] aguardando READY (primeiro boot compila; acompanhe: tail -f $LOG)"
for i in $(seq 1 720); do   # até 6h
  if curl -sf --max-time 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "[serve] READY em http://localhost:$PORT (modelo: $SERVED_NAME)"
    exit 0
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "[serve] ERRO: processo morreu. Últimas linhas do log:"; tail -40 "$LOG"; exit 1
  fi
  sleep 30
done
echo "[serve] TIMEOUT esperando READY"; tail -40 "$LOG"; exit 1
