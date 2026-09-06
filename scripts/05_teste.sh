#!/usr/bin/env bash
# Fase 4: testes do endpoint (smoke + tool calling) e chat interativo.
# Roda de qualquer lugar que alcance o endpoint (container, instância, ou laptop
# com port-forward: scripts/ssh.sh -L 8000:localhost:8000 -N).
#
# Uso: bash 05_teste.sh              # suite de smoke + tool calling
#      bash 05_teste.sh chat         # chat interativo (com tools de exemplo)
#      BASE=http://localhost:8000 bash 05_teste.sh chat
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
MODEL="${SERVED_NAME:-qwen38}"
MODE="${1:-suite}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
# O python roda de um ARQUIVO (não heredoc) pra deixar o stdin livre — o modo
# `chat` depende de input() interativo.
exec python3 "$HERE/test/teste_endpoint.py" "$BASE" "$MODEL" "$MODE"
