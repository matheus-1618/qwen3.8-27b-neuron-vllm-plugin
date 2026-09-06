#!/usr/bin/env bash
# [INSTÂNCIA] Sync do NEFF cache com S3 — reuso de compiles entre instâncias
# efêmeras (spot/capacity block). NEFFs são keyed por hash do HLO + versão do
# compilador: mesmo DLC + mesma config = cache hit, compile de ~25min vira ~0.
#
# Uso (na instância, via ssh.sh):
#   bash neff_s3.sh push   # fim de sessão: sobe ~/neff_cache pro S3
#   bash neff_s3.sh pull   # bootstrap: baixa o cache antes do primeiro serve
#
# Credencial: role da instância (trn2-qwen-ssm-role) tem policy escopada só
# neste bucket. Bucket privado (public access block total).
set -euo pipefail

BUCKET="${NEFF_BUCKET:-YOUR_S3_BUCKET}"
PREFIX="${NEFF_PREFIX:-neff_cache}"
DIR="${NEFF_DIR:-$HOME/neff_cache}"

case "${1:-}" in
  push)
    [ -d "$DIR" ] || { echo "[neff_s3] $DIR não existe, nada a subir"; exit 0; }
    echo "[neff_s3] push $DIR -> s3://$BUCKET/$PREFIX ($(du -sh "$DIR" | cut -f1))"
    aws s3 sync "$DIR" "s3://$BUCKET/$PREFIX" --only-show-errors
    echo "[neff_s3] push OK"
    ;;
  pull)
    mkdir -p "$DIR"
    echo "[neff_s3] pull s3://$BUCKET/$PREFIX -> $DIR"
    aws s3 sync "s3://$BUCKET/$PREFIX" "$DIR" --only-show-errors
    echo "[neff_s3] pull OK ($(du -sh "$DIR" | cut -f1))"
    ;;
  *)
    echo "uso: $0 push|pull" >&2; exit 1
    ;;
esac
