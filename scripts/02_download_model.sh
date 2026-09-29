#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [INSTÂNCIA] Fase 0: baixa Qwen/Qwen3.8-27B (~54GB, público, sem token) pra ~/models.
# Roda no HOST (venv própria) pra poder rodar em paralelo com o docker pull.
# MODEL_ID=Qwen/Qwen3.8-27B-FP8 pra variante FP8 (fase 5).
# Uso: bash 02_download_model.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
DEST="$HOME/models/$(basename "$MODEL_ID")"

# FAST PATH: modelo espelhado no S3 (mesma região = 10x mais rápido que HF;
# ~5 min vs ~40 min). Sobe com: aws s3 sync ~/models/<nome> s3://$BUCKET/models/<nome>
S3_BUCKET="${MODEL_S3_BUCKET:-YOUR_S3_BUCKET}"
S3_PREFIX="models/$(basename "$MODEL_ID")"
if aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/config.json" >/dev/null 2>&1; then
  echo "=== FAST PATH: modelo no S3 (s3://$S3_BUCKET/$S3_PREFIX) ==="
  mkdir -p "$DEST"
  aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX" "$DEST" --only-show-errors
  echo "=== Verificação ==="
  du -sh "$DEST"
  python3 - "$DEST" <<'PY'
import json, sys, os
cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))
assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"], cfg["architectures"]
tc = cfg["text_config"]
print("OK (S3):", cfg["architectures"][0], "| layers:", tc["num_hidden_layers"], "| vocab:", tc["vocab_size"])
PY
  exit 0
fi
echo "[download] S3 vazio ou sem acesso — seguindo pela HF (lento)"

echo "=== [1/2] venv + huggingface_hub ==="
python3 -m venv ~/.hfvenv 2>/dev/null || true
~/.hfvenv/bin/pip install -q "huggingface_hub[hf_transfer]==0.36.0"

echo "=== [2/2] Download $MODEL_ID -> $DEST ==="
mkdir -p "$DEST"
HF_HUB_ENABLE_HF_TRANSFER=1 ~/.hfvenv/bin/hf download "$MODEL_ID" \
  --local-dir "$DEST" \
  --exclude "*.pth" 2>&1 | tail -3

echo "=== Verificação ==="
ls -la "$DEST" | head
du -sh "$DEST"
python3 - "$DEST" <<'PY'
import json, sys, os
cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))
assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"], cfg["architectures"]
tc = cfg["text_config"]
print("OK:", cfg["architectures"][0], "| layers:", tc["num_hidden_layers"], "| vocab:", tc["vocab_size"])
PY
echo "DOWNLOAD OK"
