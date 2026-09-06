#!/usr/bin/env bash
# [CONTAINER] Fase 2: instala o pacote qwen38 no plugin público vLLM-Neuron.
# Copia serving_pkg/qwen38 -> vllm_neuron/model/qwen38 e registra a arquitetura
# Qwen3_5ForConditionalGeneration em vllm_neuron/model/registry.py.
# Idempotente; backup do registry.py na primeira execução.
# Padrão: <internal reference port: gemma4-31b>
# Uso (dentro do container): bash 03_install_plugin.sh
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # raiz do qwen38-27b-trn2

if [ ! -f "$HERE/serving_pkg/qwen38/__init__.py" ]; then
  echo "ERRO: serving_pkg/qwen38 ainda não existe (Fase 2 do PLANO.md)." >&2
  exit 1
fi

MODEL_DIR="$(python3 - <<'PY'
import os, sys
for base in sys.path:
    cand = os.path.join(base, "vllm_neuron", "model")
    if os.path.isdir(cand):
        print(cand); break
PY
)"
if [ -z "${MODEL_DIR:-}" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERRO: vllm_neuron/model não encontrado no sys.path. Está dentro do DLC público?" >&2
  exit 1
fi
echo "[install] vllm_neuron model dir: $MODEL_DIR"

# 1. Deploy do pacote
DST="$MODEL_DIR/qwen38"
mkdir -p "$DST"
cp -rf "$HERE/serving_pkg/qwen38/." "$DST/"
find "$DST" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "[install] pacote qwen38 -> $DST"

# 2. Registro no registry.py (idempotente, com backup)
REG="$MODEL_DIR/registry.py"
python3 - "$REG" <<'PY'
import sys
reg = sys.argv[1]
src = open(reg).read()
changed = False
if "from .qwen38 import Qwen38ForCausalLM" not in src:
    lines = src.splitlines()
    idx = max(i for i, l in enumerate(lines) if l.startswith("from ."))
    lines.insert(idx + 1, "from .qwen38 import Qwen38ForCausalLM")
    src = "\n".join(lines) + ("\n" if not src.endswith("\n") else "")
    changed = True
if '("Qwen3_5ForConditionalGeneration"' not in src:
    anchor = "    models = ["
    entry = (
        '        ("Qwen3_5ForConditionalGeneration", Qwen38ForCausalLM),\n'
        '        # Arch text-only usada pelo model dir achatado (make_local_model.py).\n'
        '        # ATENÇÃO: sobrescreve o Qwen3-Next nativo do vllm core neste container\n'
        '        # (dedicado ao Qwen3.8-27B) — vllm core trata Qwen3_5ForConditionalGeneration\n'
        '        # como multimodal, então servimos via arch text-only da mesma família.\n'
        '        ("Qwen3NextForCausalLM", Qwen38ForCausalLM),\n'
    )
    src = src.replace(anchor, anchor + "\n" + entry, 1)
    changed = True
if changed:
    import shutil, os
    bak = reg + ".bak_qwen38"
    if not os.path.exists(bak):
        shutil.copy2(reg, bak)
    open(reg, "w").write(src)
    print(f"[install] registry patched -> {reg} (backup {bak})")
else:
    print(f"[install] registry já tem Qwen38: {reg}")
PY

# 3. Sanity
python3 - <<'PY'
from vllm_neuron.model.registry import get_models
names = [n for n, _ in get_models()]
assert "Qwen3_5ForConditionalGeneration" in names, names
print("[install] OK — registrado: Qwen3_5ForConditionalGeneration")
PY
echo "[install] done."
