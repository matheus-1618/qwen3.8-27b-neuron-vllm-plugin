#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [CONTAINER] Load test / stress com AIPerf (NVIDIA) — simula N usuários concorrentes.
#
# Diferente do 06_perf.sh (vllm bench serve, mais simples), aqui usamos o AIPerf,
# que simula usuários de verdade e reporta percentis (p50/p90/p99) de TTFT, ITL,
# TPOT e E2E, além de throughput e taxa de erro.
#
# AIPerf roda numa venv ISOLADA em /root/aiperf_venv (python 3.13 do container),
# separada dos pacotes do vLLM — instalada por este script se faltar.
#
# Uso (dentro do container, servidor JÁ no ar):
#   bash 07_loadtest.sh                       # sweep 1,2,5 usuários
#   CONC="1 2 5 10" bash 07_loadtest.sh       # sweep customizado
#   IN=1024 OUT=128 REQS=20 bash 07_loadtest.sh
#
# Env:
#   BASE/URL   endpoint            (default localhost:8000)
#   SERVED     nome do modelo      (default qwen38)
#   TOKENIZER  path do tokenizer   (default /root/models/Qwen3.8-27B-text)
#   IN/OUT     tokens in/out       (default 512 / 64)
#   CONC       níveis (usuários)   (default "1 2 5")
#   REQS       requests por nível  (default 4 × concorrência)
#   OUTDIR     artefatos           (default /workspace/qwen38-27b-trn2/results/loadtest)
#
# ⚠️ LEIA ANTES DE INTERPRETAR OS NÚMEROS:
# O servidor precisa de `--max-num-seqs >= N` para atender N usuários EM PARALELO.
# Com MNS=1, concorrência de clientes mede fila. MNS=4 foi validado somente no
# envelope 12K com KV_CAP=0.05 descrito em docs/LONG_CONTEXT_TRN2_3XL.md;
# outros tamanhos/MNS exigem novo dimensionamento e validação de estado.
set -euo pipefail

URL="${BASE:-${URL:-http://localhost:8000}}"
SERVED="${SERVED:-${SERVED_NAME:-qwen38}}"
TOKENIZER="${TOKENIZER:-/root/models/Qwen3.8-27B-text}"
IN="${IN:-512}"
OUT="${OUT:-64}"
CONC="${CONC:-1 2 5}"
OUTDIR="${OUTDIR:-/workspace/qwen38-27b-trn2/results/loadtest}"
VENV="${VENV:-/root/aiperf_venv}"

# --- AIPerf numa venv isolada (não mexe nos pacotes do vLLM) ---
if [ ! -x "$VENV/bin/aiperf" ]; then
  echo "[load] instalando aiperf em $VENV (venv isolada, python $(python3 -V 2>&1 | cut -d' ' -f2))"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q "aiperf==0.12.0"
fi
AIPERF="$VENV/bin/aiperf"
echo "[load] aiperf $($AIPERF --version 2>&1 | tail -1)"

curl -sf --max-time 5 "$URL/v1/models" >/dev/null || {
  echo "ERRO: servidor não responde em $URL/v1/models — suba o 04_serve.sh primeiro" >&2; exit 1; }

# Avisa se o servidor não consegue paralelizar o que vamos pedir.
MNS_SRV="$(grep -ohE '\-\-max-num-seqs [0-9]+' /root/serve_qwen38_*.log 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo '?')"
MAXC="$(echo "$CONC" | tr ' ' '\n' | sort -n | tail -1)"
echo "[load] servidor --max-num-seqs=$MNS_SRV | concorrência máxima do teste=$MAXC"
if [ "$MNS_SRV" != "?" ] && [ "$MNS_SRV" -lt "$MAXC" ] 2>/dev/null; then
  echo "[load] AVISO: níveis acima de $MNS_SRV medem ENFILEIRAMENTO, não paralelismo."
fi
echo

mkdir -p "$OUTDIR"
for c in $CONC; do
  n="${REQS:-$(( c * 4 ))}"
  art="$OUTDIR/c${c}_in${IN}_out${OUT}"
  echo "=============== $c usuário(s) concorrente(s), $n requests ==============="
  rm -rf "$art"
  "$AIPERF" profile \
    --model "$SERVED" \
    --tokenizer "$TOKENIZER" \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --concurrency "$c" \
    --request-count "$n" \
    --synthetic-input-tokens-mean "$IN" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OUT" \
    --output-tokens-stddev 0 \
    --warmup-request-count 1 \
    --artifact-dir "$art" \
    --ui simple \
    2>&1 | grep -vE "^\s*$" | tail -32
  echo
done

echo "================================ RESUMO ================================"
python3 - "$OUTDIR" "$IN" "$OUT" "$CONC" "${REQS:-}" <<'PY_SUMMARY'
import glob, json, os, sys
root, IN, OUT, conc_raw, reqs_raw = sys.argv[1:]
expected_concurrency = [int(value) for value in conc_raw.split()]

def find_json(directory):
    for pattern in ("**/profile_export_aiperf.json", "**/*aiperf*.json", "**/*.json"):
        hits = [path for path in glob.glob(os.path.join(directory, pattern), recursive=True)
                if "input" not in os.path.basename(path).lower()]
        if hits: return hits[0]
    return None

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from walk(child)
    elif isinstance(value, list):
        for child in value: yield from walk(child)

def metric(data, name, stat="avg"):
    for key, value in walk(data):
        if key == name:
            return value.get(stat) if isinstance(value, dict) else value
    return None

def completed_count(data):
    preferred = ("successful_requests", "completed_requests", "num_completed_requests", "completed", "request_count")
    flat = list(walk(data))
    for wanted in preferred:
        for key, value in flat:
            if key.lower() != wanted: continue
            if isinstance(value, (int, float)): return int(value)
            if isinstance(value, dict) and isinstance(value.get("avg"), (int, float)):
                return int(value["avg"])
    return None

rows=[]; errors=[]
for concurrency in expected_concurrency:
    directory=os.path.join(root, f"c{concurrency}_in{IN}_out{OUT}")
    result=find_json(directory)
    if not result:
        errors.append(f"c={concurrency}: missing result JSON"); continue
    try: data=json.load(open(result))
    except Exception as error:
        errors.append(f"c={concurrency}: invalid JSON {result}: {error}"); continue
    expected=int(reqs_raw) if reqs_raw else concurrency*4
    completed=completed_count(data)
    if completed is None:
        errors.append(f"c={concurrency}: artifact has no completed/successful request count"); continue
    if completed != expected:
        errors.append(f"c={concurrency}: completed {completed}, expected {expected}"); continue
    ttft=metric(data,"time_to_first_token","p50") or metric(data,"time_to_first_token","avg")
    p99=metric(data,"time_to_first_token","p99")
    itl=metric(data,"inter_token_latency","avg")
    rps=metric(data,"request_throughput","avg")
    output_tps=metric(data,"output_token_throughput","avg")
    required=(ttft,rps,output_tps)
    if not all(isinstance(value,(int,float)) for value in required):
        errors.append(f"c={concurrency}: required metrics missing from {result}"); continue
    rows.append((concurrency,ttft,p99,itl,rps,output_tps,completed))

if errors or len(rows) != len(expected_concurrency):
    for error in errors: print("ERROR:",error,file=sys.stderr)
    raise SystemExit(1)

header=(f"{'users':>8} | {'TTFT p50(ms)':>12} | {'TTFT p99(ms)':>12} | "
        f"{'ITL avg(ms)':>11} | {'req/s':>7} | {'output tok/s':>12} | {'ok':>4}")
print(header); print("-"*len(header))
fmt=lambda value: f"{value:.1f}" if isinstance(value,(int,float)) else "n/a"
for c,ttft,p99,itl,rps,output_tps,completed in rows:
    print(f"{c:>8} | {fmt(ttft):>12} | {fmt(p99):>12} | {fmt(itl):>11} | "
          f"{fmt(rps):>7} | {fmt(output_tps):>12} | {completed:>4}")
print(f"\ninput={IN}, output={OUT}, streaming; artifacts: {root}")
PY_SUMMARY
