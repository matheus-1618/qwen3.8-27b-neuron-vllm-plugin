#!/usr/bin/env bash
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
# Com o default MNS=1, requests concorrentes apenas ENFILEIRAM — o teste mede fila,
# não paralelismo (é justamente o que queremos demonstrar). E NÃO suba MNS nesta
# instância sem antes ler o ROADMAP.md: além de OOM de HBM, o estado do DeltaNet é
# indexado por posição de batch e corrompe silenciosamente com MNS>1.
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
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q aiperf
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
python3 - "$OUTDIR" "$IN" "$OUT" <<'PY'
import glob, json, os, sys
root, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

def find_json(d):
    for pat in ("**/profile_export_aiperf.json", "**/*aiperf*.json", "**/*.json"):
        hits = glob.glob(os.path.join(d, pat), recursive=True)
        hits = [h for h in hits if "input" not in os.path.basename(h).lower()]
        if hits:
            return hits[0]
    return None

def pick(d, *names):
    """Procura métrica por nome em dicts aninhados."""
    for n in names:
        if n in d:
            return d[n]
    return None

rows = []
for art in sorted(glob.glob(os.path.join(root, f"c*_in{IN}_out{OUT}"))):
    c = int(os.path.basename(art).split("_")[0][1:])
    f = find_json(art)
    if not f:
        rows.append((c, None)); continue
    try:
        data = json.load(open(f))
    except Exception:
        rows.append((c, None)); continue
    rows.append((c, data))

if not any(d for _, d in rows):
    print("(sem JSON de resultado — veja a saída de cada nível acima)")
    raise SystemExit

def get(d, metric, stat="avg"):
    m = d.get(metric) if isinstance(d, dict) else None
    if isinstance(m, dict):
        return m.get(stat)
    return None

hdr = (f"{'usuários':>8} | {'TTFT p50(ms)':>12} | {'TTFT p99(ms)':>12} | "
       f"{'ITL avg(ms)':>11} | {'req/s':>7} | {'tok/s saída':>11}")
print(hdr); print("-" * len(hdr))
for c, d in rows:
    if not d:
        print(f"{c:>8} | {'?':>12} | {'?':>12} | {'?':>11} | {'?':>7} | {'?':>11}")
        continue
    recs = d.get("records", d)
    ttft_p50 = get(recs, "time_to_first_token", "p50") or get(recs, "time_to_first_token", "avg")
    ttft_p99 = get(recs, "time_to_first_token", "p99")
    itl = get(recs, "inter_token_latency", "avg")
    rps = get(recs, "request_throughput", "avg")
    ops = get(recs, "output_token_throughput", "avg")
    fmt = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "?"
    print(f"{c:>8} | {fmt(ttft_p50):>12} | {fmt(ttft_p99):>12} | {fmt(itl):>11} | "
          f"{fmt(rps):>7} | {fmt(ops):>11}")
print(f"\ninput={IN} tok, output={OUT} tok, streaming. Artefatos completos em {root}")
PY
