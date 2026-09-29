#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [CONTAINER] Fase 4: benchmark de performance com `vllm bench serve` (harness oficial do vLLM).
#
# Mede TTFT, TPOT, ITL, E2E e throughput com dataset sintético, varrendo concorrência.
# Salva um JSON por ponto em RESULT_DIR e imprime uma tabela consolidada no fim.
#
# Uso (dentro do container, com o servidor JÁ no ar):
#   bash 06_perf.sh                                  # sweep default: conc 1,2,4
#   IN=1024 OUT=128 CONC="1 4 8" bash 06_perf.sh     # customizado
#   PROMPTS=20 bash 06_perf.sh                       # mais amostras por ponto
#
# Env:
#   BASE        URL do servidor            (default http://localhost:8000)
#   MODEL       nome servido               (default qwen38)
#   IN / OUT    tokens de entrada/saída    (default 512 / 128)
#   CONC        níveis de concorrência     (default "1 2 4")
#   PROMPTS     requests por nível         (default 3 × concorrência, mín 4)
#   RESULT_DIR  saída dos JSONs            (default /workspace/qwen38-27b-trn2/results)
#
# ATENÇÃO: o servidor precisa ter `--max-num-seqs` >= a concorrência testada, senão
# os requests apenas enfileiram e o resultado mede fila, não paralelismo. O default
# do 04_serve.sh é MNS=1. Para o envelope 12K/MNS4 validado, use a configuração
# completa de docs/LONG_CONTEXT_TRN2_3XL.md; não aumente MNS sem redimensionar KV.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
SERVED="${SERVED_NAME:-qwen38}"
# O harness precisa do tokenizer local (path), separado do nome servido na API.
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.8-27B-text}"
IN="${IN:-512}"
OUT="${OUT:-128}"
CONC="${CONC:-1 2 4}"
RESULT_DIR="${RESULT_DIR:-/workspace/qwen38-27b-trn2/results}"

command -v vllm >/dev/null || { echo "ERRO: rode dentro do container do DLC (vllm não está no PATH)" >&2; exit 1; }
curl -sf --max-time 5 "$BASE/v1/models" >/dev/null || {
  echo "ERRO: servidor não responde em $BASE/v1/models — suba o 04_serve.sh primeiro" >&2; exit 1; }

# Avisa se max_num_seqs < maior concorrência pedida.
MNS_SRV="$(curl -s "$BASE/v1/models" >/dev/null 2>&1; grep -oE '\-\-max-num-seqs [0-9]+' /root/serve_qwen38_*.log 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo "?")"
MAXC="$(echo "$CONC" | tr ' ' '\n' | sort -n | tail -1)"
if [ "$MNS_SRV" != "?" ] && [ "$MNS_SRV" -lt "$MAXC" ] 2>/dev/null; then
  echo "AVISO: servidor com --max-num-seqs=$MNS_SRV < concorrência máxima $MAXC."
  echo "       Os pontos acima de $MNS_SRV medem ENFILEIRAMENTO, não paralelismo."
  echo "       Redimensione MNS/KV conforme docs/LONG_CONTEXT_TRN2_3XL.md."
  echo
fi

mkdir -p "$RESULT_DIR"
echo "[perf] BASE=$BASE served=$SERVED tokenizer=$MODEL_PATH input=$IN output=$OUT conc=[$CONC]"
echo "[perf] resultados -> $RESULT_DIR"
echo

for c in $CONC; do
  n="${PROMPTS:-$(( c * 3 < 4 ? 4 : c * 3 ))}"
  fname="perf_in${IN}_out${OUT}_c${c}.json"
  echo "=== concorrência $c ($n requests) ==="
  rm -f "$RESULT_DIR/$fname"
  runlog="$RESULT_DIR/${fname%.json}.stdout.log"
  set +e
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url "$BASE" \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED" \
    --dataset-name random \
    --random-input-len "$IN" \
    --random-output-len "$OUT" \
    --num-prompts "$n" \
    --max-concurrency "$c" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$RESULT_DIR" --result-filename "$fname" \
    2>&1 | tee "$runlog" | grep -E "Successful|Benchmark duration|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Mean E2EL|Median E2EL"
  bench_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$bench_rc" -ne 0 || ! -s "$RESULT_DIR/$fname" ]]; then
    echo "ERRO: benchmark c=$c falhou (rc=$bench_rc); veja $runlog" >&2
    [[ "$bench_rc" -ne 0 ]] && exit "$bench_rc"
    exit 1
  fi
  echo
done

echo "======================= RESUMO ======================="
python3 - "$RESULT_DIR" "$IN" "$OUT" <<'PY'
import glob, json, os, sys
rd, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(rd, f"perf_in{IN}_out{OUT}_c*.json"))):
    d = json.load(open(f))
    if isinstance(d, list):
        d = d[-1]
    rows.append((
        d.get("max_concurrency") or d.get("num_prompts"),
        d.get("mean_ttft_ms", 0) / 1000,
        d.get("p99_ttft_ms", 0) / 1000,
        d.get("mean_tpot_ms", 0),
        d.get("output_throughput", 0),
        d.get("mean_e2el_ms", 0) / 1000,
        d.get("completed", 0),
    ))
if not rows:
    print("(sem resultados)"); raise SystemExit
rows.sort(key=lambda r: (r[0] or 0))
print(f"{'conc':>4} | {'TTFT méd(s)':>11} | {'TTFT p99(s)':>11} | {'TPOT(ms)':>8} | "
      f"{'saída tok/s':>11} | {'E2E méd(s)':>10} | {'ok':>3}")
print("-" * 76)
for c, ttft, p99, tpot, thr, e2e, ok in rows:
    print(f"{c:>4} | {ttft:>11.2f} | {p99:>11.2f} | {tpot:>8.1f} | {thr:>11.1f} | {e2e:>10.2f} | {ok:>3}")
print(f"\ninput={IN} tok, output={OUT} tok (--ignore-eos). JSONs completos em {rd}")
PY
