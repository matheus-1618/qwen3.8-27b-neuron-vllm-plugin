#!/usr/bin/env python3
"""Diagnóstico de latência: cronometra prompts e mostra quanto do custo é reasoning.

Uso: python3 test/diag_latencia.py [BASE] [MODEL]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38"


def timed(content, max_tokens=1024):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    m = d["choices"][0]["message"]
    reasoning = m.get("reasoning") or ""
    content_out = m.get("content") or ""
    n = d["usage"]["completion_tokens"]
    return dt, n, len(reasoning), content_out, d["choices"][0]["finish_reason"]


print(f"{'prompt':<28} | {'tempo(s)':>8} | {'tok':>5} | {'tok/s':>6} | {'reason(chars)':>13} | finish")
print("-" * 92)
for p in ["hey", "oi", "Qual a capital da França?", "Escreva uma função Python de fatorial"]:
    dt, n, rlen, out, fin = timed(p)
    print(f"{p[:28]:<28} | {dt:>8.1f} | {n:>5} | {n/dt:>6.1f} | {rlen:>13} | {fin}")
    print(f"    -> {out[:100]!r}")
