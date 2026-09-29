#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Suite de testes + chat interativo do endpoint Qwen3.8-27B (chamado por scripts/05_teste.sh).

Arquivo separado (não heredoc) porque o modo `chat` precisa do stdin livre pra input().

Uso: python3 test/teste_endpoint.py <BASE_URL> <MODEL> <suite|chat>
"""
import ast, json, operator, sys, urllib.request

BASE, MODEL, MODE = sys.argv[1], sys.argv[2], sys.argv[3]

def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def chat(messages, tools=None, max_tokens=256, temperature=0.0):
    p = {"model": MODEL, "messages": messages,
         "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        p["tools"] = tools
        p["tool_choice"] = "auto"
    return post("/v1/chat/completions", p)["choices"][0]["message"]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtém o clima atual de uma cidade",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Nome da cidade"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Avalia uma expressão matemática",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_arithmetic(expression):
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](visit(node.operand))
        raise ValueError("unsupported arithmetic expression")
    if len(expression) > 128: raise ValueError("expression too long")
    return visit(ast.parse(expression, mode="eval"))

def fake_tool_exec(name, args):
    if name == "get_weather":
        return json.dumps({"city": args.get("city"), "temp_c": 24, "condition": "ensolarado"})
    if name == "calculate":
        try:
            return json.dumps({"result": safe_arithmetic(args["expression"])})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return "{}"

if MODE == "suite":
    ok = 0; fail = 0
    def check(name, cond, detail=""):
        global ok, fail
        s = "PASS" if cond else "FAIL"
        if cond: ok += 1
        else: fail += 1
        print(f"[{s}] {name}" + (f" — {detail}" if detail else ""))

    # 1. smoke factual
    # NOTA: a formulação "What is the capital of France? Answer with just the city name."
    # dispara um caso degenerado pontual do modelo (gera "User:" + EOS em 3 tokens) —
    # reproduzido também em /v1/completions com o chat template aplicado à mão, logo é
    # quirk do modelo, não do serving. Variantes equivalentes respondem certo (bateria
    # de 8 prompts diversos: 8/8). Usamos a formulação simples aqui.
    m = chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=800)
    check("smoke: capital da França", "paris" in (m.get("content") or "").lower(), repr((m.get("content") or ""))[:120])

    # 2. contagem (pega loop degenerado de estado DeltaNet quebrado)
    m = chat([{"role": "user", "content": "Count from 1 to 15, comma-separated, nothing else."}], max_tokens=800)
    got = (m.get("content") or "")
    check("smoke: contagem 1..15 (estado DeltaNet)", all(str(i) in got for i in range(1, 16)), repr(got)[:150])

    # 3. matemática
    # A formulação "17 times 23 = ? Answer with just the number." cai em um
    # EOS prematuro específico (reasoning="We", 2 tokens), enquanto três
    # formulações equivalentes retornam 391. Use uma variante estável para o
    # smoke testar aritmética/estado, não esse quirk lexical do checkpoint.
    m = chat([{"role": "user", "content": "Compute 17 multiplied by 23. Return only the integer."}], max_tokens=800)
    check("smoke: 17*23=391", "391" in (m.get("content") or ""), repr((m.get("content") or ""))[:120])

    # 4. tool calling: o modelo deve emitir tool_call
    m = chat([{"role": "user", "content": "Qual o clima agora em São Paulo em celsius?"}], tools=TOOLS)
    tcs = m.get("tool_calls") or []
    check("tool: emitiu tool_call get_weather",
          bool(tcs) and tcs[0]["function"]["name"] == "get_weather",
          json.dumps(tcs)[:200] if tcs else repr((m.get("content") or ""))[:150])

    # 5. tool calling end-to-end: devolve resultado e espera resposta final coerente
    if tcs:
        args = json.loads(tcs[0]["function"]["arguments"])
        result = fake_tool_exec("get_weather", args)
        m2 = chat([
            {"role": "user", "content": "Qual o clima agora em São Paulo em celsius?"},
            {"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs},
            {"role": "tool", "tool_call_id": tcs[0]["id"], "content": result},
        ], tools=TOOLS)
        final = m2.get("content") or ""
        check("tool: resposta final usa resultado (24)", "24" in final,
              repr(final)[:150] + (" | re-chamou tool" if m2.get("tool_calls") else ""))
    else:
        check("tool: resposta final usa resultado (24)", False, "sem tool_call na etapa anterior")

    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)

elif MODE == "chat":
    import os
    import time

    # THINK=1 (default) mostra o raciocínio do modelo em cinza; THINK=0 só progresso.
    SHOW_THINK = os.environ.get("THINK", "1") != "0"
    # Budget de saída por turno. Modelo é reasoning: o <think> consome desse budget,
    # então para tarefas de código vale um valor alto. MAXTOK=... para ajustar.
    MAX_TOK = int(os.environ.get("MAXTOK", "4096"))

    def chat_stream(messages, tools=None, max_tokens=None, temperature=0.0):
        """Envia com stream=True e imprime os tokens conforme chegam.

        Retorna a mensagem do assistant reconstruída ({content, tool_calls}).
        O modelo é reasoning: o <think> vem no campo `reasoning` do delta. Mostramos
        o raciocínio como pontinhos (pra dar sinal de vida durante os ~7s de TTFT e o
        thinking) e só a resposta final em texto.
        """
        payload = {"model": MODEL, "messages": messages,
                   "max_tokens": max_tokens or MAX_TOK, "temperature": temperature,
                   "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})

        content_parts = []
        tool_acc = {}          # index -> {id, name, arguments}
        thinking = False
        answering = False
        t0 = time.perf_counter()
        ttft = None
        n_chunks = 0

        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                if ttft is None and (delta.get("content") or delta.get("reasoning")
                                     or delta.get("tool_calls")):
                    ttft = time.perf_counter() - t0

                # raciocínio (<think>): streamado em cinza. THINK=0 esconde e
                # mostra só um indicador de progresso.
                if delta.get("reasoning"):
                    if not thinking:
                        print(f"\033[2m[pensando]\033[0m " if SHOW_THINK
                              else "  \033[2mpensando\033[0m", end="", flush=True)
                        thinking = True
                    if SHOW_THINK:
                        print(f"\033[2m{delta['reasoning']}\033[0m", end="", flush=True)
                    else:
                        n_chunks += 1
                        if n_chunks % 8 == 0:
                            print("\033[2m.\033[0m", end="", flush=True)

                # resposta: texto ao vivo
                if delta.get("content"):
                    piece = delta["content"]
                    if not answering:
                        # o modelo abre a resposta com "\n\n"; corta pra não
                        # imprimir o prefixo antes de ter texto de verdade
                        piece = piece.lstrip("\n")
                        if not piece:
                            content_parts.append(delta["content"])
                            continue
                        if thinking:
                            print("\n")     # fecha o bloco de raciocínio
                        print("qwen38> ", end="", flush=True)
                        answering = True
                        content_parts.append(piece)
                        print(piece, end="", flush=True)
                        continue
                    print(piece, end="", flush=True)
                    content_parts.append(piece)

                # tool calls chegam em pedaços; acumula por índice
                for tc in delta.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = tool_acc.setdefault(i, {"id": None, "name": None, "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

        if answering:
            print()
        elif thinking:
            print()

        dt = time.perf_counter() - t0
        print(f"  \033[2m[{dt:.1f}s total"
              + (f", {ttft:.1f}s até o 1º token" if ttft else "") + "]\033[0m")

        tool_calls = [
            {"id": v["id"] or f"call_{i}", "type": "function",
             "function": {"name": v["name"], "arguments": v["arguments"]}}
            for i, v in sorted(tool_acc.items()) if v["name"]
        ]
        return {"role": "assistant", "content": "".join(content_parts),
                "tool_calls": tool_calls}

    print(f"Chat com {MODEL} @ {BASE} — streaming ligado, tools de exemplo ativas "
          f"(get_weather, calculate).")
    print(f"Raciocínio: {'visível' if SHOW_THINK else 'oculto'} (THINK=0/1 pra alternar). "
          f"max_tokens={MAX_TOK} (MAXTOK=N pra ajustar).")
    print("Ctrl-C ou 'sair' pra encerrar.\n")
    hist = []
    while True:
        try:
            user = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("sair", "exit", "quit"):
            break
        hist.append({"role": "user", "content": user})
        while True:
            m = chat_stream(hist, tools=TOOLS)
            tcs = m.get("tool_calls") or []
            if not tcs:
                hist.append({"role": "assistant", "content": m["content"]})
                print()
                break
            hist.append({"role": "assistant", "content": m["content"], "tool_calls": tcs})
            for tc in tcs:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"  \033[36m[tool]\033[0m {tc['function']['name']}({args})")
                hist.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": fake_tool_exec(tc["function"]["name"], args)})
