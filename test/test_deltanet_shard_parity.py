# SPDX-License-Identifier: Apache-2.0
"""Parity CPU do sharding do DeltaNet (Fase 2 do plano de perf).

Prova, em CPU pura (sem Neuron), que o v-head sharding é matematicamente
idêntico ao caminho replicado:

    full(48 heads)  ==  sum_r  local_r(12 heads, slices dos loaders)

Cobre os DOIS caminhos:
  - prefill: conv causal completa + recorrência sequencial (equivalente
    exato do kernel chunked) sobre S tokens
  - decode: um passo da recorrência a partir de estado aleatório

Replica as MESMAS operações do model_bf16.py (projeções, conv depthwise,
split q/k/v, expand kv_repeat, l2norm, decay/beta, recorrência delta,
RMSNorm por head, z-gate, out_proj) e os MESMOS slices dos weight loaders
(_deltanet_qkv_rows_loader e sharding_weight_loader). Se um slice estiver
errado (ex.: mapeamento v-head→k-head), a soma dos ranks diverge do full.

Rodar no container:  python3 /workspace/qwen38-27b-trn2/test/test_deltanet_shard_parity.py
"""

import sys

import torch

torch.manual_seed(0)

# Dims reais do Qwen3.8-27B
HIDDEN = 5120
NUM_V = 48
NUM_K = 16
HEAD = 128
KCONV = 4
WS = 4
EPS = 1e-6

KEY_DIM = NUM_K * HEAD          # 2048
VALUE_DIM = NUM_V * HEAD        # 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 10240
KV_REPEAT = NUM_V // NUM_K      # 3


def l2norm(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-6)


def deltanet_forward(x, w, state0=None, conv_state0=None):
    """Caminho DeltaNet em torch puro, espelhando model_bf16.py.

    x: [S, hidden] (batch 1). w: dict de pesos GLOBAIS ou LOCAIS (shapes
    definem os head counts). Recorrência sequencial token a token —
    matematicamente igual ao kernel chunked do prefill e, com S=1 e
    estados iniciais, igual ao passo de decode.
    Retorna (out [S, hidden] SEM out_proj somado entre ranks, state, conv_state).
    """
    S = x.shape[0]
    num_v = w["A_log"].shape[0]
    # dims derivadas dos shapes reais
    key_dim = (w["in_proj_qkv"].shape[0] - num_v * HEAD) // 2
    num_k = key_dim // HEAD
    value_dim = num_v * HEAD
    conv_dim = 2 * key_dim + value_dim
    kv_repeat = num_v // num_k

    xb = x.unsqueeze(0)  # [1, S, hidden]

    qkv = torch.nn.functional.linear(xb, w["in_proj_qkv"])
    z = torch.nn.functional.linear(xb, w["in_proj_z"])
    a = torch.nn.functional.linear(xb, w["in_proj_a"])
    b = torch.nn.functional.linear(xb, w["in_proj_b"])

    mixed = qkv.transpose(1, 2)  # [1, conv_dim, S]
    if conv_state0 is None:
        conv_in = mixed
        pad = KCONV - 1
        conv_out = torch.nn.functional.conv1d(
            conv_in, w["conv1d"], bias=None, stride=1, padding=pad, groups=conv_dim
        )[:, :, :S]
    else:
        conv_in = torch.cat([conv_state0, mixed], dim=-1)  # [1, conv_dim, 3+S]
        conv_out = torch.nn.functional.conv1d(
            conv_in, w["conv1d"], bias=None, stride=1, groups=conv_dim
        )  # valid conv -> [1, conv_dim, S]
    new_conv_state = torch.cat([conv_state0, mixed], dim=-1)[:, :, -(KCONV - 1):] if conv_state0 is not None \
        else torch.nn.functional.pad(mixed, (KCONV - 1 - S, 0))[:, :, -(KCONV - 1):] if S < KCONV - 1 \
        else mixed[:, :, -(KCONV - 1):]

    mpc = torch.nn.functional.silu(conv_out).transpose(1, 2)  # [1, S, conv_dim]

    q = mpc[..., :key_dim].reshape(1, S, num_k, HEAD)
    k = mpc[..., key_dim:2 * key_dim].reshape(1, S, num_k, HEAD)
    v = mpc[..., 2 * key_dim:].reshape(1, S, num_v, HEAD)

    g = -w["A_log"].exp() * torch.nn.functional.softplus(a.float() + w["dt_bias"])
    beta = b.sigmoid().float()

    if kv_repeat > 1:
        q = q.unsqueeze(3).expand(-1, -1, -1, kv_repeat, -1).reshape(1, S, num_v, HEAD)
        k = k.unsqueeze(3).expand(-1, -1, -1, kv_repeat, -1).reshape(1, S, num_v, HEAD)

    q = q.transpose(1, 2).float()   # [1, H, S, D]
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    g = g.transpose(1, 2).float()   # [1, H, S]
    beta = beta.transpose(1, 2).float()

    q = l2norm(q) * (HEAD ** -0.5)
    k = l2norm(k)

    state = state0.clone() if state0 is not None else torch.zeros(1, num_v, HEAD, HEAD)
    outs = []
    for t in range(S):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))  # [1, H, D]

    out = torch.stack(outs, dim=2)  # [1, H, S, D]
    out = out.transpose(1, 2)       # [1, S, H, D]

    var = out.pow(2).mean(-1, keepdim=True)
    out = out * torch.rsqrt(var + EPS)
    out = out * w["norm"]

    out = out.reshape(1, S, value_dim)
    gated = out * torch.nn.functional.silu(z.float())
    result = torch.nn.functional.linear(gated, w["out_proj"].float())
    return result.squeeze(0), state, new_conv_state


def shard(wg, r):
    """Slices EXATAMENTE como os weight loaders do model_bf16.py."""
    kl, vl, hl = KEY_DIM // WS, VALUE_DIM // WS, NUM_V // WS

    def qkv_rows(t):
        qs = t[kl * r: kl * (r + 1)]
        ks = t[KEY_DIM + kl * r: KEY_DIM + kl * (r + 1)]
        vs = t[2 * KEY_DIM + vl * r: 2 * KEY_DIM + vl * (r + 1)]
        return torch.cat([qs, ks, vs], dim=0)

    return {
        "in_proj_qkv": qkv_rows(wg["in_proj_qkv"]),
        "conv1d": qkv_rows(wg["conv1d"]),
        "in_proj_z": wg["in_proj_z"][vl * r: vl * (r + 1)],
        "in_proj_a": wg["in_proj_a"][hl * r: hl * (r + 1)],
        "in_proj_b": wg["in_proj_b"][hl * r: hl * (r + 1)],
        "A_log": wg["A_log"][hl * r: hl * (r + 1)],
        "dt_bias": wg["dt_bias"][hl * r: hl * (r + 1)],
        "norm": wg["norm"],  # replicado
        "out_proj": wg["out_proj"][:, vl * r: vl * (r + 1)],  # column shard
    }


def shard_state(state, conv_state, r):
    hl, kl, vl = NUM_V // WS, KEY_DIM // WS, VALUE_DIM // WS

    def qkv_ch(t):
        qs = t[:, kl * r: kl * (r + 1)]
        ks = t[:, KEY_DIM + kl * r: KEY_DIM + kl * (r + 1)]
        vs = t[:, 2 * KEY_DIM + vl * r: 2 * KEY_DIM + vl * (r + 1)]
        return torch.cat([qs, ks, vs], dim=1)

    return state[:, hl * r: hl * (r + 1)], qkv_ch(conv_state)


def main():
    S = 19  # não múltiplo de 4 de propósito (aqui não há SP de sequência)
    x = torch.randn(S, HIDDEN, dtype=torch.float32) * 0.5

    wg = {
        "in_proj_qkv": torch.randn(CONV_DIM, HIDDEN) * 0.02,
        "in_proj_z": torch.randn(VALUE_DIM, HIDDEN) * 0.02,
        "in_proj_a": torch.randn(NUM_V, HIDDEN) * 0.02,
        "in_proj_b": torch.randn(NUM_V, HIDDEN) * 0.02,
        "conv1d": torch.randn(CONV_DIM, 1, KCONV) * 0.2,
        "A_log": torch.randn(NUM_V) * 0.1,
        "dt_bias": torch.ones(NUM_V),
        "norm": torch.randn(HEAD) * 0.1 + 1.0,
        "out_proj": torch.randn(HIDDEN, VALUE_DIM) * 0.02,
    }

    fails = 0

    # ── 1. PREFILL: full vs soma dos ranks ────────────────────────────
    out_full, state_full, conv_full = deltanet_forward(x, wg)
    out_sum = torch.zeros_like(out_full)
    states_l, convs_l = [], []
    for r in range(WS):
        o, st, cv = deltanet_forward(x, shard(wg, r))
        out_sum += o  # emula all_reduce/reduce_scatter do out_proj parcial
        states_l.append(st)
        convs_l.append(cv)
    err = (out_full - out_sum).abs().max().item()
    ok = err < 1e-3
    print(f"[1/4] prefill parity: max|full - sum(ranks)| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # estados locais devem bater com os slices do estado full
    serr = max(
        (states_l[r] - shard_state(state_full, conv_full, r)[0]).abs().max().item()
        for r in range(WS)
    )
    cerr = max(
        (convs_l[r] - shard_state(state_full, conv_full, r)[1]).abs().max().item()
        for r in range(WS)
    )
    ok = serr < 1e-4 and cerr < 1e-4
    print(f"[2/4] state parity:   recurrent {serr:.2e} · conv {cerr:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # ── 2. DECODE: um passo com estado inicial aleatório ──────────────
    state0 = torch.randn(1, NUM_V, HEAD, HEAD) * 0.05
    conv0 = torch.randn(1, CONV_DIM, KCONV - 1) * 0.5
    x1 = torch.randn(1, HIDDEN) * 0.5
    out_full, _, _ = deltanet_forward(x1, wg, state0=state0, conv_state0=conv0)
    out_sum = torch.zeros_like(out_full)
    for r in range(WS):
        st0, cv0 = shard_state(state0, conv0, r)
        o, _, _ = deltanet_forward(x1, shard(wg, r), state0=st0, conv_state0=cv0)
        out_sum += o
    err = (out_full - out_sum).abs().max().item()
    ok = err < 1e-3
    print(f"[3/4] decode parity:  max|full - sum(ranks)| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # ── 3. FASE 4: prefill segmentado == prefill single-shot ──────────
    # full(S) deve ser idêntico a seg1(S/2 do zero) + seg2(S/2 retomando
    # recurrent_state E conv_state). Valida a continuação da recorrência
    # e o prefixo causal da conv — a matemática nova do kernel/modelo.
    S2 = 32
    xs = torch.randn(S2, HIDDEN) * 0.5
    out_full, state_full, conv_full = deltanet_forward(xs, wg)
    o1, st1, cv1 = deltanet_forward(xs[: S2 // 2], wg)
    o2, st2, cv2 = deltanet_forward(xs[S2 // 2 :], wg, state0=st1, conv_state0=cv1)
    out_seg = torch.cat([o1, o2], dim=0)
    err_o = (out_full - out_seg).abs().max().item()
    err_s = (state_full - st2).abs().max().item()
    err_c = (conv_full - cv2).abs().max().item()
    ok = err_o < 1e-3 and err_s < 1e-3 and err_c < 1e-4
    print(f"[4/4] segmented parity: out {err_o:.2e} · state {err_s:.2e} · conv {err_c:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
