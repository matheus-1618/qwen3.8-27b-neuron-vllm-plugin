# SPDX-License-Identifier: Apache-2.0
"""Parity CPU do estado por sequência do DeltaNet (Fase 3 — MNS>1).

Prova, em CPU pura, que o caminho novo (max_batch=4, staging + anchors)
produz EXATAMENTE as mesmas saídas que o caminho legado (max_batch=1)
executado isoladamente por sequência, sob um schedule realista de
continuous batching:

    1. prefill A            → staged
    2. prefill B            → staged (prefills consecutivos)
    3. decode [A, B, -, -]  → consome staging
    4. decode [A, B, -, -]  → lê dos rows
    5. A termina + CONDENSE → decode [B, -, -, -] (self-healing por anchor)
    6. prefill C REUSANDO o bloco de A (anchor reciclado → dedupe)
    7. decode [B, C, -, -]
    8. prefill D em 2 SEGMENTOS (continuação lê staging) vs single-shot

O kernel NKI é substituído por uma referência sequencial em torch
(mesma matemática do passo de decode — o kernel chunked é equivalente).

Rodar no container:
  python3 /workspace/qwen38-27b-trn2/test/test_seq_state_parity.py
"""

import sys
import types

import torch

torch.manual_seed(7)

# ── Stub do TP group ANTES de importar o modelo ──────────────────────────
class _FakeTPGroup:
    world_size = 1
    rank_in_group = 0

sys.modules.setdefault("_fake_tp", types.ModuleType("_fake_tp"))

from vllm_neuron.model.qwen38 import model_bf16  # noqa: E402

model_bf16.get_tp_group = lambda: _FakeTPGroup()

# ── Referência torch do kernel (recorrência delta sequencial) ────────────
def _ref_deltanet_kernel(q, k, v, g, beta, lower, ident, lower_diag, init_state):
    """Assinatura de call_deltanet_fused; recorrência token a token."""
    S = q.shape[0]
    state = init_state.clone()          # [128, 128]
    outs = []
    for t in range(S):
        state = state * torch.exp(g[t, 0])
        kv_mem = (state * k[t].unsqueeze(-1)).sum(dim=0)       # [Dv]
        delta = (v[t] - kv_mem) * beta[t, 0]
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(0)
        outs.append((state * q[t].unsqueeze(-1)).sum(dim=0))   # [Dv]
    return torch.stack(outs, dim=0), state

from vllm_neuron.model.qwen38 import nki_kernels as _nki  # noqa: E402
_nki.call_deltanet_fused = _ref_deltanet_kernel

# ── Config mínima (head=128 é obrigatório; resto pequeno) ───────────────
HIDDEN, NUM_V, NUM_K, HEAD, KCONV = 256, 2, 1, 128, 4
BLOCK = 32

def make_cfg(max_batch):
    nc = types.SimpleNamespace(max_batch_size=max_batch, num_seqs_buckets=[max_batch])
    return types.SimpleNamespace(
        torch_dtype=torch.float32, rms_norm_eps=1e-6, hidden_size=HIDDEN,
        deltanet_num_v_heads=NUM_V, deltanet_num_k_heads=NUM_K,
        deltanet_k_head_dim=HEAD, deltanet_v_head_dim=HEAD,
        deltanet_conv_kernel_size=KCONV, neuron_config=nc,
    )

def make_layer(max_batch, weights=None):
    layer = model_bf16.Qwen38DeltaNetAttention(make_cfg(max_batch), layer_idx=0)
    if weights is None:
        weights = {}
        for n, p in layer.named_parameters():
            weights[n] = torch.randn_like(p) * 0.05
        weights["A_log"] = torch.rand(NUM_V) * 0.5
        weights["dt_bias"] = torch.ones(NUM_V)
    with torch.no_grad():
        for n, p in layer.named_parameters():
            p.copy_(weights[n])
    return layer, weights

LN = "layers.0.self_attn"

def md_prefill(first_block, n_tokens, pos0=0):
    """Metadata de prefill: slot_mapping real + tail padded (-1)."""
    slots = torch.arange(n_tokens, dtype=torch.long) + first_block * BLOCK + (pos0 % BLOCK)
    return {LN: {
        "max_query_len": n_tokens, "decode_token_threshold": 1,
        "slot_mapping": slots, "block_size": BLOCK,
    }}

def md_decode(anchors, seq_lens, mb=4):
    """Metadata de decode: rows válidas + padded. anchors/seq_lens por row (None = pad)."""
    bt = torch.full((mb, 8), -1, dtype=torch.int32)
    sm = torch.full((mb,), -1, dtype=torch.long)
    for r, (a, sl) in enumerate(zip(anchors, seq_lens)):
        if a is not None:
            bt[r, 0] = a
            sm[r] = a * BLOCK + sl
    return {LN: {
        "max_query_len": 1, "decode_token_threshold": 1,
        "slot_mapping": sm, "block_table_tensor": bt, "block_size": BLOCK,
        "max_blocks_per_seq": 8,
    }}

def prefill(layer, x, md, positions=None):
    if positions is None:
        positions = torch.arange(x.shape[0])
    return layer._forward_prefill(x, positions, md)

def decode(layer, x, md):
    return layer._forward_decode(x, md)


def main():
    failures = []
    MB = 4

    # Camada nova (MB=4) e camada legado (MB=1) com os MESMOS pesos
    L4, W = make_layer(MB)

    def fresh_legacy():
        l, _ = make_layer(1, W)
        return l

    # Sequências: prompts aleatórios + tokens de decode aleatórios
    xa = torch.randn(5, HIDDEN) * 0.1
    xb = torch.randn(7, HIDDEN) * 0.1
    xc = torch.randn(4, HIDDEN) * 0.1
    da = [torch.randn(1, HIDDEN) * 0.1 for _ in range(2)]
    db = [torch.randn(1, HIDDEN) * 0.1 for _ in range(4)]
    dc = [torch.randn(1, HIDDEN) * 0.1 for _ in range(1)]

    # ── Referência: cada sequência isolada na camada legado ─────────────
    LA, LB, LC = fresh_legacy(), fresh_legacy(), fresh_legacy()
    ref = {}
    ref["pa"] = prefill(LA, xa, md_prefill(10, 5))
    ref["pb"] = prefill(LB, xb, md_prefill(20, 7))
    ref["da"] = [decode(LA, t, md_decode([10], [5 + i], mb=1)) for i, t in enumerate(da)]
    ref["db"] = [decode(LB, t, md_decode([20], [7 + i], mb=1)) for i, t in enumerate(db)]
    ref["pc"] = prefill(LC, xc, md_prefill(10, 4))   # C reusa o bloco 10 de A
    ref["dc"] = [decode(LC, t, md_decode([10], [4 + i], mb=1)) for i, t in enumerate(dc)]

    def check(name, got, want, tol=2e-5):
        err = (got - want).abs().max().item()
        ok = err <= tol
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_err={err:.2e}")
        if not ok:
            failures.append(name)

    # ── Caminho novo: schedule intercalado na MESMA camada MB=4 ─────────
    def pad_rows(rows):
        """monta batch de decode [4, hidden] com lixo nas rows padded"""
        out = torch.randn(MB, HIDDEN) * 0.1
        for r, t in enumerate(rows):
            if t is not None:
                out[r] = t[0]
        return out

    # 1-2. prefills consecutivos
    check("prefill A", prefill(L4, xa, md_prefill(10, 5)), ref["pa"])
    check("prefill B", prefill(L4, xb, md_prefill(20, 7)), ref["pb"])

    # 3. decode [A, B, -, -] — consome staging
    y = decode(L4, pad_rows([da[0], db[0], None, None]),
               md_decode([10, 20, None, None], [5, 7, None, None]))
    check("decode1 rowA", y[0], ref["da"][0][0])
    check("decode1 rowB", y[1], ref["db"][0][0])

    # 4. decode [A, B, -, -] — lê dos rows
    y = decode(L4, pad_rows([da[1], db[1], None, None]),
               md_decode([10, 20, None, None], [6, 8, None, None]))
    check("decode2 rowA", y[0], ref["da"][1][0])
    check("decode2 rowB", y[1], ref["db"][1][0])

    # 5. A termina → CONDENSE: B vai pra row 0 (self-healing por anchor)
    y = decode(L4, pad_rows([db[2], None, None, None]),
               md_decode([20, None, None, None], [9, None, None, None]))
    check("decode3 pós-condense B", y[0], ref["db"][2][0])

    # 6. prefill C reusando o bloco 10 (anchor reciclado → dedupe)
    check("prefill C (anchor reciclado)", prefill(L4, xc, md_prefill(10, 4)), ref["pc"])

    # 7. decode [B, C, -, -]
    y = decode(L4, pad_rows([db[3], dc[0], None, None]),
               md_decode([20, 10, None, None], [10, 4, None, None]))
    check("decode4 rowB", y[0], ref["db"][3][0])
    check("decode4 rowC (staging)", y[1], ref["dc"][0][0])

    # 8. prefill segmentado com MNS>1: D em 2 segmentos == single-shot
    xd = torch.randn(8, HIDDEN) * 0.1
    LD = fresh_legacy()
    ref_d = prefill(LD, xd, md_prefill(30, 8))
    L4b, _ = make_layer(MB, W)
    seg1 = prefill(L4b, xd[:4], md_prefill(30, 4), positions=torch.arange(4))
    seg2 = prefill(L4b, xd[4:], md_prefill(30, 4, pos0=4), positions=torch.arange(4, 8))
    check("prefill D seg1 == single[:4]", seg1, ref_d[:4])
    check("prefill D seg2 == single[4:]", seg2, ref_d[4:])
    st = L4b.staging_recurrent[0]
    st_ref = LD.recurrent_state_buffer[0]
    check("estado D staged == legado", st, st_ref)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
