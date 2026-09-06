# SPDX-License-Identifier: Apache-2.0
"""Testes CPU-only do pacote qwen38 no plugin público (Fase 2).

Rodar DENTRO do container público, DEPOIS de scripts/03_install_plugin.sh:

    python3 /workspace/qwen38-27b-trn2/test/test_cpu.py [/root/models/Qwen3.8-27B]

Valida (sem device Neuron):
1. registry do plugin contém Qwen3_5ForConditionalGeneration -> Qwen38ForCausalLM
2. Qwen38Config parseia o config.json real do Qwen/Qwen3.8-27B
3. weight mappings cobrem 100% dos tensores do safetensors index
   (menos os prefixos deliberadamente pulados: mtp.*, model.visual.*)
"""

import json
import os
import sys


def main(model_path: str) -> int:
    failures = []

    # ── 1. registry ──────────────────────────────────────────────────
    print("[1/3] registry ... ", end="", flush=True)
    from vllm_neuron.model.registry import get_models
    models = dict(get_models())
    assert "Qwen3_5ForConditionalGeneration" in models, list(models)
    cls = models["Qwen3_5ForConditionalGeneration"]
    assert cls.__name__ == "Qwen38ForCausalLM", cls.__name__
    print(f"ok ({cls.__module__}.{cls.__name__})")

    # ── 2. config parse ──────────────────────────────────────────────
    print("[2/3] Qwen38Config.from_configs ... ", end="", flush=True)
    from vllm_neuron.model.qwen38 import Qwen38Config
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"SKIP (sem modelo em {model_path})")
        cfg = None
    else:
        hf_cfg = json.load(open(cfg_path))
        cfg = Qwen38Config.from_configs(hf_cfg, neuron_config=None)
        assert cfg.num_hidden_layers == 64
        assert cfg.num_full_attention_layers == 16
        assert cfg.num_linear_attention_layers == 48
        assert cfg.vocab_size == 248320
        assert cfg.attn_output_gate is True
        assert cfg.tie_word_embeddings is False
        assert cfg.head_dim == 256 and cfg.partial_rotary_factor == 0.25
        assert cfg.deltanet_num_v_heads == 48 and cfg.deltanet_num_k_heads == 16
        print("ok (64L = 16 GQA + 48 GDN, vocab 248320)")

    # ── 3. weight mapping coverage vs safetensors index ──────────────
    print("[3/3] weight mappings vs safetensors index ... ", end="", flush=True)
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if cfg is None or not os.path.isfile(idx_path):
        print("SKIP (sem index)")
    else:
        from vllm_neuron.model.qwen38.weight_loaders_bf16 import build_weight_mappings
        mappings = build_weight_mappings(cfg)
        wanted = set()
        for hf_keys in mappings.values():
            wanted.update(hf_keys)

        disk = set(json.load(open(idx_path))["weight_map"].keys())
        SKIP_PREFIXES = ("mtp.", "model.visual.", "visual.")
        disk_needed = {k for k in disk if not k.startswith(SKIP_PREFIXES)}

        missing_on_disk = wanted - disk           # mapeamos algo que não existe
        uncovered = disk_needed - wanted          # existe no disco e não mapeamos
        if missing_on_disk:
            failures.append(f"mapeados mas ausentes no disco ({len(missing_on_disk)}): "
                            f"{sorted(missing_on_disk)[:8]}")
        if uncovered:
            failures.append(f"no disco mas não mapeados ({len(uncovered)}): "
                            f"{sorted(uncovered)[:8]}")
        if not failures:
            print(f"ok ({len(wanted)} tensores mapeados, "
                  f"{len(disk) - len(disk_needed)} pulados por prefixo)")
        else:
            print("FAIL")
            for f in failures:
                print("   ", f)

    if failures:
        print("\nFAIL")
        return 1
    print("\nTestes CPU OK.")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/models/Qwen3.8-27B"
    raise SystemExit(main(path))
