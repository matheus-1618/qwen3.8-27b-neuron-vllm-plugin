#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# [CONTAINER] Fase 3: monta um model dir TEXT-ONLY do Qwen3.8-27B para o plugin público.
#
# Por quê (mesmo padrão do gemma4 PublicVLLM):
# - O config.json original tem `vision_config` → o plugin auto-gera vision_neuron_config
#   e chama from_configs(text_neuron_config=..., vision_neuron_config=...) — caminho
#   multimodal que nosso factory text-only não aceita (visto na prática: TypeError no boot).
# - Só remover `vision_config` NÃO funciona: transformers 5.14 re-injeta um vision_config
#   default no Qwen3_5Config (verificado empiricamente no DLC).
# - A solução é ACHATAR o text_config pro top-level → transformers resolve
#   Qwen3_5TextConfig (model_type qwen3_5_text), sem vision_config.
#
# Symlinks para os safetensors (sem duplicar 52GB); config.json/generation_config
# patchados como arquivos reais. Idempotente.
#
# Uso: python3 make_local_model.py [--arch NOME] [--src DIR] [--dst DIR]
import argparse
import glob
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--src", default="/root/models/Qwen3.8-27B")
p.add_argument("--dst", default="/root/models/Qwen3.8-27B-text")
p.add_argument("--arch", default="Qwen3NextForCausalLM",
               help="Arquitetura a gravar no config achatado. Default: Qwen3NextForCausalLM "
                    "— arch TEXT-ONLY da mesma família híbrida (GDN+GQA) que o vllm core "
                    "conhece; a arch original Qwen3_5ForConditionalGeneration é tratada "
                    "como multimodal pelo vllm core (exige vision_config + config wrapper "
                    "próprio, visto na prática). O plugin sobrescreve a classe no worker.")
args = p.parse_args()

os.makedirs(args.dst, exist_ok=True)

PATCHED = {"config.json"}
# Preprocessors de imagem/vídeo ficam FORA do dir text-only.
EXCLUDE = {"preprocessor_config.json", "video_preprocessor_config.json"}

for f in os.listdir(args.src):
    src = os.path.join(args.src, f)
    dst = os.path.join(args.dst, f)
    if f in PATCHED or f in EXCLUDE or f.startswith("."):
        continue
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)

cfg = json.load(open(os.path.join(args.src, "config.json")))
tc = dict(cfg["text_config"])

flat = tc  # base: text_config inteiro (model_type qwen3_5_text)
flat["architectures"] = [args.arch]
# tie_word_embeddings do top-level (false no 27B)
flat.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))

# Sanitiza mRoPE: com mrope_section presente, o vllm core seta uses_mrope=True e o
# runner passa `rotary_position_ids=` pro forward (assinatura multimodal). Em texto
# puro o mRoPE colapsa pra RoPE padrão (as 3 seções recebem posições idênticas —
# documentado no porte Qwen3.6), então removemos as chaves mrope e mantemos
# rope_theta/partial_rotary_factor.
rp = flat.get("rope_parameters")
if isinstance(rp, dict):
    for k in ("mrope_section", "mrope_interleaved"):
        rp.pop(k, None)

out = os.path.join(args.dst, "config.json")
json.dump(flat, open(out, "w"), indent=2, ensure_ascii=False)

n_st = len(glob.glob(os.path.join(args.dst, "*.safetensors")))
print(f"built {args.dst}")
print(f"  arch={args.arch} model_type={flat.get('model_type')} "
      f"layers={flat.get('num_hidden_layers')} vocab={flat.get('vocab_size')}")
print(f"  {n_st} safetensors symlinked; excluídos: {sorted(EXCLUDE)}")
