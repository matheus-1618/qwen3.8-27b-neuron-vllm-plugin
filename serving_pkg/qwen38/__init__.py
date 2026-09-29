# SPDX-License-Identifier: Apache-2.0
# Modified from: Modified from the attributed Qwen3.6 public example for Qwen3.8 integration.
"""Qwen3.8-27B (hybrid GatedDeltaNet + GQA) para o plugin público vLLM-Neuron.

Serve `Qwen/Qwen3.8-27B` via `vllm serve` em Trainium2 no plugin público
0.21.0.1.0.0 (vLLM 0.21 / Neuron SDK 2.31).

Arquitetura (idêntica ao Qwen3.6-27B; classe HF `Qwen3_5ForConditionalGeneration`):
  64 layers no padrão [3× DeltaNet + 1× GQA]; hidden 5120; 24 Q / 4 KV heads,
  head_dim 256; DeltaNet 48 v-heads / 16 k-heads @128; partial RoPE 25%;
  SwiGLU int 17408; RMSNorm (1+weight); attn_output_gate; vocab 248320;
  tie_word_embeddings=False. MTP head e vision tower do checkpoint são pulados
  (text-only serving).

Origem técnica: arquitetura e integração informadas pelo exemplo público
`qwen3.6-27b-trainium` de Armin Agha-Ebrahim no commit
fc1af21a8620c97e6f0d67f48f89a8388a569808, adaptadas aqui para Qwen3.8 e para
o plugin vLLM-Neuron 0.21. O fix de persistência usa buffers epsilon 1e-30.
Integração via vllm_neuron/model/registry.py (scripts/03_install_plugin.sh).
"""

from .config import Qwen38Config
from .factory import Qwen38ForCausalLM

__all__ = [
    "Qwen38Config",
    "Qwen38ForCausalLM",
]
