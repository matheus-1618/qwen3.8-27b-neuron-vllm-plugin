# Roadmap técnico

## Estado atual

- Qwen3.8-27B text-only no vLLM-Neuron 0.21
- um chip Trainium2, TP=4, BF16
- janela 12,288 e quatro requests longos admitidos sem OOM
- 69,952 tokens de KV com `KV_CAP=0.05`
- prefills longos serializados

## P0 — decode paginado para head_dim 256

O decode atual reúne o KV correspondente à janela estática e executa atenção explícita porque o megakernel disponível limita `head_dim` a 128. O objetivo é ler apenas os blocos necessários sem materializar toda a janela por camada/token.

Gate: mesma saída/top-1 que o caminho atual, redução de TPOT e nenhum aumento de HBM.

## P1 — segmented attention NKI para head_dim 256

O prefill segmentado funciona, mas o caminho disponível para essa dimensão não entrega o kernel ideal. Implementar d-tiling em 128 com softmax online e validar contra referência CPU/torch.

Gate: paridade numérica, needle test e melhora de TTFT em 4K/8K/12K.

## P1 — prefix cache híbrido

KV cache sozinho não representa o prefixo: 48 camadas dependem de estado DeltaNet e conv causal. O cache precisa incluir o estado recorrente no fim do prefixo e restaurá-lo por identidade de request.

Gate: prefixo compartilhado não é recomputado; resposta idêntica ao prefill completo.

## P2 — concorrência e compilação

Eliminar serialização de prefills e tornar o trace paralelo robusto aos arquivos temporários NKI. Qualquer mudança deve preservar isolamento por sequência e cache validado.

## Comparação com H100

O exemplo público Gemma4 de [Armin-Neuron](https://github.com/arminagha1234/Armin-Neuron/tree/fc1af21a8620c97e6f0d67f48f89a8388a569808/gemma4-31b/vllm-neuron-4k_16k_32k_64_PublicVLLM) traz uma referência útil em TP=8/TP=32, mas não prova desempenho deste Qwen TP=4. Execute o mesmo harness, revisão do modelo, precisão, schema e cache em Trainium2 e H100 antes de comparar custo/desempenho.
