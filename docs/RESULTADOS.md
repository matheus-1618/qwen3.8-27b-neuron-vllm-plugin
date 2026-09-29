# Resultados públicos

O resultado vigente está em [`LONG_CONTEXT_TRN2_3XL.md`](LONG_CONTEXT_TRN2_3XL.md), com JSONs sanitizados em `../results/`.

## Configuração 12K validada

- trn2.3xlarge: um chip Trainium2, TP=4, 12 vCPU, 128 GiB
- BF16 pesos e KV
- `MAX_LEN=12288`, `SEG=KV_SEG=4096`, `MNS=4`
- `GMU=0.90`, `KV_CAP=0.05`
- vLLM-Neuron 0.21 / Neuron SDK 2.31

| Medição | Resultado |
|---|---:|
| KV runtime | 69,952 tokens / 5.69× janela |
| C1 sintético longo | 9,062 input tokens, HTTP 200, 27.81 s |
| C4 sintético longo | 4/4 HTTP 200, wall 104.83 s |
| Pico HBM C4 | 68.887 GiB/chip; 17.222 GiB/core |
| Suite funcional | 5/5 |

O wall C4 aproximadamente igual a quatro vezes C1 mostra serialização de prefill. O resultado valida capacidade e estabilidade, não scaling linear.

## Histórico

Experimentos 4K anteriores orientaram sharding e estado por sequência, mas usaram outros grafos e orçamentos. Eles não são publicados como comparação direta com o resultado 12K. Use os JSONs incluídos para qualquer claim quantitativo atual.
