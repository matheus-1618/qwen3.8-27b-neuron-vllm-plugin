# Aprendizados do port Qwen3.8-27B

## Estado recorrente faz parte do contrato de serving

O Qwen3.8 combina 48 camadas GatedDeltaNet com 16 camadas de atenção completa. Preservar apenas o KV cache não basta: o estado recorrente e o estado da convolução causal precisam sobreviver entre prefill segmentado e decode, e precisam acompanhar a identidade da requisição quando o batch é condensado.

Buffers zerados podem ser constant-folded pelo compilador. A implementação usa um epsilon numericamente desprezível e mutações capturáveis, validadas por testes de paridade/estado.

## Padding não é neutro em modelos stateful

Tokens de padding descartados pelo sampler ainda podem alterar o estado DeltaNet. Slot mapping, máscara de continuação e estado por sequência precisam excluir padding explicitamente.

## Orçamento KV deve refletir a arquitetura

Com o cap padrão, o runtime reservou 6.48 GiB/core para 419,744 tokens — excessivo para um modelo que mantém KV completo em apenas 16/64 camadas. Para o envelope 12K×4, `KV_CAP=0.05` produziu 69,952 tokens e liberou aproximadamente 5.4 GiB/core para scratch e estado recorrente.

Esse valor não é universal. Recalcule quando alterar `MAX_LEN`, `MNS`, precisão ou versão do plugin.

## Compilação também é parte da reprodutibilidade

O trace paralelo apresentou `NCC_EVRF059`: o HLO referenciava um binário NKI temporário removido antes do `neuronx-cc`. Cache isolado e trace sequencial produziram um artefato válido. Preserve versão do DLC, flags, hash do código e cache validado.

## Benchmarks devem separar configurações

Resultados históricos de 4K e resultados atuais de 12K usam grafos e envelopes diferentes. Não combine números entre eles. A comparação com H100 deve usar o mesmo modelo, precisão, prompts, comprimento de saída, cache, schema e load generator, seguida de custo por resposta bem-sucedida.

## Próximos passos

1. Decode paginado para `head_dim=256`.
2. Segmented attention NKI para `head_dim=256`.
3. Prefix cache do estado recorrente DeltaNet.
4. Harness público idêntico em Trainium2 e H100.
