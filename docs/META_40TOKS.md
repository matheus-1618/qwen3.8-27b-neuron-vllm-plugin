# Metas de otimização

Estado validado: Qwen3.8-27B BF16/TP4 com janela 12,288, KV dimensionado para quatro janelas e endpoint funcional 5/5. Quatro prompts longos foram admitidos sem OOM, mas os prefills ficaram serializados.

## Metas mensuráveis

1. Reduzir TPOT sem alterar precisão ou qualidade.
2. Evitar o gather da janela inteira no decode `head_dim=256`.
3. Reutilizar prefixos no modelo híbrido, incluindo estado DeltaNet.
4. Demonstrar scaling de concorrência além de simples enfileiramento.
5. Executar um benchmark H100 apples-to-apples e calcular custo por resposta válida.

## Regras

- Cada otimização precisa de paridade numérica/funcional e A/B com o mesmo workload.
- BF16/TP4 single-chip permanece como baseline; FP8 ou mais chips são cenários separados.
- Cache de compilação só é reutilizado quando código, shapes, flags e stack coincidem.
- Números históricos de 4K não são usados para representar a configuração 12K.
