# Qwen3.8-27B no Trainium2 — guia do port

Este diretório documenta um port experimental do Qwen3.8-27B para o plugin público vLLM-Neuron 0.21.

- API e diferenças de integração: [`API_NOTES.md`](API_NOTES.md)
- Resultados históricos de engenharia: [`RESULTADOS.md`](RESULTADOS.md)
- Roadmap técnico: [`ROADMAP.md`](ROADMAP.md)
- Contexto 12K validado: [`LONG_CONTEXT_TRN2_3XL.md`](LONG_CONTEXT_TRN2_3XL.md)
- Design de estado por sequência: [`FASE3_DESIGN.md`](FASE3_DESIGN.md)

## Proveniência

A arquitetura e a abordagem inicial de integração foram informadas pelo exemplo público [`qwen3.6-27b-trainium`](https://github.com/arminagha1234/Armin-Neuron/tree/fc1af21a8620c97e6f0d67f48f89a8388a569808/qwen3.6-27b-trainium), de Armin Agha-Ebrahim, e pelo plugin público [`vllm-neuron`](https://github.com/vllm-project/vllm-neuron). Este repositório adapta a implementação para Qwen3.8 e inclui as mudanças de estado recorrente, prefill segmentado e orçamento KV descritas nos demais documentos.

## Fluxo incluído

```bash
bash scripts/00_setup_host.sh
REPO_DIR=$PWD bash scripts/01_container.sh
bash scripts/02_download_model.sh
sudo docker exec -it vllm_qwen38 bash -lc \
  'cd /workspace/qwen38-27b-trn2 && bash scripts/03_install_plugin.sh && python3 scripts/make_local_model.py'
```

Para o primeiro smoke conservador, use 4K/MNS1. Para 12K/MNS4, use exatamente a configuração e as ressalvas de [`LONG_CONTEXT_TRN2_3XL.md`](LONG_CONTEXT_TRN2_3XL.md).

## Limitações

- O endpoint não possui autenticação; o serve binda em loopback por padrão.
- Prefills longos ficam serializados no scheduler atual.
- O decode de atenção reúne a janela configurada e ainda precisa de um caminho paginado otimizado para `head_dim=256`.
- Resultados antigos de 4K e os resultados de 12K representam configurações distintas e não devem ser combinados numa única comparação.
