# Qwen3.8-27B em Trainium2 (trn2.3xlarge) — vLLM-Neuron público

Porte de [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) pro plugin
**público** [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
(0.21.0.1.0.0 / SDK 2.31), servindo numa trn2.3xlarge (TP=4).

- Racional completo, análise da arquitetura, riscos e fases: [`../PLANO.md`](../PLANO.md)
- Log de execução e estado atual: [`../PROGRESSO.md`](../PROGRESSO.md)
- Números medidos (correção, perf, memória): [`RESULTADOS.md`](./RESULTADOS.md)
- Limitações e caminhos de melhoria: [`ROADMAP.md`](./ROADMAP.md)
- Preços e comparação de instâncias: [`CUSTO_ALTERNATIVAS.md`](./CUSTO_ALTERNATIVAS.md)

## Segurança / exposição de rede

**A instância não tem nenhuma porta de inferência aberta, e não deve ter.** A API do
vLLM não tem autenticação, e a instância tem IP público — expor a 8000 seria dar
inferência (e a caixa) de graça pra internet.

Regras em vigor:
- **Security group**: só `tcp/22` liberado, e apenas do IP do dono (`/32`). Nada de 8000.
- **`04_serve.sh` binda em `127.0.0.1`** (loopback) por padrão, não em `0.0.0.0`.
  Defesa em profundidade: mesmo se alguém abrir o SG por acidente, não há listener
  na interface pública.
- **`01_container.sh` não publica portas** (`-p`). O container roda com `--network host`,
  então a API vive no loopback do host.
- **Acesso remoto é só por túnel**: `scripts/chat.sh` levanta um `ssh -L` sobre SSM.
  O tráfego vai dentro da sessão SSM/SSH — nenhuma porta de entrada precisa existir.
- **Administração via SSM** (`scripts/ssh.sh`), que não exige regra de entrada nenhuma.

Se precisar de acesso de mais alguém, o caminho é dar a eles SSM/SSH — não abrir porta.

## Replicar do zero

Pré-requisitos no laptop: AWS CLI com credenciais da conta, `session-manager-plugin`,
a chave `YOUR_KEY.pem` na raiz da pasta, instância trn2.3xlarge AL2023 com SSM habilitado.

```bash
cd qwen38-27b-trn2/scripts

# 0. sync do projeto pra instância + shell
./sync.sh
./ssh.sh

# 1. [INSTÂNCIA] host: driver neuron + docker
bash ~/qwen38-27b-trn2/scripts/00_setup_host.sh

# 2. [INSTÂNCIA] container DLC público (pull ~10GB) — pode rodar em paralelo com o 3
bash ~/qwen38-27b-trn2/scripts/01_container.sh

# 3. [INSTÂNCIA] download do modelo (~54GB)
bash ~/qwen38-27b-trn2/scripts/02_download_model.sh

# 4. [CONTAINER] instala o pacote qwen38 no plugin + monta model dir text-only
sudo docker exec -it vllm_qwen38 bash -lc \
  'bash /workspace/qwen38-27b-trn2/scripts/03_install_plugin.sh && \
   python3 /workspace/qwen38-27b-trn2/scripts/make_local_model.py'

# 5. [CONTAINER] serve (primeiro boot compila ~8-20min por bucket; NEFF cache depois)
sudo docker exec -it vllm_qwen38 bash -lc \
  'MODEL=/root/models/Qwen3.8-27B-text MAX_LEN=4096 bash /workspace/qwen38-27b-trn2/scripts/04_serve.sh'

# 6. testes (na instância, ou no laptop com port-forward)
bash ~/qwen38-27b-trn2/scripts/05_teste.sh          # suite smoke + tool calling
bash ~/qwen38-27b-trn2/scripts/05_teste.sh chat     # chat interativo com tools
```

Port-forward pro laptop: `./ssh.sh -L 8000:localhost:8000 -N` e então
`BASE=http://localhost:8000 bash 05_teste.sh chat`.

## Layout

```
serving_pkg/qwen38/   # pacote do modelo (instalado em vllm_neuron/model/qwen38)
scripts/              # pipeline replicável 00→05 + ssh/sync helpers
API_NOTES.md          # catálogo de diferenças API beta v5 -> plugin público (fase 1)
RESULTADOS.md         # números medidos na 3xl (fase 4)
```

## Estado

Ver [`../PROGRESSO.md`](../PROGRESSO.md). O `serving_pkg/qwen38/` é construído na Fase 2
adaptando `<internal reference port: qwen3.6-27b>` (mesma arquitetura, validado
cosine 1.0 vs HF) pra API do plugin público, usando
`<internal reference port: gemma4-31b>` como
gabarito de integração.
