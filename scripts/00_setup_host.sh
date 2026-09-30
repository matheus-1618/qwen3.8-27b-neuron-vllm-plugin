#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [INSTÂNCIA] Fase 0: prepara o host AL2023 pelado da trn2.3xlarge.
# Instala driver Neuron (dkms) + tools + docker. Idempotente.
# Uso: bash 00_setup_host.sh
set -euo pipefail

echo "=== [0/4] Repo Neuron (AL2023) ==="
if [ ! -f /etc/yum.repos.d/neuron.repo ]; then
  sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<'EOF'
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
  sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
fi

echo "=== [1/4] Driver Neuron (dkms) + tools ==="
# AL2023 usa kernel streams: o pacote devel do kernel 6.18 é kernel6.18-devel (não kernel-devel).
KVER="$(uname -r)"                      # ex: 6.18.39-79.141.amzn2023.x86_64
KSTREAM="kernel$(echo "$KVER" | cut -d. -f1,2)"   # ex: kernel6.18
sudo dnf install -y "${KSTREAM}-devel-1:${KVER%.x86_64}" git 2>&1 | tail -2
sudo dnf install -y aws-neuronx-dkms aws-neuronx-tools 2>&1 | tail -2

echo "=== [2/4] Docker ==="
if ! command -v docker >/dev/null; then
  sudo dnf install -y docker 2>&1 | tail -2
fi
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user || true

echo "=== [3/4] Verificação ==="
ls -l /dev/neuron* || { echo "ERRO: /dev/neuron* ausente — driver não subiu (checar dmesg | grep -i neuron)"; exit 1; }
export PATH=/opt/aws/neuron/bin:$PATH
neuron-ls || { echo "ERRO: neuron-ls falhou"; exit 1; }
sudo docker info >/dev/null && echo "docker OK"

echo "=== [4/4] PATH persistente ==="
grep -q '/opt/aws/neuron/bin' ~/.bashrc || echo 'export PATH=/opt/aws/neuron/bin:$PATH' >> ~/.bashrc

echo "SETUP HOST OK — trn2.3xlarge = 1 device Neuron esperado acima."
