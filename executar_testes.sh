#!/bin/bash

# Lista de arquivos Python a serem executados sequencialmente
comandos=(
    # prompt especialista com evaluator bf
    "./main.py --task despacho --data_dir ./data/despacho_saneador/dados_json/ --prompts ./prompts/despacho_saneador/despacho_saneador_especialista.md --nome_saida prompt_especialista_despacho_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 6 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
)

# Detecta se está rodando no Windows (via Git Bash ou WSL) ou Linux/macOS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PYTHON_CMD="py"  # Para Windows
else
    PYTHON_CMD="python3"  # Para Linux/macOS
fi

# Executar os comandos sequencialmente (um por vez)
for script in "${comandos[@]}"; do
    echo "Executando $script..."
    $PYTHON_CMD $script
done

# Manter o terminal aberto (para Git Bash ou WSL)
echo "Pressione qualquer tecla para sair..."
read -n 1 -s
