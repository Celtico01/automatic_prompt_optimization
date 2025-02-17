#!/bin/bash

# Caminho do ambiente virtual (ajuste conforme necessário)
VENV_DIR="./.venv_310"

# Ativar o ambiente virtual
if [[ -d "$VENV_DIR" ]]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
else
    echo "Ambiente virtual não encontrado em $VENV_DIR!"
    exit 1
fi

# Lista de comandos Python a serem executados sequencialmente
comandos=(
    # bf
    #"main.py --task despacho --data_dir ./data/despacho_saneador/dados_json/ --prompts ./prompts/despacho_saneador/despacho_saneador_especialista.md --nome_saida prompt_especialista_despacho_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 6 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
    #"main.py --task despacho --data_dir ./data/despacho_saneador/dados_json/ --prompts ./prompts/despacho_saneador/despacho_saneador_promptbreeder.md --nome_saida prompt_promptbreeder_despacho_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 6 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
    "main.py --task replica --data_dir ./data/replica/dados_json/ --prompts ./prompts/replica/replica_especialista.md --nome_saida prompt_especialista_replica_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
    "main.py --task replica --data_dir ./data/replica/dados_json/ --prompts ./prompts/replica/replica_promptbreeder.md --nome_saida prompt_promptbreeder_replica_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
    "main.py --task decisao_inicial --data_dir ./data/decisao_inicial/dados_json/ --prompts ./prompts/decisao_inicial/decisao_inicial_especialista.md --nome_saida prompt_especialista_decisao_inicial_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"
    "main.py --task decisao_inicial --data_dir ./data/decisao_inicial/dados_json/ --prompts ./prompts/decisao_inicial/decisao_inicial_promptbreeder.md --nome_saida prompt_promptbreeder_decisao_inicial_evaluator_bf --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl"

    # ucb
    "main.py --task replica --data_dir ./data/replica/dados_json/ --prompts ./prompts/replica/replica_especialista.md --nome_saida prompt_especialista_replica_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"
    "main.py --task replica --data_dir ./data/replica/dados_json/ --prompts ./prompts/replica/replica_promptbreeder.md --nome_saida prompt_promptbreeder_replica_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"
    "main.py --task decisao_inicial --data_dir ./data/decisao_inicial/dados_json/ --prompts ./prompts/decisao_inicial/decisao_inicial_especialista.md --nome_saida prompt_especialista_decisao_inicial_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"
    "main.py --task decisao_inicial --data_dir ./data/decisao_inicial/dados_json/ --prompts ./prompts/decisao_inicial/decisao_inicial_promptbreeder.md --nome_saida prompt_promptbreeder_decisao_inicial_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"
    "main.py --task despacho --data_dir ./data/despacho_saneador/dados_json/ --prompts ./prompts/despacho_saneador/despacho_saneador_promptbreeder.md --nome_saida prompt_promptbreeder_despacho_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"
    #"main.py --task despacho --data_dir ./data/despacho_saneador/dados_json/ --prompts ./prompts/despacho_saneador/despacho_saneador_especialista.md --nome_saida prompt_especialista_despacho_evaluator_ucb --out ./data/experimentos_prof_vasco --engine gpt --max_threads 7 --name_train_data training_data.jsonl --name_test_data test_data.jsonl --evaluator ucb"

)

# Executar os comandos sequencialmente
for script in "${comandos[@]}"; do
    echo "Executando: python $script" | tee -a log_execucao.txt
    eval "python $script" | tee -a log_execucao.txt
done

# Manter o terminal aberto (para Git Bash ou WSL)
echo "Pressione qualquer tecla para sair..."
read -n 1 -s
