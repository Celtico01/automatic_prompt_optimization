# Gradient-Based Automatic Prompt Optimization for Brazilian Document Classification

## Inspired By: 
<p>https://github.com/microsoft/LMOps/tree/main/prompt_optimization</p>

## Tested Python version:
<p>3.13.0</p>

## Explicação das Flags
<p>
task : Nome da Tarefa a ser feita (str)
data_dir : Diretorio onde fica os dados para treino e teste (str)
prompts : Caminho para o arquivo onde fica o prompt (str)
out : Diretorio de saida. (str)
max_threads : Usado para gerenciar as tarefas, quanto mais melhor(no sentido de velocidade) (int)
temperature : Temperature do modelo da OpenAI (float)
optimizer : Nome do Optimizer a ser usado (str)
rounds : Quantidade de iterações do algoritmo (int) 
"beam_size": 4, 
"n_test_exs": 400, 
"minibatch_size": 64, 
"n_gradients": 4, 
"errors_per_gradient": 4, 
"gradients_per_error": 1, 
"steps_per_gradient": 1, 
"mc_samples_per_step": 2, 
"max_expansion_factor": 8, 
"engine": "chatgpt", 
evaluator : Nome do Evaluator utilizado
scorer : Nome do Scorer utilizado
"eval_rounds": 8, 
"eval_prompts_per_round": 8, 
"samples_per_eval": 32, 
"c": 1.0, 
"knn_k": 2, 
"knn_t": 0.993, 
"reject_on_errors": false, 
"eval_budget": 2048}
</p>

