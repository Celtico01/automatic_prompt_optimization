# Gradient-Based Automatic Prompt Optimization for Brazilian Document Classification

## Inspired By: 
<p>https://github.com/microsoft/LMOps/tree/main/prompt_optimization</p>

## Tested Python version:
<p>3.13.0</p>

## Explicação das Flags
<p>
task : Nome da Tarefa a ser feita (str) \n
data_dir : Diretorio onde fica os dados para treino e teste (str)
prompts : Caminho para o arquivo onde fica o prompt (str)
out : Diretorio de saida. (str)
max_threads : Usado para gerenciar as tarefas, quanto mais melhor(no sentido de velocidade) (int)
temperature : Temperature do modelo da OpenAI (float)
optimizer : Nome do Optimizer a ser usado (str)
rounds : Quantidade de iterações do algoritmo (int) 
beam_size : Qunatidade de candidatos (int)
n_test_exs : Quantidade de exemplos usados para teste
minibatch_size : Quantidade de minibatchs usados para criar os gradientes
n_gradients : Quantidade de gradientes
errors_per_gradient : Quantos error_strings são produzidos por gradientes (int)
gradients_per_error : Quantos gradientes são produzidos por error_strings (int) 
steps_per_gradient : Quantidades de prompts produzidos por gradientes (int)
mc_samples_per_step : Quantidade de sinônimos por prompt (int)
max_expansion_factor : Controla a quantidade de prompts que podem ser gerados a partir dos originais (int) 
engine : Não é utilizado! (str)
evaluator : Nome do Evaluator utilizado (str)
scorer : Nome do Scorer utilizado (str)
eval_rounds : Quantidades de rounds de avaliação(int)
eval_prompts_per_round : Quantidade de prompts avaliados por round (int )
samples_per_eval : Quantidade de exemplos por avaliação (usado apenas na UCB)  (int)
c : Parâmetro de exploração para o evaluator UCB, quanto maior mais exploração (float 0.0 - 1.0) 
"knn_k": 2, 
"knn_t": 0.993, 
reject_on_errors : Controla se o processo de expansão de novos prompts deve ser filtrado com base em exemplos de erro (bool)
eval_budget : Calculado pelo programa (config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']). Quantas amostras o sistema ira avaliar no total
</p>

