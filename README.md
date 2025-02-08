# Gradient-Based Automatic Prompt Optimization for Brazilian Document Classification

## Inspired By
[LMOps - Prompt Optimization](https://github.com/microsoft/LMOps/tree/main/prompt_optimization)

## Tested Python Version
- Python 3.13.0

## Flags

- **task**: Nome da tarefa a ser realizada (str).  
- **data_dir**: Diretório onde os dados de treino e teste estão localizados (str).  
- **prompts**: Caminho para o arquivo contendo o prompt (str).  
- **out**: Diretório de saída para os resultados (str).  
- **max_threads**: Número máximo de threads utilizadas para gerenciar as tarefas. Maior valor pode acelerar o processo (int).  
- **temperature**: Temperatura do modelo da OpenAI (float).  
- **optimizer**: Nome do otimizador a ser utilizado (str).  
- **rounds**: Número de iterações do algoritmo (int).  
- **beam_size**: Número de candidatos a serem considerados por iteração (int).  
- **n_test_exs**: Quantidade de exemplos usados para teste (int).  
- **minibatch_size**: Quantidade de minibatches usados para criar os gradientes (int).  
- **n_gradients**: Quantidade de gradientes a serem gerados (int).  
- **errors_per_gradient**: Número de *error_strings* produzidos por gradiente (int).  
- **gradients_per_error**: Quantidade de gradientes gerados a partir de cada *error_string* (int).  
- **steps_per_gradient**: Número de prompts gerados por gradiente (int).  
- **mc_samples_per_step**: Quantidade de sinônimos gerados por prompt (int).  
- **max_expansion_factor**: Controla a quantidade máxima de prompts gerados a partir dos originais (int).  
- **engine**: Parâmetro não utilizado (str).  
- **evaluator**: Nome do avaliador utilizado (str).  
- **scorer**: Nome do avaliador de pontuação utilizado (str).  
- **eval_rounds**: Quantidade de rounds de avaliação (int).  
- **eval_prompts_per_round**: Número de prompts avaliados por round (int).  
- **samples_per_eval**: Número de exemplos usados por avaliação (apenas para UCB) (int).  
- **c**: Parâmetro de exploração para o avaliador UCB. Quanto maior, mais exploração do espaço de soluções (float, de 0.0 a 1.0).  
- **knn_k**: Número de vizinhos a ser considerado no KNN (int). *Não utilizado*.  
- **knn_t**: Limiar de confiança ou similaridade no KNN (float). *Não utilizado*.  
- **reject_on_errors**: Define se a expansão de novos prompts deve ser filtrada com base em exemplos de erro (bool).  
- **eval_budget**: Quantidade total de amostras avaliadas pelo sistema, calculado como `config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']` (int).  
