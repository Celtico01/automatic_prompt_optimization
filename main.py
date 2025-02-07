import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scores
import tasks
import predictors
import optimizers
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


def get_task_class(task_name):
    if task_name == 'despacho':
        return tasks.DefaultHFBinaryTask
    else:
        raise Exception(f'Tarefa não suportada: {task_name}')

def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Evaluator não suportado: {evaluator}')
    
def get_scorer(scorer):
    if scorer == '01':
        return scores.Cached01Scorer
    elif scorer == 'll':
        return scores.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Scorer não suportado: {scorer}')
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--prompts', default='')
    parser.add_argument('--nome_saida', default='t1', type=str)
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default='data/resultado')
    parser.add_argument('--max_threads', default=4, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    
    saida = {}

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    # to be removed
    #if os.path.exists(args.out):
        #os.remove(args.out)

    print(config)
    #to remove
    #with open(args.out, 'a') as outf:
    #    outf.write(json.dumps(config) + '\n')
    saida['config'] = config
    saida['modelo'] = os.getenv('MODEL')
    saida['max_tokens'] = os.getenv('MAX_TOKENS')
    saida['temperature'] = os.getenv('TEMPERATURE')

    candidates = [open(fp.strip(), 'r', encoding='utf-8').read() for fp in args.prompts.split(',')]

    saida['round0'] = {'hora_producao' : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                       'candidatos' : 'teste0',#candidates[0],
                       'scores' : None,
                       'f1' : None}
    cont=1
    for round in tqdm(range(config['rounds'] + 1)):
        candidato=1
        # TODO arrumar estrutura de saida...
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)
        print(1)
        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))
        print(2)
        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]
        print(3)
        saida['round'+str(cont)] = {'hora_producao' : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                    'candidatos' : candidates,
                                    'scores' : scores
                                    }
        cont+=1
        
        #to be removed
        # record candidates, estimated scores, and true scores
        #with open(args.out, 'a', encoding='utf-8') as outf:
        #    outf.write(f"======== ROUND {round}\n--------\n")
        #    outf.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n--------\n')
        #    outf.write(f'{candidates}\n--------\n')
        #    outf.write(f'{scores}\n--------\n')
        #print(4)
    
    # TODO : Testar apenas o melhor candidato e não todos! - DONE to be removed
    # TODO : Melhorar estrutura de saida - DONE to be removed
    
    # to remove
    #for candidate, score in zip(candidates, scores):
        #print(5)
        #f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
        #print(6)
        
    # to remove    
        #metrics.append(f1)
    #with open(args.out, 'a') as outf:  
        #outf.write(f'{metrics}\n--------\n')
    
    melhor_candidato = None
    melhor_score = float('-inf')
    melhor_round = 0
    for key, value in saida.items():
        if key not in ['round'+str(num) for num in range(0, config['rounds'])]:
            continue
        if key == 'round0':  # Ignorar o round 0
            continue

        candidatos_list = saida[key]['candidatos']
        scores_list = saida[key]['scores']

        print(candidatos_list)
        print(scores_list)
        
        if candidatos_list and scores_list:  # Verifica se não estão vazios
            max_index = scores_list.index(max(scores_list))  # Obtém o índice do maior score
            if scores_list[max_index] > melhor_score:  # Verifica se é o melhor score encontrado
                melhor_score = scores_list[max_index]
                melhor_candidato = candidatos_list[max_index]
                melhor_round = key

    print(f"Melhor candidato: {melhor_candidato} com score {melhor_score}")

    f1, texts, labels, preds = task.evaluate(gpt4, melhor_candidato, test_exs, n=args.n_test_exs)

    saida['melhor_candidato'] = melhor_candidato
    saida['melhor_score'] = melhor_score
    saida['melhor_round'] = melhor_round
    saida['f1_melhor'] = f1

    dict_final = {}
    dict_final[args.nome_saida] = saida

    with open(args.out + '/' + args.nome_saida + '.json', 'a+', encoding='utf-8') as file:
        json.dump(dict_final, file, ensure_ascii=False, indent=4)

    print("DONE!")
