from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template
import utils
import tasks

class Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(Predictor):
    categories = ['Não', 'Sim']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        if self.opt['engine'] == 'gpt':
            response = utils.GPT(prompt)[0]
        elif self.opt['engine'] == 'deepseek':
            raise Exception('Não Implementado ainda')
        elif self.opt['engine'] == 'llama':            
            response = utils.LLAMA(prompt)[0]
        else:
            raise Exception('Opção Inválida.')
        #response = utils.chatgpt(
        #    prompt)[0]
        pred = 1 if response.strip().upper().startswith('SIM') else 0
        return pred
