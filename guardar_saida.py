import json
import os
from datetime import datetime


path_root_file = os.path.dirname(os.path.abspath(__file__))

def salvar_saida(path='./data/resultado', prompt=None, pred=None, texto=None) -> None:
    dic = {
        'prompt' : prompt,
        'pred' : pred,
        'texto' : texto,
        'data' : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        
    }