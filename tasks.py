import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=4):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass




def process_example(ex, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            #print(len(futures))
            try:
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                    ex, pred = future.result()
                    texts.append(ex['text'])
                    labels.append(ex['label'])
                    preds.append(pred)
            except Exception as e:
                print(f"Erro ao processar um futuro: {e}")
                exit(e)

        f1_micro_positivo = f1_score(labels, preds, average='micro', labels=[1])
        f1_micro = f1_score(labels, preds, average='micro')
        
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')

        return f1_micro_positivo, f1_micro, f1_macro, f1_weighted, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1_micro_positivo, f1_micro, f1_macro, f1_weighted, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1_micro_positivo, f1_micro, f1_macro, f1_weighted, texts, labels, preds


class BinaryClassificationTask(ClassificationTask):
    categories = ['Não', 'Sim']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['Não', 'Sim']

    def get_train_examples(self, name_train_data='training_data.jsonl'):
        exs = []
        for i, row in enumerate(open(self.data_dir + name_train_data, encoding="utf-8")):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self, name_test_data='test_data.jsonl'):
        exs = []
        for i, row in enumerate(open(self.data_dir + name_test_data, encoding="utf-8")):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs
