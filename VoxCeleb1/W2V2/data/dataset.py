import csv
import json
from dataclasses import dataclass

@dataclass
class DataItem:
    path: str
    label: str

@dataclass
class TestTrial:
    key1: str
    key2: str
    label: str
    
class Datasets:
    @property
    def train_set(self):
        return self.__train_set

    @property
    def classes_labels(self):
        return self.__classes_labels

    @property
    def evaluation_set(self):
        return self.__evaluation_set

    @property
    def test_trials(self):
        return self.__test_trials

    def __init__(self, args):
        # train_set
        self.__train_set = [] 
        with open(args.path_training_dataset, 'r') as fp:
            data = json.load(fp)['data']    
        for d in data:
            self.__train_set.append(
                DataItem(
                    path=d['wav'],
                    label=d['labels']
                )
            )
                
        classes_labels = {}
        with open(args.path_data_label, 'r') as f:
            csv_reader = csv.DictReader(f)
            line_count = 0
            for row in csv_reader:
                classes_labels[row['mid']] = row['index']
                line_count += 1
        self.__classes_labels = classes_labels

        # eval_set
        self.__evaluation_set = []        
        with open(args.path_evaluation_dataset, 'r') as fp:
            data = json.load(fp)['data']    
        for d in data:
            self.__evaluation_set.append(
                DataItem(
                    path=d['wav'],
                    label=d['labels']
                )
            )
        self.__test_trials = self._parse_trials(args.path_trial)

    def _parse_trials(self, path):
        trials = []

        f = open(path) 
        for line in f.readlines():
            strI = line.split(' ')
            trials.append(
                TestTrial(
                    key1=strI[1].replace('\n', ''),
                    key2=strI[2].replace('\n', ''),
                    label=strI[0]
                )
            )
        
        return trials
