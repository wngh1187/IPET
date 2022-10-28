import csv
import json
from dataclasses import dataclass

@dataclass
class DataItem:
    path: str
    label: str

class Datasets:
    @property
    def train_set(self):
        return self.__train_set

    @property
    def classes_labels(self):
        return self.__classes_labels

    @property
    def validation_set(self):
        return self.__validation_set

    @property
    def evaluation_set(self):
        return self.__evaluation_set

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
        
        # val_set
        self.__validation_set = []
        with open(args.path_validation_dataset, 'r') as fp:
            data = json.load(fp)['data']    
        for d in data:
            self.__validation_set.append(
                DataItem(
                    path=d['wav'],
                    label=d['labels']
                )
            )

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
