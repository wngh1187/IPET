import math
from random import random
import torch
import torch.utils.data as td
import torchaudio as ta
import numpy as np
import warnings
import torch.nn.functional
import utils.util as util
import random 
import soundfile as sf

from .augmentation.musan import Musan
from .augmentation.reverberation import RIRReverberation

def get_loaders(args, dataset):
    train_set = TrainSet(args, dataset)
    train_set_sampler = td.DistributedSampler(train_set, shuffle=True)
    
    evaluationt_set = TestSet(args, dataset.evaluation_set, dataset.classes_labels)
    evaluationt_set_sampler = td.DistributedSampler(evaluationt_set, shuffle=False)

    train_loader = td.DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=train_set_sampler
    )

    evaluation_loader = td.DataLoader(
        evaluationt_set,
        batch_size=args.batch_size * 2,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=evaluationt_set_sampler
    )
    
    return train_set, train_set_sampler, train_loader, evaluationt_set, evaluation_loader

class TrainSet(td.Dataset):

    def __init__(self, args, dataset):
        self.args = args
        self.musan = Musan(args.path_musan)
        self.rir = RIRReverberation(args.path_rir)
        self.dataset = dataset
        self.items = dataset.train_set
        self.crop_size = args.frame_length * 160      
        
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        
        # read wav
        info = sf.info(item.path)
        wav_size = int(info.samplerate * info.duration)
        if wav_size <= self.crop_size:
            index = 0
        else:
            index = random.randint(0, wav_size - self.crop_size)
        audio, _ = sf.read(item.path, start=index, stop=index + self.crop_size)
        
        # random crop
        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        mix_audio = None
        mix_lambda = 0

        if random.random() < self.args.mixup:
            mix_sample_idx = random.randint(0, len(self.items)-1)
            mix_item = self.items[mix_sample_idx]
    
            info = sf.info(mix_item.path)
            wav_size = int(info.samplerate * info.duration)
            if wav_size <= self.crop_size:
                index = 0
            else:
                index = random.randint(0, wav_size - self.crop_size)
            mix_audio, _ = sf.read(mix_item.path, start=index, stop=index + self.crop_size)
            
            # random crop
            if mix_audio.shape[0] < self.crop_size:
                shortage = self.crop_size - mix_audio.shape[0]
                mix_audio = np.pad(mix_audio, (0, shortage), 'wrap')
            
            mix_lambda = np.random.beta(10, 10)

            audio = mix_lambda * audio + (1 - mix_lambda) * mix_audio
        

        if mix_lambda == 0:
            label_indices = np.zeros(self.args.num_classes)
            
            for label_str in item.label.split(','):    #one_hot for multi label classification (speech, audioset)
                label_indices[int(self.dataset.classes_labels[label_str])] = 1.0    

            label_indices = torch.FloatTensor(label_indices) 
        else: 
            label_indices = np.zeros(self.args.num_classes)
            # add sample 1 labels
            for label_str in item.label.split(','):
                label_indices[int(self.dataset.classes_labels[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_item.label.split(','):
                label_indices[int(self.dataset.classes_labels[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        
        # data augmentation
        aug_type = random.randint(0, 5)
        if aug_type == 0:
            pass
        elif aug_type == 1:
            audio = self.rir(audio)
        elif aug_type == 2:
            audio = self.musan(audio, 'speech')
        elif aug_type == 3:
            audio = self.musan(audio, 'music')
        elif aug_type == 4:
            audio = self.musan(audio, 'noise')
        elif aug_type == 5:
            audio = self.musan(audio, 'speech')
            audio = self.musan(audio, 'music')

        return audio, label_indices
            


class TestSet(td.Dataset):
    
    def __init__(self, args, dataset, classes_labels):
        self.args = args
        self.items = dataset
        self.classes_labels = classes_labels
        self.crop_size = args.frame_length * 160

    def __len__(self):
        return len(self.items)


    def __getitem__(self, index):
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)

        # stack
        buffer = []
        indices = np.linspace(0, audio.shape[0] - self.crop_size, self.args.nb_seg)
        for idx in indices:
            idx = int(idx)
            buffer.append(audio[idx:idx + self.crop_size])
        buffer = np.stack(buffer, axis=0)

        label_indices = np.zeros(self.args.num_classes)
        
        for label_str in item.label.split(','):    #one_hot for multi label classification (speech, audioset)
            label_indices[int(self.classes_labels[label_str])] = 1.0    

        label_indices = torch.FloatTensor(label_indices) 

        return buffer, label_indices, item.path