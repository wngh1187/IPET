# IPET

Pytorch code for following paper:

* **Title** : INTEGRATED PARAMETER-EFFICIENT TUNING FOR GENERAL-PURPOSE AUDIO MODELS (Submit to ICASSP2023, now available [here](  )) 
* **Autor** : Ju-ho Kim<sup>\*</sup>, Jungwoo Heo<sup>\*</sup>, Hyun-seo Shin, Chan-yeong Lim, and Ha-Jin Yu

# Abstract
<img align="middle" width="2000" src="https://github.com/wngh1187/IPET/blob/main/overall.png">

The advent of hyper-scale and general-purpose pre-trained models is shifting the paradigm of building task-specific models for target tasks. 
In the field of audio research, task-agnostic pre-trained models with high transferability and adaptability have achieved state-of-the-art performances through fine-tuning for downstream tasks. 
Nevertheless, re-training all the parameters of these massive models entails an enormous amount of time and cost, along with a huge carbon footprint. 
To overcome these limitations, the present study explores and applies efficient transfer learning methods in the audio domain. 
We also propose an integrated parameter-efficient tuning (IPET) framework by aggregating the embedding prompt (a prompt-based learning approach), and the adapter (an effective transfer learning method). 
We demonstrate the efficacy of the proposed framework using two backbone pre-trained audio models with different characteristics: the audio spectrogram transformer and wav2vec 2.0. 
The proposed IPET framework exhibits remarkable performance compared to fine-tuning method with fewer trainable parameters in four downstream tasks: sound event classification, music genre classification, keyword spotting, and speaker verification. 
Furthermore, the authors identify and analyze the shortcomings of the IPET framework, providing lessons and research directions for parameter efficient tuning in the audio domain.

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:22.01-py3' image of Nvidia GPU Cloud for conducting our experiments. 
* We used four NVIDIA RTX A5000 GPUs for training. 
* Python 3.8.12
* Pytorch 1.11.0+cu115
* Torchaudio 0.11.0+cu115

See requirements.txt for details.

## Datasets

We used five dataset for training and test: ESC50, FSD50K, GTZAN, Speechcommands V2, VoxCeleb1. 
By copyright, please refer each dataset release pages. 
The training and evaluation (or validation also) data list is pre-built as a json file. 

# Training

```
Go to the desired directory
run the code below
python3 main.py -name [your exp name] -tags [your exp tags]
```


# Citation
Please cite this paper if you make use of the code. 

```

```
