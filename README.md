# IPET

Pytorch code for following paper:

* **Title** : INTEGRATED PARAMETER-EFFICIENT TUNING FOR GENERAL-PURPOSE AUDIO MODELS (Submit to ICASSP2023, now available [here]( https://arxiv.org/abs/2211.02227 )) 
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

# Hyper-parameters details
For each task and method, we conducted a grid search of the hyper-parameters.
The table below describes the hyper-parameters determined based on the best performance.

| Model |      Dataset      | Batch size | Input frame | Specaugment(time / frequency) | Mixup | Gaussian noise | MUSAN augmentation |  Learning rate for method | \# of embedding prompts | \# of adapter dimensions |
|:-----:|:-----------------:|:----------:|:-----------:|:-----------------------------:|:-----:|:--------------:|:---------------------------:|:----------------------:|:------------------:|
|  AST  |       ESC50       |     48     |     512     |            96 / 24            |   X   |        X       |        X       | FT: $1e^{-5}$ / IPET: $1e^{-3}$ |            4            |         32         |
|  AST  |       FSD50K      |     24     |     1024    |            192 / 48           |  0.5  |        X       |        X       | FT: $1e^{-5}$ / IPET: $1e^{-3}$ |            32           |         32         |
|  AST  |       GTZAN       |     32     |     400     |            80 / 48            |  0.3  |        X       |        X       | FT: $5e^{-5}$ / IPET: $4.5e^{-3}$ |            8            |         64         |
|  AST  | Speech Command V2 |     128    |     128     |            48 / 48            |  0.5  |        O       |        X       | FT: $2.5e^{-4}$ / IPET: $5e^{-3}$ |            4            |         128        |
|  AST  |     VoxCeleb1     |     32     |     400     |            80 / 48            |   X   |        O       |        X       | FT: $5e^{-5}$ / IPET: $5e^{-4}$ |            4            |         64         |
|  W2V2 |       ESC50       |     48     |     512     |               X               |   X   |        X       |        X       | FT: $5e^{-5}$ / IPET: $2.5e^{-3}$ |           128           |         128        |
|  W2V2 |       FSD50K      |     24     |     1024    |               X               |  0.5  |        X       |        X       | FT: $5e^{-5}$ / IPET: $5e^{-3}$ |           128           |         64         |
|  W2V2 |       GTZAN       |     32     |     400     |               X               |  0.3  |        X       |        X       | FT: $2.5e^{-5}$ / IPET: $5e^{-3}$ |           128           |         64         |
|  W2V2 | Speech Command V2 |     64     |     128     |               X               |   X   |        X       |        X       | FT: $5e^{-5}$ / IPET: $1e^{-3}$ |            64           |         64         |
|  W2V2 |     VoxCeleb1     |     64     |     300     |               X               |   X   |        X       |        O       | FT: $2.5e^{-5}$ / IPET: $1e^{-3}$ |            64           |         128        |

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
@inproceedings{Kim2022IntegratedPT,
  title={Integrated Parameter-Efficient Tuning for General-Purpose Audio Models},
  author={Ju-ho Kim and Ju-Sung Heo and Hyun-seo Shin and Chanmann Lim and Ha-jin Yu},
  year={2022}
}

```
