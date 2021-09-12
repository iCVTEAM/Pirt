# Pirt
Pose-guided Inter- and Intra-part Relational Transformer for Occluded Person Re-Identification official implement

The code will be released soon.

## Introduction

This repository contains the code for the paper:
[**Pose-guided Inter- and Intra-part Relational Transformer for Occluded Person Re-Identification**](https://arxiv.org/abs/2109.03483)
Zhongxing Ma, Yifan Zhao, Jia Li
ACM Conference on Multimedia (ACM MM), 2021

## Environments

1. pytorch 1.6.0
2. python  3.8
3. pyyaml | yacs | termcolor | tqdm | faiss-cpu | tabulate | tabulate | matplotlib | tensorboard
4. sklearn | enopis

## Getting Started

Working directory: **/your/path/to/fast-reid/**

### Traning

```bash
python -u tools/train_net.py --config-file configs/Pirt.yml --num-gpus 2 OUTPUT_DIR logs/your/customed/path
```

### Evaluation

```bash
python -u tools/train_net.py --eval-only --num-gpus 2 configs/eval.yml OUTPUT_DIR logs/your/customed/path
```

The config file of the model are placed at `./configs/Pirt.yml`

### Datasets

OccludedDuke or Market-1501 datasets shoule be placed at `./datasets/OccludedDuke`

See the `./fastreid/data/datasets` folder for detailed configuration

### Pretrained Models

The pose-guided and resnet50 models should be placed at `../models_zoo/`

### Citing

```bash
@misc{ma2021poseguided,
      title={Pose-guided Inter- and Intra-part Relational Transformer for Occluded Person Re-Identification}, 
      author={Zhongxing Ma and Yifan Zhao and Jia Li},
      year={2021},
      eprint={2109.03483},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

Our code is based on the early version of [**FAST-REID**](https://github.com/JDAI-CV/fast-reid).
