# RLOMM

SIGMOD‘25 RLOMM: An Efficient and Robust Online Map Matching Framework with Reinforcement Learning

## Preparation
```bash
bash data_process.sh
```

## Environment Requirement

The code runs well under python 3.11.5. The required packages are as follows:

- pytorch==2.0.1
- numpy==1.25.2
- networkx==3.1.0
- pickle, pandas, argparse


## Usage
```bash
python main.py --config config/beijing_config.json --gpus 0
python main.py --config config/porto_config.json --gpus 0
python main.py --config config/chengdu_config.json --gpus 0
```

## Citation
```
@inproceedings{Chen2025RLOMM,
  author       = {Minxiao Chen and
                  Haitao Yuan and
                  Nan Jiang and
                  Zhihan Zheng and
                  Sai Wu and
                  Ao Zhou and
                  Shangguang Wang},
  title        = {{RLOMM:} An Efficient and Robust Online Map Matching Framework with
                  Reinforcement Learning},
  booktitle    = {SIGMOD},
  volume       = {3},
  number       = {3},
  pages        = {209:1--209:26},
  year         = {2025},
}
```
