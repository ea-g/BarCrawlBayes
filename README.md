# BarCrawlBayes

[Training Logs](https://wandb.ai/ea-g/BarCrawlBayes?workspace=user-ea-g)

## Prerequisites
- [poetry](https://python-poetry.org/)
- Python ~3.9
- a gpu or google colab
- g++ for pymc or google colab
- a [Weights and Biases account](https://wandb.ai/) 

## Setup 

1. download the barcrawl data set from uci ml and extract the contents into the data folder. Your directory structure 
should follow:

```
BarCrawlBayes (project root)
| data
| | clean_tac
| | raw_tac
| | ...
```

2. Install all dependencies for the project:

```commandline
poetry install
```

3. prepare the data

```commandline
python dataprep.py
```

4. create a `.env` file with your [weights and biases token](https://wandb.ai/authorize):

```
# content of your .env file with weights and biases key
WBKEY="whatever your key happens to be here!"
```

## Contents

- `trainreg.py`: training script for frequentist neural networks. 
This was operated via [wandb param sweeps](https://docs.wandb.ai/guides/sweeps). Note that it parameter sweeps must be 
setup separately using the sweep configs located in `configs`. Once setup, they can be run in colab such as in 
[this notebook](https://colab.research.google.com/drive/13xQ6-OWeqnQimDz5b0o_Bb08ak3zguWL?usp=sharing). 
- `trainbayes.py`: training script for Bayesian neural networks. This script finetunes FNNs into BNNs initializing 
variational distributions based on the pretrained weights from FNNs. Paths per network will need to be edited in the 
file. For an example run, see [this notebook](https://colab.research.google.com/drive/1fiYxo1FXdUPLf8GzyRQloaSg0ts_5GTV?usp=sharing)
- `eval.py`: scores models and plots metrics
- `bayesian_anova.py`: simple ANOVA bayesian style for comparing the spread of samples from the posterior predictive 
distributions.

## Disclaimer

Original ResNet1D implementation is from [Shanda Hong](https://github.com/hsd1503/resnet1d). Please cite if you use it:

```
@inproceedings{hong2020holmes,
  title={HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units},
  author={Hong, Shenda and Xu, Yanbo and Khare, Alind and Priambada, Satria and Maher, Kevin and Aljiffry, Alaa and Sun, Jimeng and Tumanov, Alexey},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1614--1624},
  year={2020}
}
```
