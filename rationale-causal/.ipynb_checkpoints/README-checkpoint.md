# Towards Trustworthy Explanation: On Causal Rationalization

This repository contains the code for ICML 2023 paper [Towards Trustworthy Explanation: On Causal Rationalization](https://arxiv.org/pdf/2204.11790.pdf).

## Getting Started

Firstly, create the python enviroment

```sh
conda create --name pytorch_py38 python=3.8
```
To install the dependencies, run following commands
```sh
# Install all python dependencies
pip install -r requirements.txt
```

## Experiments

You can download Beer and Hotel review dataset from https://github.com/YujiaBao/R2A and then put datsets in the data folder.

Before running experiments for Beer review data, utilize cr/data-processing.ipynb to get the short and the noise version data.

Training commands for causal rationalization method:

```sh
# real data
./scripts/beer/run_beer_aroma.sh causalVIB

# synthetic data
./scripts/beer_noise/run_beer_aroma.sh causalVIB
```

## Acknowledgment
We thank the authors of [Can Rationalization Improve Robustness?](https://github.com/princeton-nlp/rationale-robustness) for their implementations