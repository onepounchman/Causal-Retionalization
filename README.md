# Towards Trustworthy Explanation: On Causal Rationalization

This repository contains the code for ICML 2023 paper **Towards Trustworthy Explanation: On Causal Rationalization**.

## Getting Started

Firstly, create the python environment and activate it

```sh
conda create --name pytorch_py38 python=3.8

source activate pytorch_py38
```

To install the dependencies, run the following command
```sh
cd rationale-causal

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
We thank the authors of [Can Rationalization Improve Robustness?](https://github.com/princeton-nlp/rationale-robustness) for their implementations of the baseline methods