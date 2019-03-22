# MINE.pytorch

Mutual Information Neural Estimation

To be merge to [homura](https://github.com/moskomule/homura)

## Requirements

Python>=3.7
PyTorch>=1.0.1
miniargs>=0.0.1

## Experiments

* Mutual information estimation of a pair of samples from multivariate gaussian
    * `python multi_gaussian.py`

* Mutual information is invariant to invertible transformation
    * `python invariance.py [--function x orx^3 or sin(x)]`
    