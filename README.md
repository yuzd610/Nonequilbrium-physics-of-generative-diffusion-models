# Nonequilbrium physics of generative diffusion models

## Overview
- This repository provides the code for the framework presented in [Nonequilbrium physics of generative diffusion models](https://arxiv.org/abs/2405.11932):
- In this paper, we provide a transparent physics analysis of the diffusion
models, deriving the fluctuation theorem, entropy production, Franz-Parisi potential to understand
the intrinsic phase transitions discovered recently. Our analysis is rooted in non-equlibrium physics
and concepts from equilibrium physics, i.e., treating both forward and backward dynamics as a
Langevin dynamics, and treating the reverse diffusion generative process as a statistical inference,
where the time-dependent state variables serve as quenched disorder studied in spin glass theory.

## Requirements

To run the code, you need the following dependencies installed:

- Python 3.11
- matplotlib-base 3.8.4
- numpy 1.24.3
- tqdm 4.66.2
- scipy 1.10.1
- joblib 1.4.0

## Installation
You can install these dependencies using conda:
- conda install  matplotlib-base==3.8.4 numpy==1.24.3 tqdm==4.66.2 scipy==1.10.1 joblib==1.4.0 -c pytorch

## Usage

### Diffusion model schematic diagram
- The code (`fb.py`) can be run directly.
- Diagram of a 2D Gaussian Mixture Diffusion Model.

### Ensemble Statistics
- The scripts (`Eforward.py`) and (`Ebackward.py`) can be run directly.
- (`Eforward.py`) generates ensemble statistics for the forward process, while (`Ebackward.py`) generates ensemble statistics for the backward process.
- The code only works for two Gaussian mixtures with equal weights and symmetric means about the origin.

#### The files (`Eforward.py`) and (`Ebackward.py`) include the following input arguments:
- `n` represents the number of Monte Carlo samples.
- `pi1` denotes the weight of one Gaussian component in the mixture.
- `pi2` denotes the weight of another Gaussian component in the mixture.
- `mu1` represents the mean of one Gaussian component in the mixture.
- `mu2` represents the mean of another Gaussian component in the mixture.
- `sigma_2` denotes the variance of a single Gaussian distribution in the mixture.
- `num` specifies the number of data points to plot for Ensemble Statistics.
- `T` represents the total runtime.

### FP potential
- The code (`fb.py`) can be run directly.
- Plotted the FP potential energy diagram in the article.
#### The file (`fb.py`)  include the following input arguments:
- `mu`  represents the absolute value of the mean in one Gaussian component in the mixture.
- `sigma2`  denotes the variance of a single Gaussian distribution in the mixture.
- `t` FP potential time point
- `num` denotes the number of iterations.



### Histogram of trajectory statistics
- The scripts (`Hforward.py`) and (`Hbackward.py`) can be run directly.
- (`Hforward.py`) generates trajectory statistics for the forward process, while (`Hbackward.py`) generates trajectory statistics for the backward process.
-  The code only works for two Gaussian mixtures with equal weights and symmetric means about the origin.
#### The files (`Hforward.py`) and (`Hbackward.py`) include the following input arguments:
- `n` represents the number of trajectories.
- `pi1` denotes the weight of one Gaussian component in the mixture.
- `pi2` denotes the weight of another Gaussian component in the mixture.
- `mu1` represents the mean of one Gaussian component in the mixture.
- `mu2` represents the mean of another Gaussian component in the mixture.
- `sigma_2` denotes the variance of a single Gaussian distribution in the mixture.
- `ds` represents the discrete time step size.
- `Iterate` denotes the number of iterations.

### Numerical verification of the integral fluctuation theorem
- The scripts (`forward_n.py`) and (`backward_n.py`) can be run directly.
- (`forward_n.py`) verify positive integral fluctuations for the forward process, while  (`backward_n.py`) verify positive integral fluctuations for the backward process.
- This code is designed to work only if the Gaussian distributions in the mixture have unit variance and works for two Gaussian mixtures with equal weights and symmetric means about the origin.
#### The files (`forward_n.py`) and (`backward_n.py`) include the following input arguments:
- `pi1` denotes the weight of one Gaussian component in the mixture.
- `pi2` denotes the weight of another Gaussian component in the mixture.
- `mu1` represents the mean of one Gaussian component in the mixture.
- `mu2` represents the mean of another Gaussian component in the mixture.
- `ds` represents the discrete time step size.
- `Iterate` denotes the number of iterations.

### Potential Energy
- The scripts (`Potential Energy.py`)  can be run directly.

## Citation
This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.

## Contact
If you have any question, please contact me via yuzd610@163.com.





