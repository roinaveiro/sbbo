# Simlation Based Bayesian Optimization

This repository contains the code needed to reproduce all the experiments in the "Simulation Based Bayesian Optimization" paper.

The code is written Python. A conda environment contains all necessary dependencies. It can be installed using

`conda env create -f sbbo.yml`

And activated through 

`conda activate sbbo`

In addition, the `sbbo` package must be installed running the following in the root directory:

`pip install -e .`
 
## Running experiments

To run SBBO, execute the following:

```{bash}
python -u run.py --problem {problem}
                 --learner {learner} --acqfun {acqfun}
                 --niters {niters} --search {search}
                 --seed_conf {seed_conf} --nexp {nexp} 
```

The `--problem` argument selects the optimization problem in which SBBO will be run. Currently implemented options are: 

* `BQP`: Binary quadratic problem (Section 4.1 of the paper). 
* `CON`: Contamination problem (Section 4.2 of the paper). 
* `pRNA`: RNA design problem (Section 4.3 of the paper). 

The `--learner` argument selects the surrogate probabilistic model. Available options are: 

* `BOCS`: Sparse Bayesian linear regression with pairwise interactions.
* `BNN`:  Bayesian Neural Network.
* `GPr`:  Tanimoto Gaussian Process Regression.
* `NGBdec`: Natural gradient boosting with a shallow decision tree model as base learner.
* `NGBlinCV`: Natural gradient boosting with an sparse linear regression as base learner.

The `--acqfun` argument selects the acquisition function. Specify `EI` for expected improvement or `PI` for probability of improvement.
`--niters` specifies the number of function evaluations. `--search` selects the algorithm used to select the next function evaluation location. Possible options are:

* `MH`: to use SBBO with Metropolis-Hastings sampling scheme.
* `Gibbs`: to use SBBO with Gibbs sampling scheme.
* `RS`: to  use Random Local Search.
* `SA`: to use Simulated Annealing.

`--seed_conf` allows configuration of random seed. Set it to 23 in order to reproduce the experiments in the paper. Finally, use `--nexp` as a label for the experiment number.

After running the previous command with the selected options, results will be stored in the results folder.

In order to reproduce the convergence plot from the appendix, run the `convergence.ipynb` jupyter notebook located in the `notebooks` folder.

## Tables and visualizations

The code to reproduce tables an plots has been written in R. `tidyverse`, `kableExtra` and `latex2exp` libraries need to be installed.
First of all, results generated in the experiments of the previous section must be preprocessed running

```{bash}
Rscript preprocess_results.R
```

Then, the plots of the paper could be reproduced running

```{bash}
Rscript plots.R
```

Resulting images will be stored in the `figs` folder. 

Finally, tables can be reproduced running

```{bash}
Rscript tables.R
```

Latex code for each table will be printed.



