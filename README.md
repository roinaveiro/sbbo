# Simlation Based Bayesian Optimization

This repository contains the code needed to reproduce all the experiments in the "Simulation Based Bayesian Optimization" paper.

The code is written Python. A conda environment contains all necessary dependencies. It can be installed using

`conda env create -f sbbo.yml`

And activated throught 

`conda activate sbbo`

In addition, the `sbbo` package must be installed running the following in the root directory:

`pip install -e .`
 
## Preprocessing

```{bash}
python -u run.py --problem {problem}
                 --learner {learner} --acqfun {acqfun}
                 --niters 500 --search {search}
                 --epsilon {epsilon}
                 --seed_conf {seed[i]} --nexp {i} 
```
