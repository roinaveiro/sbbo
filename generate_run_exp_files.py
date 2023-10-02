import numpy as np
import os

run     = True
problem = 'BQP'
learner = 'BNN'
acqfun  = 'EI'
search  = 'MH'
epsilon = 0.0
n_exp   = np.arange(0,10)
#n_exp   = [11]
seed    = 23*np.ones(21).astype(int) 

header = '''#!/bin/bash
#$ -q "*@teano12.icmat.es"
#$ -pe smp 16
#$ -j yes
#$ -cwd

# Load anaconda environment
conda-init
conda activate tf-sbbo

# Run the executable
'''

tail = '''

# Deactivate anaconda environment
conda deactivate
'''

text = []
if seed is not None:

    for i in n_exp:
        current = (f"python -u run.py --problem {problem} "  
                f"--learner {learner} --acqfun {acqfun} " 
                f"--niters 500 --search {search} " 
                f"--epsilon {epsilon} " 
                f"--seed_conf {seed[i]} --nexp {i} "
                f"> exp_{i}_{problem}_{learner}_o{search}_af{acqfun}_seed{seed[i]}.out")
        
        current = header + current + tail
        
        with open(f'experiment{i}.sub', 'w') as f:
            f.write(current)
else:

    for i in n_exp:
        current = (f"python -u run.py --problem {problem} "  
                f"--learner {learner} --acqfun {acqfun} " 
                f"--niters 500 --search {search} " 
                f"--epsilon {epsilon} --nexp {i} " 
                f"> exp_{i}_{problem}_{learner}_o{search}_af{acqfun}.out")
        
        current = header + current + tail
        
        with open(f'experiment{i}.sub', 'w') as f:
            f.write(current)




if run:
    for i in n_exp:
        command = f"qsub experiment{i}.sub"
        os.system(command)
