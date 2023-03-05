import numpy as np
import os

run = True
problem = 'CON'
learner = 'NGBlin'
acqfun  = 'AVG'
search  = 'MH'
epsilon = 0.1
n_exp   = np.arange(1,10)
seed    = 23

header = '''#!/bin/bash
#$ -q teano
#$ -pe smp 5
#$ -j yes
#$ -cwd

# Load anaconda malware environment
conda activate base

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
                f"--seed_conf {seed} --nexp {i} "
                f"> exp_{i}_{problem}_{learner}_o{search}_af{acqfun}.out")
        
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
