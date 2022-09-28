# Quantile Constrained Reinforcmet Learning

Code for the paper: "Quantile Constrained Reinforcement Learning: A Reinforcement Learning Framework Constraining Outage Probability," NeurIPS 2022.

Additional dependencies: OpenAI `gym` and `safety_gym` (including `mujoco`).

## Installation Guide

1. Create conda env 
``
conda create -n qcpo python=3.7
``
2. Activate conda env
``
conda activate qcpo
``
3. Install pytorch
``
conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly opencv==3.4.2 -c pytorch
``
4. Install other packages
``
pip install pyprind psutil
``
5. Install rlpyt from https://github.com/astooke/rlpyt (See detailed installation guide below)

6. Install safety-gym from https://github.com/openai/safety-gym (See detailed installation guide below)

7. Move the whole directory (named qcpo) of this file to rlpyt/rlpyt/projects



## Run Experiments
1. Activate conda env
``
conda activate qcpo
``
2. Move to rlpyt/rlpyt/projects/qcpo/experiments
``
cd rlpyt/rlpyt/projects/qcpo/experiments
``
3. Run the code by
``
python launch_qcpo.py
``
4. The default target_outage_probability and environment are set by
```python
target_prob = 0.2 # line 13
env_ids = [                 # line 29 ~ 34
    'SimpleButtonEnv-v0',
    # 'DynamicEnv-v0',
    # 'GremlinEnv-v0',
    # 'DynamicButtonEnv-v0',
]
```
You can change these as you want. For example, 
```python 
target_prob=0.1, env_ids = ['GremlinEnv-v0'] 
``` 

## Detailed Installation Guide for rlpyt
1. Clone the git repository
``
git clone https://github.com/astooke/rlpyt.git
``
2. Move to safety-gym directory
``
cd rlpyt
``
3. Install rlpyt as editable python package
``
pip install -e .
``

## Detailed Installation Guide for safety-gym
1. Clone the git repository
``
git clone https://github.com/openai/safety-gym.git
``
2. Move to safety-gym directory
``
cd safety-gym
``
3. Install safety-gym 
``
pip install -e .
``
4. It may reinstall numpy with lower version. So remove the lower versioned numpy and reinstall the newest one
``
pip uninstall numpy &
pip install numpy
``

