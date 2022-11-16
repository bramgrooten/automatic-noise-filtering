# Automatic Noise Filtering 
_with Dynamic Sparse Training in Deep Reinforcement Learning_

# Abstract
Tomorrow's robots will need to distinguish useful information from noise when performing different tasks. 
A household robot for instance may continuously receive a plethora of information about the home, 
but needs to focus on just a small subset to successfully execute its current chore.

Filtering distracting inputs that contain irrelevant data 
has received little attention in the reinforcement learning literature. 
To start resolving this, we formulate a **problem setting** in reinforcement learning 
called the _extremely noisy environment_ (ENE) where up to 99% of the input features are pure noise.
Agents need to detect which features actually provide task-relevant information 
about the state of the environment. 

Consequently, we propose a new **method** termed _Automatic Noise Filtering_ (ANF) 
which uses the principles of dynamic sparse training to focus the input layer's connectivity 
on task-relevant features.
ANF outperforms standard SAC and TD3 by a large margin, while using up to 95% fewer weights.

Furthermore, we devise a transfer learning setting for ENEs, 
by permuting all features of the environment after 1M timesteps 
to simulate the fact that other information sources can become task-relevant as the world evolves. 
Again ANF surpasses the baselines in final performance and sample complexity. 



# Install
### Requirements
* Python 3.8
* PyTorch 1.9
* [MuJoCo-py](https://github.com/openai/mujoco-py) 
* [OpenAI gym](https://github.com/openai/gym)
* Linux (using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) may work in Windows)

### Instructions 
First make a virtual environment:
```shell
python -m venv venv
source venv/bin/activate
```

If you don't have MuJoCo yet:
```shell
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
mv mujoco210 .mujoco/
rm mujoco210-linux-x86_64.tar.gz
```

Now you have MuJoCo. Proceed with:
```shell
pip install mujoco_py==2.1.2.14 gym==0.21.0 torch==1.9.0
pip install wandb --upgrade
```
To run experiments without [W&B](https://wandb.ai/site), call this first: `wandb disabled` (before each run).
To use W&B, call this first: `wandb login` (just once).


Now try to import mujoco_py in a python console, 
and do what the error messages tell you. 
(Like adding lines to your `.bashrc` file.)
```python
$ python
>>> import mujoco_py
```


# Usage

### Train
To train an ANF agent on the ENE with 90% noise features, run:
```
python main.py --policy ANF-SAC --env HalfCheetah-v3 --fake_features 0.9
```

Possible policies: `ANF-SAC`, `ANF-TD3`, `SAC`, `TD3`.

Possible environments: `HalfCheetah-v3`, `Hopper-v3`, `Walker2d-v3`, `Humanoid-v3`.

Show all available arguments: `python main.py --help`

### Test

See the file `view_mujoco.py` to test a trained agent on a single episode and view its behavior.



