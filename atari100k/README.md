# Fork of CSR on the Atari environments
### Background
Atari100k Base code is taken from https://github.com/NM512/dreamerv3-torch

### To install
`pip install -r requirements.txt`

### To pretrain an agent on an environment
`python3 main.py --pretrain -env Pong -n_envs 8`

* `--pretrain` will target the pretraining
* `-n_evns` is the number of environments used by the actors ...

### To evaluate the world model
`python3 ...`

### To render the agent
`python3 ...`
