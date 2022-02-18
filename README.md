# OPIRL: Sample  Efficient Off-Policy Inverse  Reinforcement  Learning  via Distribution  Matching

Official implementation for [OPIRL: Sample Efficient Off-Policy Inverse Reinforcement Learning via Distribution Matching](https://arxiv.org/abs/2109.04307).  
Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2022.

## Installation

Run the following command to install all Python dependencies:
```
$ pip install -e .
$ pip install -r requirements.txt
```

Other dependencies:
- Python 3.8+
- TensorFlow 2.4+
- CUDA=11.0
- cuDNN=8.0

- Experts/reward functions are provided on [Google Drive](https://drive.google.com/file/d/1Hq5Iu8oMvA9bx_fvrUmburtevjLFhKit/view?usp=sharing)

## Run Experiments

First, unzip the expert/reward files from Google Drive.  
Then, to simply run experiments on MuJoCo tasks, run the bash scripts in `/scripts` directory.   
E.g.
```
$ sh ./scripts/run_halfcheetah.sh
```

## Others

- OPOLO: [code](https://github.com/illidanlab/opolo-code)
- f-IRL: [code](https://github.com/twni2016/f-IRL)
