# 🤖 EE-568-Deep-RL-Project

📚 Course project for **Reinforcement Learning** taught by _Volkan Cevher_ at EPFL.

🎯 **Goal**: Test state of the art algorithms on different environments in order to compare their optimization complexity and their stability in different seetings. This project aims to give researchers starting in RL an intuition about how to optimize each of these algorithms and what are their theoretical and practical specificcities.

### 🧠 Algorithms

- 🟦 DQN (Deep Q-Network)
- 🟠 PPO (Proximal Policy Optimization)
- 🧊 SAC (Soft Actor-Critic)
- ⚡ TD3 (Twin Delayed Deep Deterministic Policy Gradient)

---

### 🌍 Environments

- CartPole-v1
- MountainCar-v0
- Acrobot-v1
- Pendulum-v1

### ⚙️ Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### 🏃‍♂️ Usage

First, make sure you set the hyperparameters ranges in `algorith_hyperparams.json` for the algorithm you want to run.
Then, run the training and hyperparameter optimization script for a specific algorithm and environment:

```bash
python hyperparam_optimizer.py --algorithm <ALGORITHM> --env <ENVIRONMENT> --total_timesteps <TIMESTEPS>
```

You can follow the training progress through tensorboard by running:

```bash
tensorboard --logdir logs
```

### Visualization

To visualize the model on the environment, use:

```bash
python test_model.py --algorithm <ALGORITHM> --env <ENVIRONMENT> --model_path <MODEL_PATH>
```

### 👥 Members of the team Random_hyperparameters_generator

- Adam Mesbahi: PPO, report
- Aziz Sebbar: Poster, report
- Hassen Aissa: Optimization in optuna, TD3, report
- Matya Aydin: Plots, SAC, report
- Mehdi Zoghlami: Poster, report
- Yassine Turki: DQN, report

---
